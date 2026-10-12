import os,sys,json,time,base64,re,subprocess,traceback,gc
from pathlib import Path
B=Path(__file__).resolve().parent
R=B/"gguf-results"
W=B/"gguf-work"
MODEL="CADCODER/CAD-Coder"
REV="ff60c11d60e839c1da2c4fe445129df7dd89773e"
CONVERTER="9b61acf06041dcbaff6afa5f28940e93297f8520"
RUNTIME="f2cc7282ce0042641df55e0e563b75e824e2b647"
NAMES=("mounting-plate","stepped-flange")
PROMPT="Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words."
def save(name,value):
    R.mkdir(exist_ok=True); (R/name).write_text(json.dumps(value,indent=2))
def run(args,log,env=None):
    with (R/log).open("w") as f: subprocess.run([str(a) for a in args],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
def fixtures():
    import bench
    bench.OUT=R; bench.prepare()
def convert():
    import torch
    from safetensors import safe_open
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    torch.set_num_threads(2)
    W.mkdir(exist_ok=True)
    state={"model":MODEL,"revision":REV,"converter":CONVERTER,"runtime":RUNTIME,"stage":"download","engine":"llama.cpp Q4_K_M CPU"}
    save("conversion.json",state)
    api=HfApi(); info=api.model_info(MODEL,revision=REV)
    names=[s.rfilename for s in info.siblings if "/" not in s.rfilename and (s.rfilename.endswith(".safetensors") or s.rfilename in {"config.json","model.safetensors.index.json","tokenizer.model","tokenizer.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json"})]
    weights=Path(snapshot_download(MODEL,revision=REV,allow_patterns=names,local_dir=str(W/"weights"),max_workers=2))
    cfg=json.loads((weights/"config.json").read_text())
    assert cfg.get("mm_projector_type")=="mlp2x_gelu", "Unsupported projector"
    assert cfg.get("mm_vision_select_feature","patch")=="patch", "Unsupported feature selection"
    assert cfg.get("mm_vision_tower")=="openai/clip-vit-large-patch14-336", "Unexpected vision architecture"
    assert cfg.get("image_aspect_ratio","square") in ("square","pad"), "Unsupported image tiling"
    assert cfg.get("mm_patch_merge_type","flat")=="flat", "Unsupported patch merge"
    assert not cfg.get("mm_use_im_start_end",False), "Unsupported image boundary tokens"
    context=min(3072,int(cfg.get("max_position_embeddings",4096)))
    assert context>=1536
    state.update(context=context,max_tokens=min(2048,context-768))
    index=json.loads((weights/"model.safetensors.index.json").read_text())["weight_map"]
    vp="model.vision_tower.vision_tower."
    selected={k:v for k,v in index.items() if k.startswith(vp) or k.startswith("model.mm_projector.")}
    vision={}; projector={}
    for shard in sorted(set(selected.values())):
        with safe_open(str(weights/shard),framework="pt",device="cpu") as f:
            for key in f.keys():
                if key not in selected: continue
                t=f.get_tensor(key).float().clone()
                if key.startswith(vp): vision[key[len(vp):]]=t
                else: projector[key]=t
    assert len(vision)>300 and len(projector)==4, "Incomplete checkpoint vision/projector weights"
    assert list(projector["model.mm_projector.0.weight"].shape)==[cfg["hidden_size"],1024]
    vit=W/"vision"; vit.mkdir(exist_ok=True)
    torch.save(vision,vit/"pytorch_model.bin"); torch.save(projector,vit/"llava.projector")
    clip_info=api.model_info(cfg["mm_vision_tower"])
    clip_cfg=json.loads(Path(hf_hub_download(cfg["mm_vision_tower"],"config.json",revision=clip_info.sha)).read_text())
    vc=clip_cfg["vision_config"]
    vc["projection_dim"]=vc.get("projection_dim",clip_cfg.get("projection_dim",768))
    vc["architectures"]=["CLIPVisionModel"]
    vc["mm_vision_select_layer"]=cfg.get("mm_vision_select_layer",-2)
    vc["image_aspect_ratio"]=cfg.get("image_aspect_ratio","pad")
    (vit/"config.json").write_text(json.dumps(vc,indent=2))
    state.update(stage="conversion",vision_tensors=len(vision),projector_tensors=len(projector),vision_config_revision=clip_info.sha)
    save("conversion.json",state)
    del vision,projector,t; gc.collect()
    env={**os.environ,"PYTHONPATH":str(B/"converter/gguf-py")}
    run([sys.executable,B/"converter/tools/mtmd/convert_image_encoder_to_gguf.py","-m",vit,"--llava-projector",vit/"llava.projector","--clip-model-is-vision","--output-dir",W],"convert-vision.log",env)
    script=B/"converter/examples/convert_legacy_llama.py"
    helptext=subprocess.check_output([sys.executable,str(script),"--help"],env=env,stderr=subprocess.STDOUT,text=True)
    assert "--skip-unknown" in helptext and "--outfile" in helptext
    args=[sys.executable,script,weights,"--skip-unknown","--outtype","f16","--outfile",W/"language-f16.gguf"]
    if "--concurrency" in helptext: args += ["--concurrency","1"]
    run(args,"convert-language.log",env)
    run([B/"runtime/build/bin/llama-quantize",W/"language-f16.gguf",W/"language-q4.gguf","Q4_K_M","4"],"quantize.log")
    assert (W/"language-q4.gguf").stat().st_size>1024**3
    assert (W/"mmproj-model-f16.gguf").stat().st_size>100*1024**2
    state.update(stage="ready",language_bytes=(W/"language-q4.gguf").stat().st_size,vision_bytes=(W/"mmproj-model-f16.gguf").stat().st_size)
    save("conversion.json",state)
    (W/"language-f16.gguf").unlink()
def infer():
    import urllib.request
    settings=json.loads((R/"conversion.json").read_text())
    args=[str(B/"runtime/build/bin/llama-server"),"-m",str(W/"language-q4.gguf"),"--mmproj",str(W/"mmproj-model-f16.gguf"),"--host","127.0.0.1","--port","8080","--ctx-size",str(settings["context"]),"--parallel","1","--threads","4","--n-gpu-layers","0","--chat-template","vicuna"]
    logfile=(R/"server.log").open("w")
    p=subprocess.Popen(args,stdout=logfile,stderr=subprocess.STDOUT)
    try:
        ready=False; deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if p.poll() is not None: raise RuntimeError("Native server exited before becoming ready")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/health",timeout=5) as response:
                    ready=response.status==200
                if ready: break
            except Exception: pass
            time.sleep(1)
        if not ready: raise RuntimeError("Native server readiness timeout")
        save("inference.json",{"model":MODEL,"revision":REV,"backend":"llama.cpp","quantization":"Q4_K_M","runtime_revision":RUNTIME,"status":"generating","max_tokens":settings["max_tokens"],"seconds_per_case":900})
        for name in NAMES:
            d=R/name; started=time.monotonic(); rec={"status":"running","backend":"llama.cpp","quantization":"Q4_K_M","repair_attempts":0}
            (d/"generation.json").write_text(json.dumps(rec,indent=2))
            try:
                image="data:image/png;base64,"+base64.b64encode((d/"reference-iso.png").read_bytes()).decode()
                body={"messages":[{"role":"system","content":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},{"role":"user","content":[{"type":"image_url","image_url":{"url":image}},{"type":"text","text":PROMPT}]}],"temperature":0,"seed":7,"max_tokens":settings["max_tokens"],"stream":False}
                request=urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request,timeout=900) as response: result=json.load(response)
                (d/"response.json").write_text(json.dumps(result,indent=2))
                choice=result["choices"][0]; raw=choice["message"]["content"]
                (d/"raw.txt").write_text(raw)
                rec.update(finish_reason=choice["finish_reason"],usage=result.get("usage"))
                if choice["finish_reason"]!="stop": raise RuntimeError("Incomplete generation; not scored")
                code=re.sub(r"```(?:python)?\s*\n|```","",raw).strip()
                compile(code,"generated.py","exec")
                (d/"generated.py").write_text(code+"\n"); rec["status"]="generated"
            except Exception:
                rec["status"]="failed"; rec["error"]=traceback.format_exc()
            rec["seconds"]=round(time.monotonic()-started,3)
            (d/"generation.json").write_text(json.dumps(rec,indent=2))
        save("inference.json",{"model":MODEL,"revision":REV,"backend":"llama.cpp","quantization":"Q4_K_M","status":"finished"})
    finally:
        p.terminate()
        try: p.wait(timeout=15)
        except subprocess.TimeoutExpired: p.kill(); p.wait()
        logfile.close()
def evaluate():
    import bench
    from PIL import Image,ImageDraw
    bench.OUT=R; bench.evaluate()
    result=json.loads((R/"report.json").read_text()); result["variant"]="GGUF Q4_K_M CPU"; result["run_id"]=os.environ.get("GITHUB_RUN_ID")
    save("report.json",result)
    for name in NAMES:
        file=R/name/"comparison.png"
        if file.exists():
            image=Image.open(file).convert("RGB"); draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,1023,54),fill="white"); draw.text((15,10),name+": reference (left), CAD-Coder GGUF (right)",fill="black"); image.save(file)
if __name__=="__main__":
    R.mkdir(exist_ok=True)
    mode=sys.argv[1]
    try: {"fixtures":fixtures,"convert":convert,"infer":infer,"evaluate":evaluate}[mode]()
    except Exception:
        save(mode+"-failure.json",{"phase":mode,"error":traceback.format_exc()}); traceback.print_exc(); sys.exit(1)
