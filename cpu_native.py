import sys, json, time, gc, hashlib, traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"
OUT.mkdir(exist_ok=True)
try:
    file=BASE/"bench.py"
    original_bytes=file.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(original_bytes)).encode()+b"\0"+original_bytes).hexdigest()
    if blob!="d99d4796356db961fd2039a8e58adc0d8ee7b4e3": raise RuntimeError("Benchmark source changed; refusing unverified patch")
    original=original_bytes.decode(); src=original
    import torch
    import bitsandbytes as bnb
    from transformers.generation.streamers import BaseStreamer
    torch.set_num_threads(4); torch.manual_seed(7)
    cpu_info=Path("/proc/cpuinfo").read_text()
    (OUT/"cpu-info.txt").write_text(cpu_info)
    native_bf16=any(flag in cpu_info for flag in ("avx512_bf16","amx_bf16"))
    scores={}; weight=torch.randn(2048,2048)
    for name,dtype in (("float32",torch.float32),("bfloat16",torch.bfloat16)):
        try:
            layer=bnb.nn.Linear4bit(2048,2048,bias=False,compute_dtype=dtype,quant_type="nf4",compress_statistics=True).eval()
            layer.weight=bnb.nn.Params4bit(weight.to(dtype).clone(),requires_grad=False,quant_type="nf4",compress_statistics=True).to("cpu")
            timings={}
            with torch.inference_mode():
                for phase,rows in (("prefill",128),("decode",1)):
                    x=torch.randn(rows,2048,dtype=dtype)
                    y=layer(x); assert torch.isfinite(y).all()
                    start=time.perf_counter()
                    for _ in range(3): y=layer(x)
                    timings[phase]=(time.perf_counter()-start)/3
            scores[name]=timings
            del layer,x,y; gc.collect()
        except Exception: scores[name]={"error":traceback.format_exc()}
    if "error" in scores["float32"]: raise RuntimeError("FP32 NF4 preflight failed")
    chosen="float32"
    if native_bf16 and "error" not in scores["bfloat16"] and all(scores["bfloat16"][k]<scores["float32"][k] for k in ("prefill","decode")):
        chosen="bfloat16"
    del weight; gc.collect()
    (OUT/"cpu-performance.json").write_text(json.dumps({"native_bf16_instructions":native_bf16,"kernel_seconds":scores,"selected_compute_dtype":chosen,"note":"Kernel probe only; full-model throughput is recorded separately."},indent=2))
    class ProgressStreamer(BaseStreamer):
        def __init__(self,directory,tokenizer):
            self.directory=directory; self.tokenizer=tokenizer; self.ids=[]; self.skip_prompt=True
            self.started=time.monotonic(); self.first_token=None; self.finished=False; self.flush()
        def put(self,value):
            if self.skip_prompt:
                self.skip_prompt=False; return
            ids=value.reshape(-1).tolist()
            if ids and self.first_token is None: self.first_token=time.monotonic()-self.started
            self.ids.extend(ids)
            if len(self.ids)==1 or len(self.ids)%16==0: self.flush()
        def end(self):
            self.finished=True; self.flush()
        def flush(self):
            elapsed=time.monotonic()-self.started
            data={"stream_ended":self.finished,"tokens":len(self.ids),"elapsed_seconds":elapsed,"first_token_seconds":self.first_token,"decode_tokens_per_second":(len(self.ids)-1)/max(.001,elapsed-self.first_token) if self.first_token is not None and len(self.ids)>1 else None}
            (self.directory/"generation-progress.json").write_text(json.dumps(data,indent=2))
            (self.directory/"partial-code.txt").write_text(self.tokenizer.decode(self.ids,skip_special_tokens=True))
    assert src.count("torch.bfloat16")>=7
    src=src.replace("torch.bfloat16","torch."+chosen).replace("bfloat16 compute",chosen+" compute")
    def change(old,new):
        global src
        if src.count(old)!=1: raise RuntimeError("Unexpected source anchor: "+old[:100])
        src=src.replace(old,new)
    change('info=HfApi().model_info(state["model"])','info=HfApi().model_info(state["model"],revision="ff60c11d60e839c1da2c4fe445129df7dd89773e")')
    marker='    vision.to(device="cpu",dtype=torch.'+chosen+')\n'
    restore='''    from safetensors import safe_open
    weight_map=json.loads((Path(path)/"model.safetensors.index.json").read_text())["weight_map"]
    prefix="model.vision_tower.vision_tower."
    vision_keys={k:v for k,v in weight_map.items() if k.startswith(prefix)}
    if not vision_keys: raise RuntimeError("Checkpoint vision tensors missing; refusing base-vision substitution")
    vision_state={}
    for shard in sorted(set(vision_keys.values())):
        with safe_open(str(Path(path)/shard),framework="pt",device="cpu") as handle:
            for key in handle.keys():
                if key in vision_keys: vision_state[key[len(prefix):]]=handle.get_tensor(key)
    vision.vision_tower.load_state_dict(vision_state,strict=True)
    state["checkpoint_vision_tensors_restored"]=len(vision_state)
    del vision_state; gc.collect(); update()
'''
    change(marker,marker+restore)
    change('            torch.manual_seed(7)\n            with torch.inference_mode(): generated=','            torch.manual_seed(7)\n            streamer=ProgressStreamer(d,tokenizer)\n            with torch.inference_mode(): generated=')
    change('max_time=900,use_cache=True,pad_token_id=tokenizer.eos_token_id)','max_time=900,use_cache=True,pad_token_id=tokenizer.eos_token_id,streamer=streamer)')
    compile(src,str(file),"exec")
    (OUT/"adapted-bench.txt").write_text(src)
    (OUT/"native-adapter.json").write_text(json.dumps({"compute_dtype":chosen,"storage":"unchanged NF4 double quantization","vision":"strictly restored from the same checkpoint","model_revision":"ff60c11d60e839c1da2c4fe445129df7dd89773e","source_sha256":hashlib.sha256(original_bytes).hexdigest(),"adapted_sha256":hashlib.sha256(src.encode()).hexdigest()},indent=2))
    exec(compile(src,str(file),"exec"),{"__name__":"__main__","__file__":str(file),"ProgressStreamer":ProgressStreamer})
except Exception:
    (OUT/"cpu-native-failure.json").write_text(json.dumps({"error":traceback.format_exc()},indent=2))
    raise
