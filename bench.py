import os, sys, json, time, hashlib, traceback, re, gc, subprocess, shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"
NAMES=("mounting-plate","stepped-flange")
PROMPT="Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words."
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2))
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def render(mesh,path,view):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    v=(mesh.vertices-mesh.bounds.mean(axis=0))/mesh.extents.max()
    light=np.array([.3,-.5,.8]); light/=np.linalg.norm(light)
    colors=np.repeat((.4+.4*np.maximum(mesh.face_normals@light,0))[:,None],3,axis=1)
    fig=plt.figure(figsize=(4,4),dpi=128,facecolor="white")
    ax=fig.add_axes([0,0,1,1],projection="3d")
    ax.add_collection3d(Poly3DCollection(v[mesh.faces],facecolors=colors,edgecolors="none",linewidths=0))
    ax.set(xlim=(-.6,.6),ylim=(-.6,.6),zlim=(-.6,.6)); ax.set_box_aspect((1,1,1)); ax.set_proj_type("ortho")
    ax.view_init(*{"iso":(25,-55),"front":(0,-90),"top":(90,-90)}[view]); ax.set_axis_off()
    fig.savefig(path,dpi=128,facecolor="white"); plt.close(fig)
    return np.asarray(Image.open(path).convert("RGB"))
def measured(shape,mesh):
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    ex=TopExp_Explorer(shape,TopAbs_SOLID); n=0
    while ex.More(): n+=1; ex.Next()
    s=float(mesh.extents.max())
    return {"valid":bool(BRepCheck_Analyzer(shape).IsValid()),"solids":n,"aspect":(mesh.extents/s).tolist(),"volume":float(abs(mesh.volume)/s**3),"watertight":bool(mesh.is_watertight),"euler":int(mesh.euler_number)}
def prepare():
    import trimesh
    from OCC.Core.gp import gp_Ax2,gp_Pnt,gp_Dir
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut,BRepAlgoAPI_Fuse
    from OCC.Extend.DataExchange import write_step_file,write_stl_file
    def cyl(x,y,z,r,h): return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x,y,z),gp_Dir(0,0,1)),r,h).Shape()
    plate=BRepPrimAPI_MakeBox(1.,.55,.1).Shape()
    for x in (.25,.75): plate=BRepAlgoAPI_Cut(plate,cyl(x,.275,-.01,.07,.12)).Shape()
    flange=BRepAlgoAPI_Fuse(cyl(0,0,0,.5,.12),cyl(0,0,.12,.25,.18)).Shape()
    flange=BRepAlgoAPI_Cut(flange,cyl(0,0,-.01,.1,.32)).Shape()
    expected=("ada2384756b448b4ce44c88243d999d4b5aac9ac150efb6f533a02d295dc8094","39ab4a8c78b4bc42bf61aebe7e6fc6c05fb0235b44212896382d6b38f79a21f3")
    matches=[]
    for name,shape,wanted in zip(NAMES,(plate,flange),expected):
        d=OUT/name; d.mkdir(parents=True,exist_ok=True)
        write_step_file(shape,str(d/"reference.step"))
        write_stl_file(shape,str(d/"reference.stl"),mode="binary",linear_deflection=.001,angular_deflection=.12)
        mesh=trimesh.load(str(d/"reference.stl"),force="mesh")
        for view in ("iso","front","top"): render(mesh,d/("reference-"+view+".png"),view)
        actual=digest(d/"reference-iso.png")
        save(d/"reference.json",{"geometry":measured(shape,mesh),"input_sha256":actual,"matches_original_input_bytes":actual==wanted,"expected_sha256":wanted})
        matches.append(actual==wanted)
    save(OUT/"fixture-status.json",{"status":"prepared","inputs_byte_identical":all(matches),"comparison_note":"Exact control images" if all(matches) else "Same synthetic geometry and render settings, but bytes differ; not an exact same-image head-to-head comparison"})
def infer():
    import torch
    import bitsandbytes as bnb
    from PIL import Image
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoTokenizer,BitsAndBytesConfig
    torch.set_num_threads(4); torch.manual_seed(7)
    state={"model":"CADCODER/CAD-Coder","variant":"NF4 double-quantized CPU; bfloat16 compute","prompt":PROMPT,"temperature":0,"max_new_tokens":3450,"max_seconds_per_case":900,"status":"preflight"}
    def update(): save(OUT/"inference.json",state)
    update()
    state["ram_bytes"]=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    state["free_disk_bytes"]=shutil.disk_usage(BASE).free; update()
    if state["ram_bytes"]<14*1024**3: raise RuntimeError("This test requires a 16-GB-class public runner")
    if state["free_disk_bytes"]<32*1024**3: raise RuntimeError("Insufficient disk for the original model download")
    layer=bnb.nn.Linear4bit(128,128,bias=False,compute_dtype=torch.bfloat16,quant_type="nf4",compress_statistics=True)
    layer.weight=bnb.nn.Params4bit(torch.randn(128,128,dtype=torch.bfloat16),requires_grad=False,quant_type="nf4",compress_statistics=True).to("cpu")
    with torch.inference_mode(): probe=layer(torch.randn(1,128,dtype=torch.bfloat16))
    assert probe.shape==(1,128) and torch.isfinite(probe).all()
    del layer,probe; gc.collect()
    root=BASE/"upstream"
    f=root/"llava/model/language_model/llava_llama.py"; text=f.read_text()
    edits=[("        return_dict: Optional[bool] = None,\n","        return_dict: Optional[bool] = None,\n        **kwargs,\n"),("            return_dict=return_dict\n","            return_dict=return_dict,\n            **kwargs\n"),("self.pretraining_tp = config.pretraining_tp","self.pretraining_tp = getattr(config, 'pretraining_tp', 1)")]
    for old,new in edits:
        if text.count(old)!=1: raise RuntimeError("Unexpected upstream compatibility target")
        text=text.replace(old,new)
    compile(text,str(f),"exec"); f.write_text(text)
    (root/"llava/model/__init__.py").write_text("from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig\n")
    (OUT/"compatibility.txt").write_bytes(subprocess.check_output(["git","-C",str(root),"diff"]))
    sys.path.insert(0,str(root))
    from llava.model.language_model.llava_llama import LlavaConfig,LlavaLlamaForCausalLM
    from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN,DEFAULT_IMAGE_PATCH_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token,process_images
    tiny=LlavaLlamaForCausalLM(LlavaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=64)).eval()
    with torch.inference_mode(): test=tiny.generate(torch.tensor([[1,5,6]]),max_new_tokens=2,min_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=2)
    assert test.ndim==2 and test.numel()>0
    del tiny,test; gc.collect()
    state["cpu_nf4_and_generation_api_preflight"]="passed"; state["status"]="downloading"; update()
    info=HfApi().model_info(state["model"])
    state["model_revision"]=info.sha; update()
    files=[x.rfilename for x in info.siblings if "/" not in x.rfilename and (x.rfilename.endswith(".safetensors") or x.rfilename in {"config.json","generation_config.json","model.safetensors.index.json","tokenizer.model","tokenizer.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json"})]
    if not any(x.endswith(".safetensors") for x in files): raise RuntimeError("No root safetensors checkpoint found")
    path=snapshot_download(state["model"],revision=info.sha,allow_patterns=files,local_dir=str(BASE/"weights"),max_workers=2)
    state["status"]="loading"; update()
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=["lm_head","mm_projector","vision_tower"])
    tokenizer=AutoTokenizer.from_pretrained(path,use_fast=False)
    model=LlavaLlamaForCausalLM.from_pretrained(path,quantization_config=q,torch_dtype=torch.bfloat16,device_map={"":"cpu"},low_cpu_mem_usage=True,use_safetensors=True,attn_implementation="eager").eval()
    if getattr(model.config,"mm_use_im_patch_token",True): tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN],special_tokens=True)
    if getattr(model.config,"mm_use_im_start_end",False): tokenizer.add_tokens([DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN],special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    vision=model.get_vision_tower()
    if not vision.is_loaded: vision.load_model(device_map="cpu")
    vision.to(device="cpu",dtype=torch.bfloat16)
    state["vision_model"]=getattr(vision,"vision_tower_name",None)
    state["status"]="generating"; update()
    for name in NAMES:
        d=OUT/name; started=time.monotonic(); result={"status":"running","quantization":"NF4","repair_attempts":0}; save(d/"generation.json",result)
        try:
            image=Image.open(d/"reference-iso.png").convert("RGB")
            image_token=DEFAULT_IMAGE_TOKEN
            if getattr(model.config,"mm_use_im_start_end",False): image_token=DEFAULT_IM_START_TOKEN+image_token+DEFAULT_IM_END_TOKEN
            conv=conv_templates["vicuna_v1"].copy(); conv.append_message(conv.roles[0],image_token+"\n"+PROMPT); conv.append_message(conv.roles[1],None)
            ids=tokenizer_image_token(conv.get_prompt(),tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt").unsqueeze(0)
            imgs=process_images([image],vision.image_processor,model.config)
            imgs=[x.to(dtype=torch.bfloat16,device="cpu") for x in imgs] if isinstance(imgs,list) else imgs.to(dtype=torch.bfloat16,device="cpu")
            torch.manual_seed(7)
            with torch.inference_mode(): generated=model.generate(ids,images=imgs,image_sizes=[image.size],do_sample=False,num_beams=1,max_new_tokens=3450,max_time=900,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.batch_decode(generated,skip_special_tokens=True)[0].strip(); (d/"raw.txt").write_text(raw)
            result["tokens"]=int(generated.shape[-1]); result["ended_with_eos"]=int(generated[0,-1])==tokenizer.eos_token_id
            if not result["ended_with_eos"]: raise RuntimeError("Generation ended without EOS; incomplete output is not scored")
            code=re.sub(r"```(?:python)?\s*\n|```","",raw).strip()
            compile(code,"generated.py","exec")
            (d/"generated.py").write_text(code+"\n")
            result["status"]="generated"
        except Exception: result["status"]="failed"; result["error"]=traceback.format_exc()
        result["seconds"]=round(time.monotonic()-started,3); save(d/"generation.json",result)
    state["status"]="finished"; update()
def evaluate():
    import numpy as np
    import trimesh
    from scipy.ndimage import binary_fill_holes
    from PIL import Image,ImageDraw
    from OCC.Extend.DataExchange import read_step_file
    result={"model":"CAD-Coder","variant":"4-bit NF4 CPU","scale_policy":"Center and isotropically normalize longest dimension; no rotation fitting or geometry repair; absolute scale not inferred.","cases":[]}
    for name in NAMES:
        d=OUT/name; case={"name":name,"passed":None,"status":"not_run"}; result["cases"].append(case)
        if not (d/"generation.json").exists():
            case["reason"]="No model generation recorded"; continue
        try:
            case["generation"]=json.loads((d/"generation.json").read_text())
            if case["generation"]["status"]!="generated":
                case["status"]="generation_failed_or_capped"; case["reason"]=case["generation"].get("error"); continue
            case["passed"]=False
            if not (d/"predicted.step").exists(): raise RuntimeError("Sandbox did not produce STEP")
            shape=read_step_file(str(d/"predicted.step")); mesh=trimesh.load(str(d/"predicted.stl"),force="mesh")
            pred=measured(shape,mesh); ref=json.loads((d/"reference.json").read_text())["geometry"]
            case["predicted"]=pred; case["reference"]=ref; case["views"]={}
            for view in ("iso","front","top"):
                a=np.asarray(Image.open(d/("reference-"+view+".png")).convert("RGB")); b=render(mesh,d/("predicted-"+view+".png"),view)
                am=a.min(axis=2)<245; bm=b.min(axis=2)<245; holes=binary_fill_holes(am)&~am
                case["views"][view]={"iou":float((am&bm).sum()/max(1,(am|bm).sum())),"missing_fraction":float((am&~bm).sum()/max(1,am.sum())),"excess_fraction":float((bm&~am).sum()/max(1,bm.sum())),"hole_fill_fraction":float((holes&bm).sum()/max(1,holes.sum())),"rgb_mae":float(np.abs(a.astype(float)-b).mean()/255)}
            case["aspect_error"]=float(np.abs(np.array(pred["aspect"])-ref["aspect"]).max()); case["volume_error"]=abs(pred["volume"]/ref["volume"]-1)
            case["passed"]=bool(pred["valid"] and pred["solids"]==1 and pred["watertight"] and pred["euler"]==ref["euler"] and case["aspect_error"]<=.05 and case["volume_error"]<=.1 and min(x["iou"] for x in case["views"].values())>=.85 and case["views"]["top"]["hole_fill_fraction"]<=.1)
            canvas=Image.new("RGB",(1024,1600),"white"); ImageDraw.Draw(canvas).text((15,10),name+": reference (left), CAD-Coder NF4 (right)",fill="black")
            for i,view in enumerate(("iso","front","top")):
                for x,prefix in ((0,"reference"),(512,"predicted")): canvas.paste(Image.open(d/(prefix+"-"+view+".png")).convert("RGB"),(x,55+i*512))
            canvas.save(d/"comparison.png"); case["status"]="evaluated"
        except Exception: case["status"]="failed"; case["error"]=traceback.format_exc()
    save(OUT/"report.json",result)
if __name__=="__main__":
    mode=sys.argv[1]; OUT.mkdir(exist_ok=True)
    try: {"prepare":prepare,"infer":infer,"evaluate":evaluate}[mode]()
    except Exception:
        save(OUT/(mode+"-failure.json"),{"stage":mode,"error":traceback.format_exc()}); traceback.print_exc(); sys.exit(1)
