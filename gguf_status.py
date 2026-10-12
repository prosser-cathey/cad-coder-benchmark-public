import os,sys,json,base64,datetime,subprocess
repo=os.environ["GITHUB_REPOSITORY"]
assert repo=="prosser-cathey/cad-coder-benchmark-public"
run=os.environ["GITHUB_RUN_ID"]
path="repos/"+repo+"/contents/runs/"+run+".json"
data={"phase":sys.argv[1],"run_id":run,"head_sha":os.environ["GITHUB_SHA"],"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"url":"https://github.com/"+repo+"/actions/runs/"+run,"worker_result":os.environ.get("WORKER_RESULT"),"publisher_status":os.environ.get("PUBLISHER_STATUS"),"model":"CAD-Coder GGUF Q4_K_M"}
old=subprocess.run(["gh","api",path],text=True,capture_output=True)
payload={"message":"Record CAD benchmark phase: "+sys.argv[1],"content":base64.b64encode(json.dumps(data,indent=2).encode()).decode(),"branch":"main"}
if old.returncode==0: payload["sha"]=json.loads(old.stdout)["sha"]
elif "404" not in old.stderr: raise RuntimeError(old.stderr)
p=subprocess.run(["gh","api","--method","PUT",path,"--input","-"],input=json.dumps(payload),text=True,capture_output=True)
if p.returncode: raise RuntimeError(p.stderr)
