import json,runpy,traceback
from pathlib import Path
import cadquery as cq
try:
    ns=runpy.run_path("/work/generated.py")
    if "solid" not in ns: raise RuntimeError("The author evaluation convention requires variable solid")
    solid=ns["solid"]
    if not isinstance(solid,(cq.Workplane,cq.Shape)): raise TypeError("solid is not CadQuery geometry")
    cq.exporters.export(solid,"/work/predicted.step")
    cq.exporters.export(solid,"/work/predicted.stl",tolerance=.001,angularTolerance=.12)
    Path("/work/sandbox.json").write_text(json.dumps({"status":"exported"}))
except Exception:
    Path("/work/sandbox.json").write_text(json.dumps({"status":"failed","error":traceback.format_exc()},indent=2))
    raise
