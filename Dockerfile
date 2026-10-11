FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy>=2,<3" cadquery==2.5.2 && pip check
RUN python -c "from pathlib import Path; import cadquery as cq; import numpy as np; s=cq.Workplane('XY').box(10,10,2).faces('>Z').workplane().hole(2); assert s.val().isValid(); cq.exporters.export(s,'/tmp/cad-preflight.step'); t=cq.importers.importStep('/tmp/cad-preflight.step'); assert t.val().isValid(); Path('/tmp/cad-preflight.step').unlink(); print({'numpy':np.__version__,'cadquery':cq.__version__,'step_roundtrip':'passed'})"
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
COPY sandbox.py /runner.py
WORKDIR /work
ENTRYPOINT ["python", "/runner.py"]
