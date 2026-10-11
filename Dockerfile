FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy<2" cadquery==2.5.2
COPY sandbox.py /runner.py
WORKDIR /work
ENTRYPOINT ["python", "/runner.py"]
