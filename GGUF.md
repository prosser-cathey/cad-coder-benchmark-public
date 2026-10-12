# Free GGUF benchmark

This is the CPU-native llama.cpp route for the released CAD-Coder checkpoint. It is a Q4_K_M quantized test, not full precision and not the earlier NF4/PyTorch runtime. No model weights, credentials, or Datum production code are published.

The same synthetic fixtures, authors' generic image prompt, and existing geometry checks are reused. Vision and projector tensors come from CAD-Coder's own checkpoint; only configuration is read from its CLIP base. Model revision and both converter/runtime source revisions are pinned in gguf_job.py. Generated programs execute only in the existing restricted, network-disabled CadQuery container.

Current phase and run ID are published under runs/. Final results and independent worker diagnostics are written to gguf-results/. Workflow success is not equivalent to model accuracy: inspect each case's generated output and passed field. Missing artifacts are reported explicitly.

Only standard public GitHub CPU runners are used: maximum 60 worker minutes plus 5 publishing minutes. No paid GPU, paid API, subscription, or credit purchase is involved. This implementation is not runtime-validated until the workflow produces its reports.
