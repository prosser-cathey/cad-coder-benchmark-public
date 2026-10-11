# CAD-Coder synthetic image-to-CAD benchmark

Only synthetic fixtures, benchmark code, and results are public. No Datum code, credentials, vendor photos, private benchmark history, or model weights are published.

Uses the image-conditioned CAD-Coder by Anna Doris et al.: https://github.com/anniedoris/CAD-Coder (Apache-2.0 code), pinned at 7f40b4827b029731ef97313a116d62519e313960. Weights retain their own terms; their resolved revision is recorded. This is a 4-bit NF4 CPU smoke test, NOT full-precision paper reproduction or a Zoo test.

Two synthetic images: mounting plate and stepped flange. Geometry and camera settings reproduce the controls; input byte hashes are recorded. If hashes differ, the report explicitly marks that this is not an exact same-image head-to-head comparison. Author's exact prompt, vicuna_v1, greedy decoding, max 3450 tokens / 900 seconds per image, one attempt, no code or geometry repair. Only the image and generic prompt enter the model. Physical scale is not inferred.

Generated Python runs in a non-root, network-disabled Docker container with CPU, memory, PID, time, and file-size limits, without reference CAD or credentials. Geometry checks and rendering run separately. Results include STEP/STL, code, reference/predicted views, silhouette IoU, missing/excess pixels, hole fill, RGB error, normalized proportions, volume, and topology. Scoring thresholds are in bench.py. Failures and capped generations are not reported as completed accuracy tests.

Maximum 60 test + 5 publishing minutes on standard public CPU runners; no GPU or paid inference API. Unused preinstalled tools are removed only from the ephemeral runner. Result commits do not retrigger tests. Compatibility patches and package versions are preserved.
