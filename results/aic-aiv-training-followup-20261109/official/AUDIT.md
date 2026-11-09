# Official optimization audit, retrieved 2026-11-09

This audit separates available optimization mechanisms from proof that they solve this workload's AIC inactivity. It concerns the installed MindSpore 2.7.1 / MindFormers 1.7 stack; it is not a claim that every current Ascend stack has the same behavior.

|Official mechanism|Evidence and contract|Decision|
|---|---|---|
|MindSpore JitConfig O1|Versioned official API says common optimization and automatic operator fusion; still operator dispatch. O0 disables optional optimization.|Measure first; compare actual fused nodes and low-AIC occupancy independently.|
|MindSpore JitConfig O2|Official API describes Ascend GraphEngine; static graph/static shapes and graph sinking required, extra compilation cost.|Only after O1 compatibility checks; keep unsupported runs as failures.|
|MindFormers FusedAdamW|Official v1.7.0 source fetched from mindspore-ai/mindformers; byte-for-byte matches installed implementation. Uses generated fused AdamW primitive, same betas/eps/decay configurable.|Validate the real optimizer fragment on recorded shard geometries before considering model substitution.|
|FlashAttention and fused normalization|Official v1.7.0 GPT layer specs exactly match installed bytes. Retained O1 final IR contains 72 FlashAttentionScore / 36 gradient nodes and 289 RmsNorm / 145 gradient nodes (includes recomputation).|Already present; do not treat their availability as a new intervention.|
|FusedPmaAdamW|Official installed source additionally performs EMA/SMA parameter averaging and periodically replaces parameters.|Changes the optimization algorithm; excluded from transparent AdamW substitution.|
|MindSpore AdamWeightDecay|Exists as an optimized optimizer but its update/step and bias-correction contract cannot be assumed identical to the current MindFormers AdamW.|Do not substitute as a transparent fix.|

There is no evidence here of a universal switch that places independent model AIC work into every AIV window. Lower AIV time after fusion and higher AIC busy percentages do not by themselves prove cooperative scheduling or recovered FLOPs.

The fixed workload uses TP4/DP1, sequence parallel, full recomputation, FP32 parameters and optimizer state with BF16 model compute. Changing optimizer sharding/parallel topology/precision is outside this first matrix because it changes both workload and communication behavior.

References, retrieved source bytes and hashes are stored beside this audit. The JitConfig source URL is recorded in JitConfig-r2.7.1.json; official FusedAdamW identity is in fused_adamw-v1.7.0.json.

## CANN 8.3 HBM source audit

The pinned CANN 8.3 `hbm_model.py` reports `SCALE = 0.000030517578125`, equivalent to 32 bytes per counter unit after the KB conversion; the parser groups `HBMOriginalData` by timestamp/event/channel. `InfoConfReader.get_delta_time()` is applied to device timestamps before wall-clock display. The archive keeps these source files and the host/device clock anchors used by the corrected export.

## Current community development line, checked 2026-11-09

The live official master documentation now describes MindSpore Transformers r2.0/PyNative as the evolution direction. Its installation page labels this development line as requiring MindSpore >=2.10 and HyperParallel, with matching CANN/driver requirements; it also says 2.0 is not yet available from PyPI. It is not the installed MindSpore2.7.1/MindFormers1.7/CANN8.3 stack. These facts are documentation evidence, not a verified migration or benchmark. Live HTML and hashes are retained under `current/`.

The current official parallel-training page exposes pipeline P2P/compute overlap and backward/forward overlap controls. These are conditional pipeline scheduling capabilities, not evidence that arbitrary independent AIC work can be inserted into every optimizer AIV window. Our PP1 fixed matrix does not activate them. The current training-features page still defines global L2 clipping across all gradients/all-reduce before optimizer update.

Conclusion: official fusion and parallel scheduling mechanisms already exist; neither a claim of “no official solution” nor of “all low-AIC issues solved” follows. Validate the existing installed fused optimizer first; before proposing a new scheduler as a general ecosystem fix, independently reproduce the concrete residual fragment on a supported newer stack. The newer-stack migration is a subsequent experiment, not silently mixed into the present version-controlled matrix.

## Broader official Ascend training controls

Official MindSpeed-LLM MC2 documentation describes integrating communication
with computation to reduce AllGather/MatMul and other compute-communication
serialization. It is a relevant existing control for TP/sequence-parallel
communication gaps. CANN's MC2 profiling documentation exposes compute and
communication tile timing, wait time, and overlap, which are preferable to
inferring communication gaps from engine averages alone. The official FSDP2
tuning guide also covers HSDP, fusion, layer prefetch and micro-batch selection;
these require a different parallel/runtime setup from this fixed TP4/DP1 graph.
These capabilities were not benchmarked in this campaign, and their existence
is not a claim that this model's optimizer-state tail is already solved.

The pinned installed training_graph source search found no MC2/MatmulAllGather/
AllGatherMatmul implementation reference. The frozen observation workload also
removes parallel_speed_up_json_path when building its fixed deterministic
configuration. Therefore current results characterize that explicit graph,
not every official optimized training path. Before a novel general scheduler
claim, compare a supported official compute-communication fusion route and
its native overlap trace alongside the fused-optimizer whole-model control.
Documentation source identities are retained under ascend/; failed downloads
are explicit and are not treated as retrieved source bytes.
