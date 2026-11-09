# Subsequent experiments after this observation campaign

1. Reproduce the retained optimizer-copy fragment on a supported newer MindSpore/MindFormers/CANN stack. Current official master documents a separate PyNative/HyperParallel route. Keep software-stack comparisons separate from the registered O1 size matrix; do not claim the installed1.7 path characterizes every current release.
2. If resident FusedAdamW passes the predeclared local numerical/timing gates, test a separately registered full-model substitution with identical clipping, betas/eps/decay/LR and parameter/state precision. Use varied real training batches and declare numerical criteria before results. Confirm actual kernel dispatch, normal step latency, HBM flow and tail duration. Local fragment gains are not an end-to-end speedup estimate.
3. Runtime readiness is the gate for genuine AIC/AIV coordination. The current model's global L2 norm/clipping precedes all291 optimizer first-moment updates, so moving a full per-layer AdamW update ahead of remaining backward work is not generally legal. The paired dX/dW candidate also has a recomputed activation dependency. Instrument actual producer completion, collective completion, gradient scale availability and buffer/alias lifetime on one bounded candidate before trying overlap.
4. A distinct future hypothesis is a dependency-preserving next-step forward wavefront: after global clipping, allow next-step forward on a layer only after all its current-step parameter updates finish, while independent later-layer updates continue. This requires proof of weight/moment ownership, immutable per-step step/LR values, loss/overflow semantics, activation lifetime and the same synchronous-SGD result; it is currently unimplemented and unproven. First check whether the official pipeline/overlap routes already cover the desired scheduling contract. Do not remove clipping, reuse stale weights, or treat this as already-ready model work.
5. Larger micro-batches, gradient accumulation and supported pipeline schedules are established ways to amortize optimizer and communication overhead. Evaluate them as official workload/scheduling controls, separately from a novel AIC/AIV co-run claim. Fixed repeated-text observations do not establish prevalence across models, input distributions, parallel topologies or precision regimes.

6. Before a custom general scheduler, assess the official MindSpeed-LLM MC2
compute-communication fusion route and CANN MC2 tile/overlap diagnostics on a
supported stack. This is a separate framework/communication-path control,
not a transparent flag in this pinned MindSpore training graph. Measure whether
its improvements concern the backward communication intervals, the optimizer
tail, or both. Official FSDP/HSDP/prefetch/micro-batch guidance is also relevant
when changing parallel strategy; keep those changes out of the registered
TP4/DP1 baseline comparison.
