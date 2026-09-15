# E0 migration and native training replay

Local experimental environment/Qwen code was selectively migrated with original hashes in migration_manifest.json. The candidate environment already existed; no driver/toolkit installation was changed. The old environment remains the default. Launch paths resolve from the checkout, output directories must be new, and each launch writes exact argv/source hashes and an environment lock.

All five original model safetensors were fully streamed and hashed, with strict extent/shape/dtype checks, into model-audit. Model config/tokenizer identities are in model_identity.json. replay-001 reproduces the original eight-step Native TP4 workload on devices 0–3 and completed with exit 0. EN adds independent checkpoint/horizon/stop controls and fresh-process restore.

E0 is not fully accepted: old/candidate GPT-2/Ours hardware smoke is still outstanding; raw Ours tests require a dedicated registered extent. Environment selection is not environment promotion. Native training does not call the project's raw SPDK backend and does not establish compatibility of the candidate project library. Actual library maps and hashes from the deterministic restore processes are under ../EN/restore-002.
