# Baseline evidence

`retained_runs.json` indexes the latest completed 30-step run per adapter. Each saved method has fresh-process restore evidence; `none` has no checkpoint. Prior retries and old aggregate reports are not retained.

Ours attach and full-state recovery subsequently passed; see [current recovery](../full-state-recovery-net-20260907/README.md). Original run metadata remains unchanged. External payloads and upstream worker environments must be prepared separately.
