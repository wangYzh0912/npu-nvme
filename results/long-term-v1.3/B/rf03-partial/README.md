# RF-03 partial migration

The production facade now constructs a framework-independent CheckpointScheduler. It owns paired request/snapshot capacity, leases, request sequencing, admission shutdown and drain watermarks. It receives no DirectCheckpoint reference. FrozenCapture owns the parameter registry and snapshot copy/release through explicitly supplied framework, ACL and pointer interfaces. Training cells have a canonical module and identity-preserving legacy alias.

160 portable cases pass after migration. A final 20-case ownership/admission run passes, including a drain with later admission and a copied Host snapshot. An initial new test fixture accidentally reacquired its own non-reentrant lock and was terminated; removing that fixture-only outer lock resolved it. This was not a production scheduler failure.

RF-03 remains in progress. Live capture, legacy commit/restore orchestration and the transfer worker remain in DirectCheckpoint; the facade still contains framework operations. No claim of a completed thin facade, strict RestoreSession, reader pins or a sole commit writer is made. H01-compat remains unrun pending an explicitly registered raw test region.
