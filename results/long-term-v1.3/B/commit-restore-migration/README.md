# Legacy commit/restore ownership migration

LegacyCommitCoordinator owns the V2 catalog. MetadataIO receives the binding/context explicitly and implements bounded metadata reads/writes and flush ordering. Compatibility catalog properties route to that single state object. No V2 format or retention algorithm was changed. Full commit still has legacy rank/step behavior; D1 must add unique-writer enforcement and reader pins.

LegacyRestore owns full-state selection/validation/read/apply/verify ordering through explicitly supplied selector, transport and target interfaces. LegacyStateTarget owns framework allocations, numeric representation, application and checksum readback. LegacyBatchTransport only calls the current batch ABI. FrozenCapture now enumerates namespaced model/optimizer descriptors. Runtime imports neither framework modules nor NumPy.

Validation: 181 portable cases pass. Five metadata fixture cases compare real encoded bytes, I/O/flush sequence, catalog generation and rollback with outputs captured from pre-migration commit 5b7ef43, including replica/superblock/first/second flush errors. Five actual Host target/transport cases cover success, missing/invalid shape, read failure and late checksum failure. All pass. Test counts overlap.

Legacy restore applies into the supplied target before checking all checksums; the checksum fault case deliberately records that unchanged behavior. This is not a strict unready RestoreSession and must not be advertised as D1 acceptance. Write worker, live/FaF/incremental helpers and weights-only load remain in the compatibility manager, so RF-03/B are still in progress; H01-compat remains unrun without a registered raw extent.
