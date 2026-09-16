# B: RF-01 / RF-02 software migration

Entry: A software gate-002, commit 646760c. Self-reviewed.

Pure protocol/layout/V2 format/chunk construction and ctypes declarations now have canonical `npu_nvme` imports. Legacy facades retain names and class identity. Device-dependent allocation is separated from pure chunk construction. Legacy c_bindings remains eager on import; the canonical binding loads only on explicit open. V2 bytes are compared with fixtures produced from the A commit.

79 C functions moved into runtime, reactor, request, validation, DMA, write/read pipeline, metadata, step-listener and metrics modules. Existing five ABI compatibility wrappers remain in npu_nvme.c. Cross-module private symbols have hidden visibility. No intentional transport or V2 format changes.

Validation: production shared library built; all 3,641 dynamic exports unchanged; C sizeof/offsetof matches ctypes; 17 production C cases pass ASan/UBSan; 160 portable Python cases pass. Final fixture/ABI rerun: 5 pass. See adjacent logs and module map.

B remains in progress: RF-03 runtime/framework ownership is not complete and H01-compat needs a registered dedicated raw test extent. CPU results do not establish DMA/NVMe compatibility. Legacy synchronous pointer wrappers still drain before returning an observation timeout to protect caller-owned memory, pending B2/C2 migration.
