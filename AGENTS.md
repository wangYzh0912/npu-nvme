# Development Conventions

- Review behavior through the current entry point and its complete call chain.
  `src/npu_nvme.c` contains the ABI version export; implementation lives in the
  compilation units listed in `CMakeLists.txt`. See `SOURCE_IMPLEMENTATION_AUDIT.md`
  for the problem-to-implementation map and known gaps.
- Keep changes within the requested behavior. Research and deferred features do
  not become implementation requirements merely because an old plan lists them.
- Prefer explicit contracts and one owner for each responsibility. Validate
  configuration, disk or IPC input, native return values, and allocation geometry
  at the boundary that owns them. Internal code may rely on established contracts.
- Avoid repeated validation of unchanged internal objects, speculative fallbacks,
  compatibility branches for unsupported runtimes, and extra state flags that can
  be derived from an existing owner or lifecycle. Add abstractions only to remove
  demonstrated complexity or duplication.
- Propagate errors with their cause. Do not catch errors merely to continue with a
  default value, silently switch backends, or report success after cleanup fails.
- Preserve the necessary DMA lifetime, request-reference, disk-range, integrity,
  and restore-readiness rules. An observation timeout does not authorize releasing
  a buffer still borrowed by a device. Simplify ownership rather than layering
  additional guards over it.
- Scale verification to risk. Add focused regression cases for reproduced bugs;
  distinguish source implementation, CPU or stubbed-C checks, and hardware
  acceptance. Empty tests and capability dictionaries do not prove functionality.
