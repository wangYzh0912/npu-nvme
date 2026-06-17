#!/usr/bin/env python3
"""Diagnose TBE import conflict between conda env and CANN toolkit."""
import sys

print("=== Python path order ===")
for i, p in enumerate(sys.path[:20]):
    marker = " <-- TBE" if "op_impl" in p or "tbe" in p else ""
    print(f"  [{i:2d}] {p}{marker}")

print()
print("=== Importing tbe ===")
try:
    import tbe
    print(f"tbe location: {tbe.__file__}")
    print(f"tbe paths: {tbe.__path__}")
except Exception as e:
    print(f"FAILED to import tbe: {e}")

print()
print("=== Importing tbe.common ===")
try:
    import tbe.common
    print(f"tbe.common paths: {tbe.common.__path__}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print()
print("=== Importing tbe.common.utils ===")
try:
    from tbe.common.utils import util
    print(f"SUCCESS: {util}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print()
print("=== Listing tbe subpackages ===")
try:
    import pkgutil
    for importer, modname, ispkg in pkgutil.iter_modules(tbe.__path__, prefix="tbe."):
        if ispkg:
            print(f"  [pkg] {modname}")
except Exception as e:
    print(f"FAILED: {e}")

print()
print("=== Checking CANN tbe path ===")
cann_tbe = "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"
import os
if os.path.isdir(cann_tbe):
    print(f"CANN tbe dir exists: {cann_tbe}")
    for f in sorted(os.listdir(cann_tbe))[:20]:
        is_dir = os.path.isdir(os.path.join(cann_tbe, f))
        print(f"  {'[dir]' if is_dir else '[file]'} {f}")
    has_init = os.path.exists(os.path.join(cann_tbe, "__init__.py"))
    print(f"  has __init__.py: {has_init}")
