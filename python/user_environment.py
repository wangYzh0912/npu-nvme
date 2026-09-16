"""Explicit per-process CANN environments; no shell or system configuration edits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def read_profile(manifest_path, name, repo, *, library_override=None):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported environment manifest schema")
    if name == "default":
        name = manifest.get("default", "old")
        # Promotion requires the future full-state acceptance implementation;
        # a hand-edited default must not silently promote an unverified stack.
        if name != "old":
            raise ValueError("candidate default promotion is not yet implemented; use an explicit profile")
    profile = manifest.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise ValueError(f"environment {name!r} has no verified installation paths")
    profile = dict(profile)
    if library_override is not None:
        profile["library"] = str(Path(library_override).resolve())
    profile.setdefault("library", str(Path(repo).resolve() / "build_out/lib/libnpu_nvme.so"))
    toolkit = Path(profile["toolkit"])
    profile.setdefault("version_root", str(toolkit.parent if toolkit.name == "aarch64-linux" else toolkit))
    for field in ("python", "toolkit", "set_env", "library", "version_root"):
        path = Path(profile[field])
        if not path.is_absolute():
            raise ValueError(f"{field} must be absolute")
        if not (path.is_dir() if field in ("toolkit", "version_root") else path.is_file()):
            raise ValueError(f"missing {field}: {path}")
    if name == "candidate":
        for field in ("python", "toolkit", "set_env", "library", "version_root"):
            if not within(profile[field], manifest["private_root"]):
                raise ValueError(f"candidate {field} is outside private_root")
    driver_path = Path(manifest["driver_version_file"])
    driver = dict(line.split("=", 1) for line in driver_path.read_text().splitlines()
                  if "=" in line)
    if driver.get("Version") != manifest["expected_driver_version"]:
        raise ValueError("driver version changed; repeat compatibility audit")
    identity = {"profile": name, **profile,
                "driver_sha256": fingerprint(driver_path),
                "set_env_sha256": fingerprint(profile["set_env"]),
                "library_sha256": fingerprint(profile["library"])}
    version_root = Path(profile["version_root"])
    identity["component_versions"] = {
        component: fingerprint(version_root / component / "version.info")
        if (version_root / component / "version.info").is_file() else None
        for component in ("runtime", "compiler", "hccl", "opp", "opp_kernel")}
    identity["environment_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def with_python_identity(profile):
    """Identify installed distributions without importing MindSpore or CANN."""
    script = ("import importlib.metadata as m, json, platform; "
              "print(json.dumps({'python': platform.python_version(), "
              "'packages': sorted((d.metadata['Name'], d.version) for d in m.distributions() "
              "if d.metadata['Name'])}, sort_keys=True))")
    result = subprocess.run([profile["python"], "-I", "-c", script], capture_output=True,
                            text=True, check=True, timeout=15)
    identity = {k: v for k, v in profile.items() if k != "environment_id"}
    packages = json.loads(result.stdout)
    identity["python_version"] = packages["python"]
    identity["python_packages_sha256"] = hashlib.sha256(
        json.dumps(packages, sort_keys=True).encode()).hexdigest()
    identity["environment_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def isolated_environment(profile, repo, inherited=None):
    """Build search paths without executing vendor scripts or user shell hooks.

    The installed version's setenv script can copy custom operators into HOME;
    even inspection must not execute those side effects. Its hash is retained
    as provenance while paths are constructed explicitly here.
    """
    inherited = os.environ if inherited is None else inherited
    removed = {"PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT",
               "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
               "BASH_ENV", "ENV", "VIRTUAL_ENV", "TOOLCHAIN_HOME", "TBE_IMPL_PATH",
               "NPU_NVME_LIBRARY_PATH", "NPU_NVME_ENVIRONMENT_ID",
               "ASCEND_INSTALL_PATH", "ASCEND_CUSTOM_OPP_PATH"}
    # Preserve device/rank scheduling and unrelated user settings; remove paths
    # that can inject another Python, toolkit or virtual environment.
    env = {k: v for k, v in inherited.items()
           if k not in removed and not k.startswith(("PYTHON", "CONDA", "_CE_"))
           and not (k.startswith("ASCEND_") and ("PATH" in k or "HOME" in k))}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    toolkit = Path(profile["toolkit"])
    version_root = Path(profile["version_root"])
    opp = toolkit / "opp" if (toolkit / "opp").is_dir() else version_root / "opp"
    if not opp.is_dir():
        raise ValueError(f"selected toolkit has no OPP directory: {opp}")
    driver_root = Path("/usr/local/Ascend/driver")

    def paths(*items):
        return os.pathsep.join(str(p) for p in items if Path(p).is_dir())

    env["PATH"] = paths(Path(profile["python"]).parent, toolkit / "bin",
                        toolkit / "compiler/ccec_compiler/bin",
                        toolkit / "tools/ccec_compiler/bin", version_root / "compiler/ccec_compiler/bin",
                        "/usr/local/bin", "/usr/bin", "/bin")
    python_root = Path(profile["python"]).parent.parent
    python_site = next(iter((python_root / "lib").glob("python*/site-packages")),
                       python_root / "lib" / "python3.11" / "site-packages")
    env["PYTHONPATH"] = paths(Path(repo).resolve() / "python", Path(repo).resolve(),
                              python_site,
                              toolkit / "python/site-packages",
                              version_root / "python/site-packages",
                              opp / "built-in/op_impl/ai_core/tbe",
                              toolkit / "aarch64-linux/lib64")
    env["PYTHONNOUSERSITE"] = "1"
    env["LD_LIBRARY_PATH"] = paths(
        Path(profile["library"]).parent, toolkit / "lib64",
        toolkit / "lib64/plugin/opskernel", toolkit / "lib64/plugin/nnengine",
        toolkit / "tools/aml/lib64", toolkit / "tools/aml/lib64/plugin",
        opp / "built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
        driver_root / "lib64", driver_root / "lib64/common", driver_root / "lib64/driver")
    env["ASCEND_HOME_PATH"] = str(toolkit)
    env["ASCEND_TOOLKIT_HOME"] = str(toolkit)
    env["ASCEND_OPP_PATH"] = str(opp)
    env["ASCEND_AICPU_PATH"] = str(opp.parent)
    env["TOOLCHAIN_HOME"] = str(version_root / "toolkit")
    env["NPU_NVME_LIBRARY_PATH"] = profile["library"]
    env["NPU_NVME_ENVIRONMENT_ID"] = profile["environment_id"]
    env["NPU_NVME_PROFILE"] = profile["profile"]
    env["NPU_NVME_CANN_VERSION_ROOT"] = str(version_root)
    return env


def verify_library_paths(profile, env):
    """Reject missing dependencies, stubs, or a library linked to another CANN."""
    result = subprocess.run(["ldd", profile["library"]], env=env, capture_output=True,
                            text=True, check=True, timeout=15)
    if "not found" in result.stdout:
        raise ValueError(f"unresolved project library dependency: {result.stdout.strip()}")
    version_root = Path(profile["version_root"])
    resolved = {}
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        name, target = line.split("=>", 1)
        path = target.strip().split(" (", 1)[0]
        resolved[name.strip()] = path
        if not path.startswith("/"):
            continue
        if "stub" in Path(path).parts or "stubs" in Path(path).parts:
            raise ValueError(f"runtime resolved to a stub: {path}")
        if ("/Ascend/" in path or name.strip().startswith(
                ("libascend", "libacl", "libruntime", "libge_", "libhccl", "libgert"))):
            if not within(path, version_root) and not within(path, "/usr/local/Ascend/driver"):
                raise ValueError(f"project library resolves to another CANN: {path}")
    return resolved


def verify_runtime_abi(profile, env):
    """Load only the selected library; never initialize ACL, SPDK or a device."""
    expected = profile.get("required_abi_major")
    if expected is None:
        return None
    if type(expected) is not int or expected <= 0:
        raise ValueError("invalid required ABI major")
    code = """import ctypes,json,sys
lib=ctypes.CDLL(sys.argv[1])
fn=lib.npu_nvme_abi_version
fn.restype=ctypes.c_uint32
print(json.dumps({'abi_major':fn(),'loaded_library':sys.argv[1],
 'maps':sorted({line.split()[-1] for line in open('/proc/self/maps')
 if any(name in line for name in ('libnpu_nvme','libascendcl','libruntime.so'))})}))
"""
    result = subprocess.run([profile["python"], "-c", code, profile["library"]],
                            env=env, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("ABI load probe failed: " + result.stderr.strip())
    observed = json.loads(result.stdout)
    if observed["abi_major"] != expected:
        raise ValueError("runtime ABI differs from required major")
    return observed
