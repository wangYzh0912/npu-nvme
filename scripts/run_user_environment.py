#!/usr/bin/env python3
"""Run a command in an explicit, isolated old/candidate CANN environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from user_environment import (isolated_environment, read_profile,
                              verify_library_paths, with_python_identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/user_environments.json")
    parser.add_argument("--profile", choices=("old", "candidate", "default"), default="default")
    parser.add_argument("--inspect", action="store_true", help="print resolved paths without launching a workload")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        profile = read_profile(args.manifest, args.profile, ROOT)
        profile = with_python_identity(profile)
        env = isolated_environment(profile, ROOT)
        libraries = verify_library_paths(profile, env)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"environment blocked: {error}\n")
    if args.inspect:
        keys = ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH")
        print(json.dumps({**profile, "paths": {k: env[k] for k in keys},
                          "resolved_libraries": libraries}, indent=2))
        return
    argv = args.command
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        parser.error("provide a command after --, or use --inspect")
    if argv[0] in ("python", "python3"):
        argv[0] = profile["python"]
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
