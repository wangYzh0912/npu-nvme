"""One training command surface for the retained checkpoint methods."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .adapters import ADAPTERS
from .protocol import DependencyBlocked
from .runner import prepare as prepare_run, run as train_run, restore as restore_run
from .state_bridge import write_json
from .inventory import build_inventory, storage_evidence

ROOT = Path(__file__).resolve().parents[3]


def native_identity(config):
    path = ROOT / 'build_out/lib/libnpu_nvme.so'
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None, 'required_abi': 2}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema_version') != 2:
        raise ValueError('training config schema_version must be 2')
    retired = {'ours_chunk_bytes','ours_keep_last_n','ours_max_inflight','keep_last_n','capture_mode','io_mode'} & config.keys()
    if retired:
        raise ValueError(f'removed configuration keys: {sorted(retired)}')
    required = {'model','seed','input_tokens','batch_size','dropout','warmup_steps','formal_steps','checkpoint_every','continue_steps','loss_rtol','loss_atol','npu_device','fs_test_dir','raw_pci','results_root','timeout_seconds','methods'}
    if required - config.keys():
        raise ValueError(f'missing configuration: {sorted(required - config.keys())}')
    if config['batch_size'] != 1:
        raise ValueError('current training fixture requires batch_size 1')
    for key in ('input_tokens','formal_steps','checkpoint_every','continue_steps','timeout_seconds'):
        if not isinstance(config[key], (int,float)) or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if config['formal_steps'] % config['checkpoint_every']:
        raise ValueError('final training step must be a checkpoint for continuation oracle')
    if set(config['methods']) - ADAPTERS.keys():
        raise ValueError('unknown method configuration')
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    if config.get('project_commit', head) != head or Path(config.get('project_root', ROOT)).resolve() != ROOT:
        raise ValueError('configuration points to a different implementation')
    config.update(project_commit=head, project_root=str(ROOT))
    for key in ('results_root','fs_test_dir'):
        config[key] = str((ROOT / config[key]).resolve())
    return config


def method_config(config, name):
    return dict(config, **config['methods'].get(name, {}))


def is_formal_semantic_port(row):
    checkpoints = row.get('checkpoints', [])
    return (row.get('status') == 'trend_measured' and
            row.get('kind') in {'npu-semantic-port','host-adapted-semantic-port'} and
            row.get('checkpoint_count', 0) == len(checkpoints) > 0 and
            row.get('restore', {}).get('status') == 'pass' and
            all(c.get('state') == 'PERSISTED' for c in checkpoints))


def env_snapshot(config):
    def command(argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True,
                                  check=False, timeout=10).__dict__
        except Exception as error:
            return {"error": repr(error)}
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "repo_commit": config["project_commit"],
        "native_library": native_identity(config),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
        "git_status": subprocess.run(["git", "status", "--porcelain", "-uno"],
                                      capture_output=True, text=True).stdout,
        "hardware": {
            "npu_device": config["npu_device"], "numa_node": config["numa_node"],
            "raw_pci": config["raw_pci"],
            "raw_pci_info": command(["lspci", "-s", config["raw_pci"], "-nnk"]),
            "findmnt": command(["findmnt", "-no", "TARGET,SOURCE,FSTYPE,OPTIONS", "-T", config["fs_test_dir"]]),
            "lsblk": command(["lsblk", "-o", "NAME,PATH,MODEL,SERIAL,PKNAME,FSTYPE,MOUNTPOINT"]),
            "storage_evidence": storage_evidence(config),
        },
    }

def preflight(config, selected, runtime_probe=False):
    root = Path(config["results_root"]) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    output = {"config": config, "environment": env_snapshot(config), "adapters": {}}
    for name in selected:
        try:
            status = ADAPTERS[name].preflight(config)
            if runtime_probe and name == "ours" and status.get("status") == "ready":
                status = ADAPTERS[name].runtime_probe(config)
            output["adapters"][name] = status
        except Exception as error:
            output["adapters"][name] = {"adapter": name, "status": "build_failed",
                                         "reason": repr(error)}
    write_json(root / "preflight.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output

def adapter_evidence(name):
    adapter = ADAPTERS[name]
    lines = [
        f"# Adapter: {name}",
        f"# Port class: {adapter.kind}",
        f"# Locked upstream core invoked: {getattr(adapter, 'upstream_core_invoked', None)}",
        "# Mechanisms preserved:",
    ]
    lines.extend(f"# - {item}" for item in getattr(
        adapter, "mechanisms_preserved", ()))
    lines.append("# Platform substitutions:")
    lines.extend(f"# - {item}" for item in getattr(
        adapter, "platform_substitutions", ()))
    return "\n".join(lines) + "\n"

def run_one(config, name, mode="fit"):
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter: {name}")
    stamp = str(time.time_ns())
    run_dir = Path(config["results_root"]) / name / f"{name}_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", env_snapshot(config))
    lock_path = Path(__file__).with_name("upstream.lock.json")
    if lock_path.exists():
        shutil.copy2(lock_path, run_dir / "upstream.lock.json")
    worker_lock = Path(__file__).with_name("worker_environment.lock.json")
    if worker_lock.exists():
        shutil.copy2(worker_lock, run_dir / "worker_environment.lock.json")
    schema_path = Path(config["results_root"]) / "prepare" / "state_schema.json"
    if schema_path.exists():
        shutil.copy2(schema_path, run_dir / "state_schema.json")
    (run_dir / "adapter_diff.patch").write_text(
        adapter_evidence(name), encoding="utf-8")
    (run_dir / "stdout.log").touch()
    (run_dir / "stderr.log").touch()
    try:
        status = ADAPTERS[name].preflight(config)
        if status.get("status") != "ready":
            failure_status = status.get("status", "not_attempted")
            failure = {"status": failure_status, "adapter": name,
                       "error": status.get("reason"), "preflight": status}
            write_json(run_dir / "failure.json", failure)
            return failure
        if mode == "fit":
            # Both phases run outside this coordinator.  The restore process
            # is not started until the source interpreter has fully exited.
            source_cmd = [sys.executable, str(ROOT / "train.py"), "_source",
                          "--config", str(run_dir / "config.json"),
                          "--adapter", name, "--run-dir", str(run_dir)]
            source_proc = subprocess.run(
                source_cmd, capture_output=True, text=True, check=False, cwd=ROOT,
                timeout=float(config["timeout_seconds"]))
            (run_dir / "stdout.log").write_text(
                source_proc.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(
                source_proc.stderr, encoding="utf-8")
            source_path = run_dir / "source.json"
            if source_proc.returncode or not source_path.exists():
                failure = {
                    "status": "roundtrip_failed", "adapter": name,
                    "stage": "source", "returncode": source_proc.returncode,
                    "error": "source subprocess did not complete",
                }
                write_json(run_dir / "failure.json", failure)
                return failure
            result = json.loads(source_path.read_text(encoding="utf-8"))
            if name == "none":
                result["restore"] = {"status": "not_applicable",
                                      "reason": "training baseline has no checkpoint"}
                write_json(run_dir / "result.json", result)
                return result
            restore_cmd = [sys.executable, str(ROOT / "train.py"), "verify-restart",
                           "--config", str(run_dir / "config.json"),
                           "--adapter", name, "--run-dir", str(run_dir)]
            restore_proc = subprocess.run(
                restore_cmd, capture_output=True, text=True, check=False, cwd=ROOT,
                timeout=float(config["timeout_seconds"]))
            (run_dir / "restore.stdout.log").write_text(
                restore_proc.stdout, encoding="utf-8")
            (run_dir / "restore.stderr.log").write_text(
                restore_proc.stderr, encoding="utf-8")
            restore_path = run_dir / "restore.json"
            if restore_path.exists():
                restore = json.loads(restore_path.read_text())
            else:
                restore = {"status": "restore_failed",
                           "error": "restore subprocess produced no restore.json",
                           "returncode": restore_proc.returncode}
            result["restore"] = restore
            if restore_proc.returncode != 0 or restore.get("status") != "pass":
                result["status"] = "restore_failed"
            write_json(run_dir / "result.json", result)
            return result
        raise ValueError(f"unsupported mode {mode}")
    except DependencyBlocked as error:
        failure = {"status": "dependency_blocked", "adapter": name,
                   "error": repr(error), "next_step": "prepare locked dependency"}
        write_json(run_dir / "failure.json", failure)
        return failure
    except Exception as error:
        failure = {"status": "build_failed", "adapter": name,
                   "error": repr(error)}
        write_json(run_dir / "failure.json", failure)
        return failure


def prepare_fixture(config, config_path):
    root = Path(config['results_root']) / 'prepare'
    if (root / 'state_schema.json').is_file():
        previous = json.loads((root / 'config.json').read_text())
        for key in ('project_commit','model','seed','input_tokens','dropout','warmup_steps','formal_steps','continue_steps'):
            if previous[key] != config[key]:
                raise ValueError(f'fixture configuration differs: {key}')
        return
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / 'config.json'
    write_json(frozen, config)
    with (root / 'prepare.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'train.py'), '_prepare', '--config', str(frozen)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       timeout=float(config['timeout_seconds']), check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight','fit','benchmark','verify-restart','inspect','_prepare','_source'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--config', required=True)
        cmd.add_argument('--adapter', choices=tuple(ADAPTERS))
        cmd.add_argument('--all', action='store_true')
        cmd.add_argument('--run-dir')
        cmd.add_argument('--generation', default='latest-committed')
        cmd.add_argument('--continue-steps', type=int)
        cmd.add_argument('--output')
        cmd.add_argument('--mode', choices=('verify','timing'), default='verify')
        cmd.add_argument('--repeat-index', type=int)
        cmd.add_argument('--continue-on-failure', action='store_true')
        cmd.add_argument('--runtime-probe', action='store_true')
        cmd.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.all and args.adapter:
            parser.error('choose --all or --adapter')
        names = list(ADAPTERS) if args.all else [args.adapter or 'ours']
        if args.dry_run:
            print(json.dumps({'command':args.command, 'methods':{n:method_config(config,n) for n in names},
                              'capture':'frozen','native_library':native_identity(config)}, indent=2))
            return 0
        if args.command == 'preflight':
            report = preflight(config, names, runtime_probe=args.runtime_probe)
            return int(any(r['status'] != 'ready' for r in report['adapters'].values()))
        if args.command == '_prepare':
            prepare_run(config, Path(config['results_root']) / 'prepare')
            return 0
        if args.command in ('fit','benchmark'):
            prepare_fixture(config, args.config)
            failures = []
            for name in names:
                result = run_one(method_config(config, name), name)
                print(json.dumps(result, default=str))
                if result.get('status') != 'trend_measured':
                    failures.append(name)
                    if not args.continue_on_failure: break
            return int(bool(failures))
        if args.command in ('_source','verify-restart'):
            if not args.adapter or not args.run_dir:
                parser.error('this phase requires --adapter and --run-dir')
            config = method_config(config, args.adapter)
            if args.command == '_source':
                result = train_run(config, args.adapter, args.run_dir)
            else:
                result = restore_run(config, args.adapter, args.run_dir, generation=args.generation,
                    continue_steps=args.continue_steps, output_path=args.output, mode=args.mode,
                    repeat_index=args.repeat_index)
            print(json.dumps(result, default=str))
            return int(result['status'] not in ('pass','trend_measured','timing_measured'))
        if args.command == 'inspect':
            if args.run_dir:
                result = build_inventory(config, {names[0]:args.run_dir})
            else:
                from npu_nvme.storage.inspection import inspect_disk
                result = inspect_disk(config['raw_pci'], config['npu_device'])
            print(json.dumps(result, default=str, indent=2))
            return 0
    except Exception as error:
        print(json.dumps({'status':'fail','error':repr(error)}), file=sys.stderr)
        return 1

