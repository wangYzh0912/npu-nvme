"""Serial hardware orchestration for the phase-one feasibility campaign."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import statistics
import shutil

from npu_nvme.cli.incremental_feasibility import preflight, write


ROOT = Path(__file__).resolve().parents[3]


def validate_training_run(path, config):
    from npu_nvme.runtime.training_catalog import read_checked
    reports=[read_checked(path/f'rank_{rank}'/'training.json') for rank in range(4)]
    for rank, report in enumerate(reports):
        experiment=report.get('incremental', {})
        if (report.get('rank')!=rank or report.get('status')!='pass' or
                not report.get('initial_full_restore', {}).get('verified') or
                len(report.get('losses', []))!=config['warmup_steps']+config['formal_steps'] or
                experiment.get('formal_end_ns', 0)<=experiment.get('formal_begin_ns', 0)):
            raise ValueError('formal run contract incomplete: '+str(path))
    return reports


def cleanup_regenerable_cache(path):
    cache=Path(path)/'training'/'qwen3_ms_converted_weight'
    if cache.exists():shutil.rmtree(cache)


def validate_baseline(paths, config):
    reports=[validate_training_run(Path(path),config) for path in paths]
    times=[(max(r['incremental']['formal_end_ns'] for r in rows)-
            min(r['incremental']['formal_begin_ns'] for r in rows))/1e9 for rows in reports]
    losses=[[r['loss'] for r in rows[0]['losses'][config['warmup_steps']:]] for rows in reports]
    oracle=losses[0]
    for row in losses:
        if any(abs(a-b)>config['loss_atol']+config['loss_rtol']*abs(a) for a,b in zip(oracle,row)):
            raise ValueError('B0 numerical trajectory differs between repeats')
    spread=(max(times)-min(times))/statistics.median(times)
    return dict(seconds=times,median_seconds=statistics.median(times),spread_fraction=spread,
                numerics='pass',stable=len(times)>=2 and spread<=config['baseline_spread_limit'])


def _base_training(config, role):
    from npu_nvme.runtime.qwen_config import resolve
    value = json.loads((ROOT / "config/qwen_training.json").read_text())
    value.update(model=config["models"][role], method="none",
                 stop_step=config["warmup_steps"] + config["formal_steps"],
                 lr_horizon=config["warmup_steps"] + config["formal_steps"],
                 checkpoint_interval=config["warmup_steps"] + config["formal_steps"] + 1)
    return resolve(value, ROOT)


def _owner(config, campaign, run_id, owner_dir):
    owner_dir.parent.mkdir(parents=True,exist_ok=True)
    connection = {
        "library": config["library"], "pci_addr": config["raw_pci_addr"],
        "chunk_bytes": config["transport_chunk_bytes"],
        "timeout_seconds": 14400, "config_sha256": config["config_sha256"],
        "campaign_id": campaign.name, "catalog": str(owner_dir / "raw-catalog.json"),
        "maximum_run_bytes": (36 << 30) * config['formal_steps'],
        "socket": f"/tmp/incremental-{run_id}.sock",
    }
    path = owner_dir.parent / (owner_dir.name + "-connection.json"); write(path, connection)
    manifest = config["environment_manifest"]
    command = [sys.executable, str(ROOT / "scripts/run_user_environment.py"), "--manifest", manifest,
        "--profile", "candidate", "--", "python", str(ROOT / "tools/incremental_raw_owner.py"),
        "--connection", str(path), "--out", str(owner_dir)]
    log = (owner_dir.parent / (owner_dir.name + ".log")).open("w")
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + 180
    while not (owner_dir / "ready.json").exists():
        if process.poll() is not None:
            log.close(); raise RuntimeError("incremental raw owner initialization failed")
        if time.monotonic() >= deadline:
            process.terminate(); log.close(); raise TimeoutError("incremental raw owner readiness")
        time.sleep(.1)
    return process, log, connection


def _run_one(config, campaign, group, repeat, *, canonical=False, role="main", profile=False, probe=None):
    from npu_nvme.cli.qwen_training import run_fit
    name = f"{role}-{group.lower()}-{'canonical' if canonical else f'rep{repeat}'}"
    if profile: name += '-profile'
    if probe: name += '-'+probe['name']
    base_name=name;attempt=0
    output = campaign / "runs" / name
    while output.exists():
        result = json.loads((output / "result.json").read_text())
        if result.get("validation_status") == "pass":
            validate_training_run(output,config)
            return output
        if result.get('execution_status')!='completed':
            raise RuntimeError('existing run still owns resources: '+str(output))
        attempt+=1;name=f'{base_name}-attempt{attempt}';output=campaign/'runs'/name
    training = _base_training(config, role); run_id = uuid.uuid4().hex
    if canonical: training['timeout_seconds']=43200
    ratio = {"K5": .05, "K10": .10, "K20": .20}.get(group)
    experiment = {
        "group": group, "ratio": ratio, "run_id": run_id,
        "warmup_steps": config["warmup_steps"], "formal_steps": config["formal_steps"],
        "block_elements": config["block_elements"], "hbm_staging_bytes": config["hbm_staging_bytes"],
        "chunk_bytes": config["transport_chunk_bytes"], "library": config["library"],
        "strategy": training["strategy"], "timeout_seconds": training["timeout_seconds"],
        "initial_full": str(campaign / f"initial-full-{role}"),
        "initial_identity": config["config_sha256"] + ':' + role,
        "model_role": role,
        "profile": profile,
        "canonical": canonical,
        "native_scheduler":{name:os.environ.get(name,'65536') for name in ('NPU_NVME_COPY_BYTES','NPU_NVME_CHECKSUM_BYTES')},
    }
    if probe: experiment["probe"]=probe
    owner = log = connection = None
    if group != "B0" or (probe and probe.get("stage",0)>=5):
        owner_dir = campaign / "owners" / name
        owner_dir.parent.mkdir(parents=True,exist_ok=True)
        connection=dict(library=config['library'],pci_addr=config['raw_pci_addr'],chunk_bytes=config['transport_chunk_bytes'],
            timeout_seconds=training['timeout_seconds'],config_sha256=config['config_sha256'],campaign_id=campaign.name,
            catalog=str(owner_dir/'raw-catalog.json'),maximum_run_bytes=(36<<30)*config['formal_steps'],
            socket=f'/tmp/incremental-{run_id}.sock',verify_before_ack=canonical,native_scheduler=experiment['native_scheduler'])
        connection_path=owner_dir.parent/(owner_dir.name+'-connection.json');write(connection_path,connection)
        if canonical:
            connection.update(initial_full=experiment['initial_full'],strategy=experiment['strategy'],block_elements=experiment['block_elements'],media_shadow=str(owner_dir/'shadow'))
            experiment['media_shadow']=connection['media_shadow']
            write(connection_path,connection)
        experiment["socket"] = connection["socket"]
        experiment['owner_connection']=str(connection_path);experiment['owner_output']=str(owner_dir)
    training["incremental_experiment"] = experiment
    monitor_dir=campaign/'memory'/name;monitor_dir.mkdir(parents=True,exist_ok=True)
    stop_monitor=monitor_dir/'stop'
    stop_monitor.unlink(missing_ok=True)
    monitor=subprocess.Popen([sys.executable,str(ROOT/'tools/incremental_memory_monitor.py'),
        '--output',str(monitor_dir/'samples.jsonl'),'--stop-file',str(stop_monitor)],cwd=ROOT,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        rc = run_fit(training, output)
        if rc: raise RuntimeError("incremental training run failed: " + name)
        validate_training_run(output,config)
        if connection:
            owner_result=json.loads((owner_dir/'result.json').read_text())
            receipt_catalog=json.loads((owner_dir/'raw-catalog.json').read_text())
            if [r['step'] for r in receipt_catalog['commits']]!=list(range(1,config['formal_steps']+1)):
                raise ValueError('raw owner did not commit each formal step')
            if owner_result.get('status')!='pass' or not owner_result.get('closed'):
                raise RuntimeError('incremental raw owner failed: '+name)
            if canonical and not all(r.get("media_verified") is True for r in json.loads((owner_dir/"raw-catalog.json").read_text())["commits"]):
                command=[sys.executable,str(ROOT/'scripts/run_user_environment.py'),'--manifest',config['environment_manifest'],
                    '--profile','candidate','--','python',str(ROOT/'tools/incremental_verify_media.py'),
                    '--connection',str(owner_dir.parent/(owner_dir.name+'-connection.json')),'--out',str(owner_dir/'verification')]
                subprocess.run(command,cwd=ROOT,check=True,timeout=training['timeout_seconds'])
        cleanup_regenerable_cache(output)
        return output
    finally:
        stop_monitor.touch()
        try:monitor.wait(timeout=15)
        except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=10)
        if owner and owner.poll() is None: owner.terminate()
        if owner:
            try: owner.wait(timeout=30)
            except subprocess.TimeoutExpired: owner.kill(); owner.wait()
        if log: log.close()


def _phase_runs(config, phase):
    if phase == "p0": return [("B0", repeat, False, "main") for repeat in range(config["performance_repeats"])]
    if phase == "p1": return [("B0", 0, False, "main")]
    if phase == "p3":
        return [(group, repeat, False, "main") for repeat in range(config["performance_repeats"])
                for group in ("B1", "K5", "K10", "K20")]
    if phase == "p4": return [(group, 0, True, "main") for group in ("B1", "K5", "K10", "K20")]
    if phase == "p6": return [(group, repeat, False, "auxiliary") for repeat in range(config['performance_repeats'])
                              for group in ("B0", "B1", "K5", "K10", "K20")]
    return []


def probe_specs(config, phase):
    import math
    from npu_nvme.experiments.tp_blocks import logical_fragments
    schema=json.loads((ROOT/'config/qwen_runtime_schema.json').read_text())
    elements=sum(math.prod(row['global_shape']) for row in schema['tensors'] if row['role']=='model')//4
    if phase=='p2':
        rows=[dict(name=f'{kind}-{factor}',kind=kind,factor=factor,target_elements=elements)
              for kind in ('score','score_select') for factor in (.5,1.,2.)]
        rows += [dict(name=f'copy-{factor}-{granularity}',kind='copy',factor=factor,
                      output_bytes=math.ceil(elements*4*.2*factor/4096)*4096,granularity=granularity)
                 for factor in (.5,1.,2.) for granularity in config['transport_probe_granularities']]
        return rows
    if phase=='p5':
        return [dict(name=f'ablation-{stage}',kind='ablation',stage=stage,target_elements=elements,
                     output_bytes=math.ceil(elements*4*.1/4096)*4096,granularity=4<<20)
                for stage in range(1,7)]
    return []


def run(config, output, *, phase="all", resume=False, performance_repeats=1, max_repeats=2):
    if not 1 <= performance_repeats <= max_repeats <= 2:
        raise ValueError("effective repetitions must satisfy 1 <= repeats <= max_repeats <= 2")
    # Scheduling is deliberately separate from immutable initial FULL identity.
    schedule=dict(config,performance_repeats=performance_repeats,
                  extra_repeats_on_instability=max_repeats-performance_repeats)
    selected_phases = [phase] if phase != "all" else [f"p{value}" for value in range(7)]
    needs_raw = any(any(group != 'B0' for group, _, _, _ in _phase_runs(schedule, item))
                    for item in selected_phases)
    needs_raw = needs_raw or 'p5' in selected_phases
    if needs_raw and os.geteuid() != 0:
        raise PermissionError("incremental raw phases must run as root")
    output = Path(output).resolve()
    if output.exists() and not resume: raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    audit = preflight(config); write(output / "preflight.json", audit)
    manifest_path = output / "campaign.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "status": "running", "config_sha256": config["config_sha256"], "phases": {}}
    if manifest["config_sha256"] != config["config_sha256"]:
        raise ValueError("campaign resume configuration differs")
    manifest["execution_policy"] = dict(performance_repeats=performance_repeats, max_repeats=max_repeats,
        initial_identity_preserved=True)
    phases = selected_phases
    try:
        for selected in phases:
            row = manifest["phases"].setdefault(selected, {"status": "running", "runs": []})
            for group, repeat, canonical, role in _phase_runs(schedule, selected):
                path = _run_one(config, output, group, repeat, canonical=canonical, role=role, profile=selected=='p1')
                if str(path) not in row["runs"]: row["runs"].append(str(path))
                write(manifest_path, manifest)
            for probe in probe_specs(config,selected):
                path=_run_one(config,output,'B0',0,probe=probe)
                if str(path) not in row['runs']:row['runs'].append(str(path))
                write(manifest_path,manifest)
            if not _phase_runs(schedule, selected):
                row['status']='validation_pending' if probe_specs(config,selected) else 'not_implemented' 
            else:
                row["status"] = "pilot_completed" if selected == 'p0' else "validation_pending"
            if selected=='p0':
                row['baseline']=validate_baseline(row['runs'], config)
                if not row['baseline']['stable']:
                    for repeat in range(performance_repeats,max_repeats):
                        path=_run_one(config,output,'B0',repeat)
                        if str(path) not in row['runs']:row['runs'].append(str(path))
                    row['baseline']=validate_baseline(row['runs'], config)
                row['status']='pass' if row['baseline']['stable'] else 'inconclusive_variance'
            elif selected=='p1':
                row['status']='validation_pending'
            elif selected=='p6':
                path=_run_one(config,output,'B0',0,role='auxiliary',profile=True)
                if str(path) not in row['runs']:row['runs'].append(str(path))
            write(manifest_path, manifest)
        manifest["status"] = "completed" if set(manifest["phases"]) == {f'p{i}' for i in range(7)} and all(row.get("status") == "pass" for row in manifest["phases"].values()) else "partial"
        write(manifest_path, manifest); return 0
    except BaseException as error:
        manifest.update(status="failed", error=repr(error)); write(manifest_path, manifest); raise
