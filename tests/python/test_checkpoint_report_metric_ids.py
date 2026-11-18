"""CPU SQLite regressions for per-export metric IDs and per-stack HBM totals."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def database(path):
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE STRING_IDS (id INTEGER, value TEXT);
        CREATE TABLE SESSION_TIME_INFO (startTimeNs INTEGER, endTimeNs INTEGER);
        CREATE TABLE SAMPLE_PMU_TIMELINE (deviceId INTEGER, timestampNs INTEGER,
                                         usage REAL, coreType INTEGER);
        CREATE TABLE HBM (deviceId INTEGER, timestampNs INTEGER,
                          bandwidth REAL, hbmId INTEGER, type INTEGER);
    ''')
    conn.executemany('INSERT INTO STRING_IDS VALUES (?,?)',
                     [(42, 'AIC'), (43, 'AIV'), (56, 'read'), (57, 'write')])
    conn.execute('INSERT INTO SESSION_TIME_INFO VALUES (?,?)', (2_000_000_000, 22_000_000_000))
    for timestamp in (2_010_000_000, 2_030_000_000):
        conn.executemany('INSERT INTO SAMPLE_PMU_TIMELINE VALUES (?,?,?,?)',
                         [(4, timestamp, 12.5, 42), (4, timestamp, 6.25, 43)])
        for stack in range(4):
            conn.executemany('INSERT INTO HBM VALUES (?,?,?,?,?)',
                             [(4, timestamp, 1024**2, stack, 56),
                              (4, timestamp, 2*1024**2, stack, 57)])
    conn.commit()
    conn.close()


def test_control_summary_uses_export_strings_and_sums_stacks(tmp_path):
    db = tmp_path / 'capture.db'
    database(db)
    spec = importlib.util.spec_from_file_location('m1_report', ROOT/'tools/checkpoint_io_m1_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window = dict(mode='isolated', repeat=0, realtime_begin_ns=2_000_000_000,
                  realtime_end_ns=22_000_000_000, elapsed_ms=20_000,
                  aic_iterations=1, aiv_iterations=0)
    result = tmp_path / 'result.json'
    result.write_text(json.dumps(dict(windows=[window])))
    rows, _ = module.write_control(result, db, tmp_path)
    assert rows[0]['aic_pmu_mean_pct'] == 12.5
    assert rows[0]['aiv_pmu_mean_pct'] == 6.25
    assert rows[0]['hbm_read_mean_mib_s'] == 4
    assert rows[0]['hbm_write_mean_mib_s'] == 8
    assert rows[0]['hbm_read_samples'] == 2


def test_checkpoint_phase_summary_uses_export_strings(tmp_path):
    db = tmp_path / 'capture.db'
    database(db)
    manifest = tmp_path / 'captures.json'
    capture = dict(database=str(db), launch_realtime_ns=2_000_000_000,
                   launch_monotonic_ns=1_000_000_000,
                   trigger=dict(timestamp_ns=950_000_000))
    manifest.write_text(json.dumps(dict(status='pass', captures=[
        dict(capture, phase='restore'), dict(capture, phase='save')])))
    out = tmp_path / 'out'
    subprocess.run([sys.executable, str(ROOT/'tools/checkpoint_io_phase_summary.py'),
                    '--captures', str(manifest), '--output', str(out)], check=True,
                   capture_output=True, text=True)
    rows = json.loads((out/'checkpoint-phase-summary.json').read_text())
    assert len(rows) == 2
    for row in rows:
        assert row['aic_usage_pct']['mean'] == 12.5
        assert row['aiv_usage_pct']['mean'] == 6.25
        assert row['card_hbm_read_mib_s']['mean'] == 4
        assert row['card_hbm_write_mib_s']['mean'] == 8
        assert row['card_hbm_read_mib_s']['sample_times'] == 2
        assert row['session_start_after_trigger_seconds'] == .05
