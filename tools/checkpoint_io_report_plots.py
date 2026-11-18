#!/usr/bin/env python3
"""Plot recorded checkpoint stages and partial hardware sample windows."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    restore = json.loads((a.results / 'step52-restore-event-summary.json').read_text())
    stages = [s for s in restore['stages'] if s['stage'] == 'owner_chunk_read_verify_send_ack']
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    for row in stages:
        rank = row['rank']
        ax.barh(rank, restore['owner_total_seconds'], color='#e6e8ec', height=.6)
        ax.barh(rank, row['span_seconds'], left=row['begin_offset_seconds'],
                color=f'C{rank}', height=.6)
        ax.text(row['begin_offset_seconds'] + row['span_seconds']/2, rank,
                f"{row['span_seconds']:.1f} s", ha='center', va='center', color='white')
    ax.axvline(restore['owner_total_seconds'], color='#444', linestyle='--', linewidth=1)
    ax.set(yticks=range(4), yticklabels=[f'Rank {r} / NPU {r+4}' for r in range(4)],
           xlabel='Seconds after owner restore begin',
           title='Recorded step-52 restore: serial rank transfer windows')
    ax.invert_yaxis()
    ax.text(.01, -.21, 'Color: chunk read / verify / IPC / apply / ACK span. Gray: other rank work or readiness wait.',
            transform=ax.transAxes, fontsize=9)
    fig.savefig(a.results / 'restore-rank-timeline.png', dpi=160)
    plt.close(fig)

    phase_path = a.results / 'checkpoint-phase-timeseries-10ms.csv'
    if not phase_path.exists():
        return
    with phase_path.open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex='col', layout='constrained')
    for column, phase in enumerate(('restore', 'save')):
        selected = [r for r in rows if r['phase'] == phase]
        for key, label, color in (('aic_usage_pct', 'AIC', '#175ca4'),
                                  ('aiv_usage_pct', 'AIV', '#d57b19')):
            valid = [r for r in selected if r[key] != '']
            axes[0, column].plot([float(r['offset_seconds']) for r in valid],
                                  [float(r[key]) for r in valid], label=label, color=color)
        for key, label, color in (('card_hbm_read_mib_s', 'HBM read', '#245dbb'),
                                  ('card_hbm_write_mib_s', 'HBM write', '#bd4f37')):
            valid = [r for r in selected if r[key] != '']
            axes[1, column].plot([float(r['offset_seconds']) for r in valid],
                                  [float(r[key]) for r in valid], label=label, color=color,
                                  linewidth=.8)
        axes[0, column].set(title=f'NPU4 {phase}: partial 20-second capture', ylabel='Mean engine usage (%)')
        peak = max(float(r[key]) for r in selected for key in ('aic_usage_pct', 'aiv_usage_pct')
                   if r[key] != '')
        axes[0, column].set_ylim(0, max(1, peak*1.1))
        if peak == 0:
            axes[0, column].text(.5, .5, 'All sampled AIC/AIV values were zero',
                                  transform=axes[0, column].transAxes, ha='center', fontsize=10)
        axes[1, column].set(xlabel='Seconds after capture start', ylabel='Card bandwidth (MiB/s)')
        for ax in axes[:, column]:
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=.2)
    fig.savefig(a.results / 'checkpoint-phase-hardware-timeline.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
