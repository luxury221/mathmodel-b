from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive image')
    data = json.loads(args.analysis.read_text(encoding='utf-8'))
    profiles = ['random', 'boundary', 'clustered', 'near_origin', 'max_radius', 'adversarial_heading']
    labels = ['Random', 'Boundary', 'Clustered', 'Near origin', 'Max radius', 'Adversarial']
    binding = {row['profile']: -row['saved_seconds_per_source'] for row in data['paired_cases'] if row['mode'] == 'station_bind'}
    nudge = {row['profile']: -row['saved_seconds_per_source'] for row in data['paired_cases'] if row['mode'] == 'coverage_nudge'}
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), gridspec_kw={'width_ratios': [1.2, 1]})
    positions = np.arange(len(profiles))
    axes[0].bar(positions, [binding[profile] for profile in profiles], color='#db8a35', width=0.55, label='Bind probe to station')
    axes[0].scatter(positions, [nudge[profile] for profile in profiles], color='#267bb5', s=42, zorder=4, label='Coverage-constrained nudge')
    for index, profile in enumerate(profiles):
        if binding[profile] > 0:
            axes[0].text(index, binding[profile] + 0.45, f'+{binding[profile]:.2f}', ha='center', fontsize=10)
    axes[0].axhline(0, color='#777777', linewidth=0.8)
    axes[0].set_xticks(positions, labels, rotation=20, ha='right')
    axes[0].set(ylabel='Added complete time (s / source); lower is better', ylim=(-1.2, 18), title='Actual mission result: no improvement')
    axes[0].legend(loc='upper left', fontsize=8.7, frameon=False)
    axes[0].grid(axis='y', alpha=0.15)
    counts = [1319, 604, 286, 0]
    names = ['Point-generation attempts', 'Actual optimizer calls', 'Known-source proof passed', 'Full patrol coverage passed']
    axes[1].barh(np.arange(4), counts, color=['#aab9c4', '#839cad', '#53798d', '#c2514b'], height=0.55)
    for index, count in enumerate(counts):
        axes[1].text(count + 25, index, str(count), va='center', color='#a93632' if count == 0 else '#333333')
    axes[1].set_yticks(np.arange(4), names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 1510), xlabel='Calls / checks, not independent cases', title='Why generated probe points were rejected')
    axes[1].grid(axis='x', alpha=0.12)
    figure.suptitle('Q4 v39: local geometric validity did not reduce mission time', fontsize=14, y=0.97)
    figure.text(0.5, 0.025, 'Offline only. All 24 missions cleared every source. Complete time includes final confirmation.', ha='center', fontsize=9)
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.82, wspace=0.54)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
