from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'offline_review'


def main():
    nodes = json.loads((OUTPUT / 'network_nodes.json').read_text(encoding='utf-8'))
    summary = json.loads((OUTPUT / 'offline_summary.json').read_text(encoding='utf-8'))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, name, title, route_name in zip(
        axes, ('grid25', 'dual21'), ('25-node triangular network', '21-node dual-ring network'),
        ('grid25_route', 'dual21_explicit_route'),
    ):
        points = np.asarray(nodes[name])
        route = np.vstack(([0, 0], nodes[route_name]))
        axis.add_patch(Circle((0, 0), 1800, facecolor='#e8eef5', edgecolor='#768392', alpha=0.6))
        axis.plot(route[:, 0], route[:, 1], color='#2864a8', linewidth=1.2, alpha=0.85)
        axis.scatter(points[:, 0], points[:, 1], color='#bd5a21', s=28, zorder=3)
        axis.scatter([0], [0], marker='*', color='#141e30', s=100, zorder=4)
        axis.set(title=title, xlabel='East (m)', ylabel='North (m)', aspect='equal')
        axis.grid(alpha=0.18)
    figure.savefig(OUTPUT / 'network_routes.png', dpi=170)
    plt.close(figure)
    labels = {
        'q3_grid7_cover': '7 / cover', 'q3_grid7_active': '7 / active',
        'q4_grid25_cover': '25 / cover', 'q4_grid25_active': '25 / active',
        'q4_dual21_cover': '21 / cover', 'q4_dual21_active': '21 / active',
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, problem in zip(axes, ('q3', 'q4')):
        records = [record for record in summary['summary'] if record['method'].startswith(problem)]
        records.sort(key=lambda record: record['mean_seconds_per_source'])
        means = [record['mean_seconds_per_source'] for record in records]
        quantiles = [record['p90_seconds_per_source'] for record in records]
        offsets = np.arange(len(records))
        axis.bar(offsets, means, color='#2864a8', width=0.62, label='Mean')
        axis.scatter(offsets, quantiles, color='#bd5a21', marker='D', s=35, label='90th percentile')
        axis.set_xticks(offsets, [labels[record['method']] for record in records])
        axis.set(title=f'{problem.upper()}: 56 paired synthetic scenes per method', ylabel='Virtual seconds / cleared source')
        axis.legend(frameon=False)
        axis.grid(axis='y', alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle('Offline prototype benchmark — not official competition results', fontsize=12)
    figure.savefig(OUTPUT / 'offline_performance.png', dpi=170)
    plt.close(figure)
    profile = pd.read_csv(OUTPUT / 'offline_by_profile.csv')
    profile = profile[profile.method.str.startswith('q4')]
    pivot = profile.pivot(index='profile', columns='method', values='mean_seconds_per_source')
    pivot = pivot.rename(columns=labels)
    axis = pivot.plot.bar(figsize=(10, 4.5), rot=0, color=['#185a7d', '#71b5c7', '#a85020', '#e6ae74'])
    axis.set(title='Q4 stratified synthetic results: improvements are not universal', ylabel='Virtual seconds / cleared source')
    axis.legend(title='Nodes / localization', frameon=False)
    axis.grid(axis='y', alpha=0.2)
    axis.set_axisbelow(True)
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT / 'q4_by_profile.png', dpi=170)
    plt.close(axis.figure)
    print('Created network_routes.png, offline_performance.png, q4_by_profile.png')


if __name__ == '__main__':
    main()
