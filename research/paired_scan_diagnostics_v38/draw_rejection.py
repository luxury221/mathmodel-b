from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from shapely.geometry import shape


def paint_region(axis, region):
    parts = list(region.geoms) if hasattr(region, 'geoms') else [region]
    for part in parts:
        if hasattr(part, 'exterior'):
            vertices = np.array(part.exterior.coords)
            axis.fill(vertices[:, 0], vertices[:, 1], color='#d55e55', alpha=0.6, zorder=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive image path')
    record = json.loads(args.diagnostic.read_text(encoding='utf-8'))['rejections'][0]
    region = shape(record['geometry'])
    receivers = np.array(record['radio_positions'])
    planned = np.array(record['planned_points'])
    observed_count = len(receivers) - 2 - len(record['kept'])
    actual = receivers[:observed_count]
    future = receivers[observed_count + 2:]
    witness = record['physical_witness']
    source = np.array(witness['position'])
    heading = witness['heading_deg']
    angle = math.radians(315)
    removed = 1870 * np.array([math.cos(angle), math.sin(angle)])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), gridspec_kw={'width_ratios': [1.08, 1]})
    for axis in axes:
        axis.set_aspect('equal')
        axis.grid(alpha=0.15)
        axis.set_xlabel('x (m)')
        axis.set_ylabel('y (m)')
        axis.add_patch(Circle((0, 0), 1800, fill=False, color='#64748b', linestyle='--', linewidth=1.2))
        axis.add_patch(Wedge(source, 1000, heading - 90, heading + 90, color='#aa55a0', alpha=0.07))
        paint_region(axis, region)
        axis.scatter(actual[:, 0], actual[:, 1], color='#778899', s=28, label='Actually scanned', zorder=3)
        axis.scatter(future[:, 0], future[:, 1], facecolors='white', edgecolors='#247bb3', s=65, label='Kept future stations', zorder=4)
        axis.scatter(planned[:, 0], planned[:, 1], marker='*', s=160, color='#e69f00', label='Proposed pair (not executed)', zorder=5)
        axis.scatter(*removed, marker='x', s=130, color='#d55e55', linewidths=2.3, label='Station proposed for deletion', zorder=6)
        axis.scatter(*source, marker='D', s=52, color='#94248c', label='Constructed missed source', zorder=7)
    axes[0].set(xlim=(-2130, 2130), ylim=(-2130, 2130), title='Continuous coverage still has a hole')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8.5, ncol=2, frameon=False)
    axes[1].set(xlim=(480, 1550), ylim=(-1870, -1100), title='The hole is physically real, not roundoff')
    direction = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    axes[1].annotate('', xy=source + 195 * direction, xytext=source,
                     arrowprops={'arrowstyle': '-|>', 'color': '#94248c', 'linewidth': 1.7})
    axes[1].annotate('Missed source\nheading 302.69 degrees', xy=source, xytext=(810, -1315), fontsize=9,
                     arrowprops={'arrowstyle': '-', 'color': '#94248c'}, color='#6f1f6a')
    axes[1].text(0.03, 0.96, f"Remaining area: {record['remaining_area']:,.1f} sq. m\n22 radio scans: all no-signal\n1 optical check: no target",
                 transform=axes[1].transAxes, fontsize=9, va='top', bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.92})
    figure.suptitle('Q4 paired-scan study: reject the tempting shortcut', fontsize=15, y=0.97)
    figure.text(0.5, 0.91, f"Proxy saving {record['proxy_saving_seconds']:.1f} s is invalid without full coverage; official runs used: 0",
                ha='center', fontsize=10, color='#444444')
    figure.subplots_adjust(left=0.065, right=0.98, top=0.83, bottom=0.2, wspace=0.24)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
