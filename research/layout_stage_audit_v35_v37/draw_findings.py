from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from shapely import from_wkt


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    hybrid = ROOT / 'reports/hybrid_discovery_v35/geometry_20260912_051747'
    repair = ROOT / 'reports/nonuniform_repair_v37/repair_20260912_053044'
    output = ROOT / 'reports/layout_stage_v35_v37/coverage_findings.png'
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    rows = [load(path) for path in (hybrid / 'certificates').glob('*.json')]
    complete = [row for row in rows if row['complete']]
    baseline = load(hybrid / 'baseline.json')['proxy_cost_meters']
    repaired = [load(path) for path in (repair / 'iterations').glob('*.json')]
    best = min((row for row in repaired if 'remaining_area' in row), key=lambda row: row['remaining_area'])
    region = from_wkt(best['remaining_geometry'])
    pieces = list(region.geoms) if hasattr(region, 'geoms') else [region]
    largest = max(pieces, key=lambda part: part.area)
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), constrained_layout=True)
    left, right = axes
    left.scatter([len(row['optical_centers']) for row in complete], [row['proxy_cost_meters'] for row in complete],
                 color='#226a9c', s=32, alpha=0.8)
    left.axhline(baseline, color='#9c4625', linestyle='--', label='原21点网')
    left.axhline(0.99 * baseline, color='#267158', linestyle=':', label='至少改善1%的准入门槛')
    left.set(title='混合补洞：完整布局仍未达到费用门槛', xlabel='额外光学补点数\n静态代理，非官方成绩', ylabel='几何代理费用（米等值）')
    left.set_xlim(-0.7, 17)
    left.grid(alpha=0.18)
    left.legend(loc='upper left', fontsize=9)
    horizontal, vertical = largest.exterior.xy
    right.fill(horizontal, vertical, facecolor='#ee9b83', edgecolor='#a62a14', linewidth=1.5)
    for interior in largest.interiors:
        horizontal, vertical = interior.xy
        right.fill(horizontal, vertical, facecolor='white', edgecolor='#a62a14', linewidth=0.7)
    lower_horizontal, lower_vertical, upper_horizontal, upper_vertical = largest.bounds
    horizontal_margin = max(10, 0.2 * (upper_horizontal - lower_horizontal))
    vertical_margin = max(10, 0.2 * (upper_vertical - lower_vertical))
    right.set_xlim(lower_horizontal - horizontal_margin, upper_horizontal + horizontal_margin)
    right.set_ylim(lower_vertical - vertical_margin, upper_vertical + vertical_margin)
    right.set_aspect('equal')
    right.grid(alpha=0.18)
    right.set(title='非规则20点修补：最大剩余缺口放大', xlabel='横坐标（米）', ylabel='纵坐标（米）')
    right.text(0.03, 0.04, f"全域剩余 {best['remaining_area']:.2f} 平方米\n图中只放大最大分片；全部残片均保留核验",
               transform=right.transAxes, fontsize=9, color='#802711', bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
    figure.suptitle('Q4几何诊断：没有以漏扫换取更低耗时', fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor='white')
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
