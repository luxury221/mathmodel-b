from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from plan_geometry import hull_vertices
from shapely.geometry import Point


@dataclass
class ShapedProbe:
    anchor: np.ndarray
    candidates: np.ndarray
    lower_range: float
    forward: float
    lateral: float
    maximum_squared_range_change: float
    minimum_forward_gap: float
    minimum_cone_slack: float


def certified_shaped_probe(region, anchor, bearing_deg, fraction=0.85, aspect=0.35):
    if not 0 < fraction < 1 or not 0 < aspect <= 1:
        raise ValueError('Invalid shaped V parameters')
    anchor = np.asarray(anchor, dtype=float)
    lower = max(0.0, region.distance(Point(anchor)) - 1e-4)
    if lower < 1e-3:
        return None
    angle = math.radians(bearing_deg)
    direction = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([-direction[1], direction[0]])
    forward = fraction * lower
    lateral = aspect * forward
    vertices = hull_vertices(region)
    relative = vertices - anchor
    longitudinal = relative @ direction
    transverse = relative @ tangent
    gap = float(np.min(longitudinal - forward))
    cone_slack = float(np.min(lateral * longitudinal - forward * np.abs(transverse)))
    candidates = np.array([anchor + forward * direction + lateral * tangent,
                           anchor + forward * direction - lateral * tangent])
    old_squared = np.sum(relative**2, axis=1)
    new_squared = np.sum((candidates[:, None] - vertices[None, :])**2, axis=2)
    maximum_change = float(np.max(new_squared - old_squared[None, :]))
    if gap <= 1e-8 or cone_slack <= 1e-8 or maximum_change > -1e-8:
        return None
    return ShapedProbe(anchor.copy(), candidates, lower, forward, lateral, maximum_change, gap, cone_slack)
