from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from plan_geometry import hull_vertices
from shapely.geometry import Point


@dataclass
class VProbe:
    anchor: np.ndarray
    candidates: np.ndarray
    lower_range: float
    forward: float
    maximum_squared_range_change: float


def certified_v_probe(region, anchor, bearing_deg, fraction=0.7):
    anchor = np.asarray(anchor)
    lower = max(0.0, region.distance(Point(anchor)) - 1e-4)
    if lower < 1e-3:
        return None
    error = math.radians(1.005)
    if not 0 < fraction < min(math.cos(error), math.cos(error) - math.sin(error)):
        raise ValueError('V-step fraction violates its reception proof')
    angle = math.radians(bearing_deg)
    forward = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([-forward[1], forward[0]])
    step = fraction * lower
    candidates = np.array([anchor + step * (forward + tangent), anchor + step * (forward - tangent)])
    vertices = hull_vertices(region)
    old_squared = np.sum((vertices - anchor)**2, axis=1)
    new_squared = np.sum((candidates[:, None] - vertices[None, :])**2, axis=2)
    maximum_change = float(np.max(new_squared - old_squared[None, :]))
    if maximum_change > 1e-7:
        raise ValueError('V-step fails its continuous range certificate')
    return VProbe(anchor, candidates, lower, step, maximum_change)
