from __future__ import annotations

import math

import numpy as np
import shapely
from geometry import clip_polygon
from shapely.geometry import MultiPoint, Polygon


MARGIN = 1e-5
DOMAIN_BOX = np.array([[-6000.0, -6000.0], [6000.0, -6000.0], [6000.0, 6000.0], [-6000.0, 6000.0]])


def conservative_update(original, updated):
    if updated.equals(original) or original.difference(updated).area <= 1e-6:
        return original
    return updated


def halfplane_region(normal, bound):
    vertices = clip_polygon(DOMAIN_BOX, np.asarray(normal), float(bound))
    return MultiPoint(vertices).convex_hull if len(vertices) else Polygon()


def range_bisector(positive, negative):
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    direction = negative - positive
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    normal = direction / length
    bound = float(direction @ (negative + positive) / (2 * length))
    return normal, bound


def guaranteed_visibility(negative, index, bins):
    vertices = DOMAIN_BOX.copy()
    for degrees in (index * 360 / bins - 180 / bins, index * 360 / bins + 180 / bins):
        angle = math.radians(degrees)
        normal = np.array([math.cos(angle), math.sin(angle)])
        vertices = clip_polygon(vertices, normal, float(normal @ negative) - MARGIN)
        if not len(vertices):
            return Polygon()
    return MultiPoint(vertices).convex_hull


class RangeCoupler:
    def __init__(self):
        self.applied = {}
        self.visibility_cache = {}
        self.pairs = 0
        self.effective_cuts = 0

    def fresh_pairs(self, state):
        applied = self.applied.setdefault(state.channel, set())
        for positive_index, (positive, _bearing) in enumerate(state.positives):
            for negative_index, negative in enumerate(state.negatives):
                pair = positive_index, negative_index
                if pair in applied:
                    continue
                applied.add(pair)
                self.pairs += 1
                bisector = range_bisector(positive, negative)
                if bisector is not None:
                    yield negative, bisector

    def visibility(self, negative, index, bins):
        key = tuple(negative), index, bins
        if key not in self.visibility_cache:
            self.visibility_cache[key] = guaranteed_visibility(np.asarray(negative), index, bins)
        return self.visibility_cache[key]

    def refine_omni(self, state):
        region = state.region
        for _negative, (normal, bound) in self.fresh_pairs(state):
            allowed = halfplane_region(normal, bound + MARGIN)
            if allowed.covers(region):
                continue
            updated = conservative_update(region, region.intersection(allowed))
            self.effective_cuts += int(not updated.equals(region))
            region = updated
        if region.is_empty or not region.is_valid:
            raise ValueError('Omnidirectional range coupling contradicted observations')
        return region

    def refine_directional(self, state, belief):
        touched = False
        bins = belief.coverage.bins
        for negative, (normal, bound) in self.fresh_pairs(state):
            allowed = halfplane_region(normal, bound + MARGIN)
            if not allowed.covers(belief.omni):
                previous = belief.omni
                belief.omni = previous.intersection(allowed)
                changed = not previous.equals(belief.omni)
                self.effective_cuts += int(changed)
                touched |= changed
            excluded = halfplane_region(-normal, -bound - MARGIN)
            for index, region in enumerate(belief.coverage.regions):
                if region.is_empty or not region.intersects(excluded):
                    continue
                impossible = excluded.intersection(self.visibility(negative, index, bins))
                if impossible.is_empty or not region.intersects(impossible):
                    continue
                updated = region.difference(impossible)
                changed = not updated.equals(region)
                if changed:
                    belief.coverage.regions[index] = updated
                    self.effective_cuts += 1
                    touched = True
        if not touched:
            return state.region
        union = shapely.union_all([belief.omni, *belief.coverage.regions]).intersection(state.region)
        if union.is_empty or not union.is_valid:
            raise ValueError('Directional range coupling contradicted observations')
        return conservative_update(state.region, union)
