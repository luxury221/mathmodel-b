from __future__ import annotations

import math

import numpy as np
import shapely
from plan_geometry import disk_polygon, initial_outer_region
from shapely.geometry import Polygon
from geometry import clip_polygon, initial_region


class DirectionalCoverage:
    def __init__(self, bins=36):
        self.bins = bins
        self.regions = [initial_outer_region() for _ in range(bins)]
        self.offsets = []
        self.observations = []
        half_bin = 180.0 / bins
        for index in range(bins):
            angles = np.deg2rad(np.linspace(index * 360.0 / bins + 90 + half_bin + 0.0001,
                                           index * 360.0 / bins + 270 - half_bin - 0.0001, 97))
            self.offsets.append(np.vstack(([0.0, 0.0], 999.9 * np.column_stack((np.cos(angles), np.sin(angles))))))

    def cuts(self, position):
        return [Polygon(offsets + np.asarray(position)) for offsets in self.offsets]

    def gain(self, position):
        return sum(region.intersection(cut).area for region, cut in zip(self.regions, self.cuts(position)))

    def observe_absence(self, position):
        self.regions = [region.difference(cut) for region, cut in zip(self.regions, self.cuts(position))]
        if not all(region.is_valid for region in self.regions):
            raise ValueError('Invalid directional absence geometry')
        self.observations.append(np.asarray(position).copy())

    @property
    def empty(self):
        return all(region.is_empty for region in self.regions)

    @property
    def area(self):
        return sum(region.area for region in self.regions) / self.bins

    def contains_hypothesis(self, position, heading_deg):
        if heading_deg is None:
            return any(region.covers(shapely.Point(position)) for region in self.regions)
        index = int(math.floor((heading_deg % 360.0) / (360.0 / self.bins) + 0.5)) % self.bins
        return self.regions[index].covers(shapely.Point(position))


class OmniCoverage:
    def __init__(self):
        self.region = initial_outer_region()
        self.observations = []

    def gain(self, position):
        return self.region.intersection(disk_polygon(position, 999.9)).area

    def observe_absence(self, position):
        self.region = self.region.difference(disk_polygon(position, 999.9))
        self.observations.append(np.asarray(position).copy())

    @property
    def empty(self):
        return self.region.is_empty

    @property
    def area(self):
        return self.region.area


class DirectionalBelief:
    def __init__(self, bins=36):
        self.coverage = DirectionalCoverage(bins)
        self.omni = initial_outer_region()
        self.positive_count = 0
        self.negative_count = 0

    def update(self, region, positives, negatives):
        for position in negatives[self.negative_count:]:
            self.coverage.observe_absence(position)
            self.omni = self.omni.difference(disk_polygon(position, 999.9))
        for position, _bearing in positives[self.positive_count:]:
            for index, previous in enumerate(self.coverage.regions):
                allowed = []
                for angle in np.deg2rad([index * 360.0 / self.coverage.bins - 180.0 / self.coverage.bins,
                                         index * 360.0 / self.coverage.bins + 180.0 / self.coverage.bins]):
                    normal = np.array([np.cos(angle), np.sin(angle)])
                    vertices = clip_polygon(initial_region(), normal, np.dot(normal, position) + 1e-6)
                    allowed.append(shapely.MultiPoint(vertices).convex_hull if len(vertices) else Polygon())
                self.coverage.regions[index] = previous.intersection(shapely.union_all(allowed))
        self.positive_count = len(positives)
        self.negative_count = len(negatives)
        self.coverage.regions = [possible.intersection(region) for possible in self.coverage.regions]
        self.omni = self.omni.intersection(region)
        posterior = shapely.union_all([self.omni, *self.coverage.regions])
        if posterior.geom_type == 'GeometryCollection':
            posterior = posterior.convex_hull
        if posterior.is_empty or not posterior.is_valid:
            raise ValueError('Directional observations contradict conservative geometry')
        return posterior
