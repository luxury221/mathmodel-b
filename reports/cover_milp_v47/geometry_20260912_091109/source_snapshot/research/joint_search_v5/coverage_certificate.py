from __future__ import annotations

import itertools

import numpy as np
import shapely
from shapely.geometry import Polygon
from plan_geometry import disk_polygon, initial_outer_region


class TriangleCoverage:
    def __init__(self, receiving_radius=999.9, triangle_margin=1e-6):
        self.receiving_radius = receiving_radius
        self.triangle_margin = triangle_margin
        self.observations = []
        self.disks = []
        self.region = initial_outer_region()
        self.pair_lenses = {}
        self.pieces = 0
        self.clear_observations = []

    def additional_cover(self, position):
        position = np.asarray(position)
        disk = disk_polygon(position, self.receiving_radius, 256)
        near = [index for index, point in enumerate(self.observations)
                if np.linalg.norm(point - position) < 2 * self.receiving_radius]
        pieces = []
        for first, second in itertools.combinations(near, 2):
            pair = (first, second)
            if pair not in self.pair_lenses:
                self.pair_lenses[pair] = self.disks[first].intersection(self.disks[second])
            lens = self.pair_lenses[pair]
            if lens.is_empty:
                continue
            triangle = Polygon([self.observations[first], self.observations[second], position])
            if not triangle.is_valid or triangle.area <= 1e-5:
                continue
            triangle = triangle.buffer(-self.triangle_margin, join_style='mitre')
            piece = triangle.intersection(lens).intersection(disk)
            if not piece.is_empty:
                pieces.append(piece)
        return shapely.union_all(pieces) if pieces else Polygon(), disk

    def gain(self, position):
        additional, _disk = self.additional_cover(position)
        return self.region.intersection(additional).area

    def observe_absence(self, position):
        position = np.asarray(position).copy()
        if any(np.array_equal(position, previous) for previous in self.observations):
            return
        additional, disk = self.additional_cover(position)
        self.region = self.region.difference(additional)
        remaining_near = self.region.intersection(disk)
        if not remaining_near.is_empty:
            near = [index for index, previous in enumerate(self.disks) if previous.intersects(remaining_near)]
            quads = []
            for first, second, third in itertools.combinations(near, 3):
                common = disk.intersection(self.disks[first]).intersection(self.disks[second]).intersection(self.disks[third])
                if common.is_empty:
                    continue
                hull = shapely.MultiPoint([position, self.observations[first], self.observations[second], self.observations[third]]).convex_hull
                if hull.geom_type != 'Polygon':
                    continue
                interior = hull.buffer(-self.triangle_margin, join_style='mitre')
                piece = common.intersection(interior)
                if not piece.is_empty:
                    quads.append(piece)
            if quads:
                self.region = self.region.difference(shapely.union_all(quads))
        if not self.region.is_valid:
            raise ValueError('Invalid continuous coverage remainder')
        self.observations.append(position)
        self.disks.append(disk)
        self.pieces += int(not additional.is_empty)

    def observe_clear_absence(self, position):
        self.region = self.region.difference(disk_polygon(position, 19.99, 128))
        self.clear_observations.append(np.asarray(position).copy())

    @property
    def empty(self):
        return self.region.is_empty

    @property
    def area(self):
        return self.region.area
