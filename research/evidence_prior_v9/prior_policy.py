from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
from plan_geometry import disk_polygon, hull_vertices
from policy_v5 import ContinuousPolicy


PRIOR_MODES = ('prior_route', 'prior_trial')


class PriorPolicy(ContinuousPolicy):
    def __init__(self, *args, mode='prior_route'):
        self.prior_mode = mode
        self.prior_cache = {}
        self.prior_events = []
        self.tried_channels = set()
        super().__init__(*args, mode='certified_center')
        self.stats.update({'prior_centers': 0, 'prior_trials': 0, 'prior_hits': 0, 'prior_cluster': 0, 'prior_radial': 0})

    def estimate(self, state):
        if state.status != 'DETECTED' or len(self.clear_positions) < 3:
            return None
        key = (state.channel, state.revision, len(self.clear_positions))
        if key in self.prior_cache:
            return self.prior_cache[key]
        points = np.asarray(self.clear_positions)
        neighborhoods = [points[np.linalg.norm(points - point, axis=1) <= 60] for point in points]
        neighbors = max(neighborhoods, key=len)
        candidate_region = None
        kind = None
        if len(neighbors) >= 3:
            center = np.mean(neighbors, axis=0)
            width = max(80.0, float(np.linalg.norm(neighbors - center, axis=1).max()) + 40)
            candidate_region = state.region.intersection(disk_polygon(center, width, 256))
            kind = 'cluster'
        if candidate_region is None or candidate_region.is_empty:
            radii = np.linalg.norm(points, axis=1)
            median = float(np.median(radii))
            deviation = float(np.median(np.abs(radii - median)))
            if median > 1500 and deviation < 60:
                width = max(80.0, 2 * deviation + 40)
                annulus = disk_polygon([0, 0], min(1840.0, median + width), 256)
                annulus = annulus.difference(disk_polygon([0, 0], max(0.0, median - width), 256))
                candidate_region = state.region.intersection(annulus)
                kind = 'radial'
        if candidate_region is None or candidate_region.is_empty or candidate_region.area < 1e-8:
            self.prior_cache[key] = None
            return None
        point = candidate_region.centroid
        if not state.region.covers(point):
            point = nearest_points(state.region, point)[0]
        proposal = np.asarray(point.coords[0])
        self.prior_cache[key] = (proposal, kind)
        self.stats['prior_' + kind] += 1
        self.prior_events.append({'channel': state.channel, 'revision': state.revision,
                                  'cleared_observations': len(points), 'kind': kind, 'proposal': proposal.tolist(),
                                  'true_belief_area_unchanged': state.region.area, 'proposal_area': candidate_region.area})
        return proposal, kind

    def center_radius(self, state):
        original, radius = super().center_radius(state)
        estimate = self.estimate(state)
        if estimate is None or radius <= 40:
            return original, radius
        proposal, _kind = estimate
        center = 0.1 * original + 0.9 * proposal
        if not state.region.covers(Point(center)):
            center = np.asarray(nearest_points(state.region, Point(center))[0].coords[0])
        enclosing_radius = float(np.linalg.norm(hull_vertices(state.region) - center, axis=1).max())
        self.stats['prior_centers'] += 1
        return center, enclosing_radius

    def localize(self, state):
        estimate = self.estimate(state)
        original_center, original_radius = super().center_radius(state)
        if (self.prior_mode == 'prior_trial' and estimate is not None and len(self.clear_positions) >= 4
                and original_radius > 60 and state.channel not in self.tried_channels):
            self.tried_channels.add(state.channel)
            self.stats['prior_trials'] += 1
            success = self.clear(state.channel, estimate[0])
            self.stats['prior_hits'] += int(success)
            self.reuse_stop()
            if success:
                return
        return super().localize(state)
