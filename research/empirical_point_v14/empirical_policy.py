from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
from plan_geometry import hull_vertices
from scan_policy import ScanEconomyPolicy


MODES = ('empirical_route', 'empirical_trial', 'empirical_point')


class EmpiricalPointPolicy(ScanEconomyPolicy):
    def __init__(self, *args, mode='empirical_point'):
        if mode not in MODES:
            raise ValueError('Unknown empirical point mode')
        self.empirical_mode = mode
        self.empirical_cache = {}
        self.tried_channels = set()
        self.empirical_events = []
        self.type_outcomes = {kind: {'hits': 0, 'failures': 0} for kind in ('cluster', 'radial')}
        super().__init__(*args, mode='station_only')
        self.stats.update({'empirical_centers': 0, 'empirical_trials': 0, 'empirical_hits': 0, 'empirical_failed_trials': 0})

    def disabled(self, kind):
        outcomes = self.type_outcomes[kind]
        return outcomes['hits'] == 0 and outcomes['failures'] >= 3

    def estimate(self, state):
        if state.status != 'DETECTED' or len(self.clear_positions) < 3:
            return None
        disabled = tuple(self.disabled(kind) for kind in ('cluster', 'radial'))
        key = (state.channel, state.revision, len(self.clear_positions), disabled, tuple(self.position))
        if key in self.empirical_cache:
            return self.empirical_cache[key]
        points = np.asarray(self.clear_positions)
        neighborhoods = [points[np.linalg.norm(points - point, axis=1) <= 60] for point in points]
        neighbors = max(neighborhoods, key=len)
        proposals = []
        if len(neighbors) >= 3 and not self.disabled('cluster'):
            center = np.median(neighbors, axis=0)
            width = max(40.0, float(np.median(np.linalg.norm(neighbors - center, axis=1))) * 2 + 20)
            proposals.append((center, 'cluster', center, width, len(neighbors)))
        radii = np.linalg.norm(points, axis=1)
        median = min(1800.0, float(np.median(radii)))
        deviation = float(np.median(np.abs(radii - np.median(radii))))
        if not proposals and median > 1500 and deviation < 60 and not self.disabled('radial'):
            width = max(40.0, 2 * deviation + 20)
            original, _radius = super().center_radius(state)
            if np.linalg.norm(original) > 1e-8:
                proposals.append((original * median / np.linalg.norm(original), 'radial', median, width, len(points)))
            for anchor, bearing in state.positives:
                angle = math.radians(bearing)
                direction = np.array([math.cos(angle), math.sin(angle)])
                projection = float(anchor @ direction)
                discriminant = projection**2 + median**2 - float(anchor @ anchor)
                if discriminant < 0:
                    continue
                for distance in (-projection - math.sqrt(discriminant), -projection + math.sqrt(discriminant)):
                    if 0 <= distance <= 1500:
                        proposals.append((anchor + distance * direction, 'radial', median, width, len(points)))
        choices = []
        for proposed, kind, center, width, support in proposals:
            original = Point(proposed)
            projected = nearest_points(state.region, original)[0]
            if original.distance(projected) > width:
                continue
            point = np.asarray(projected.coords[0])
            angular_loss = 0.0
            for anchor, bearing in state.positives:
                vector = point - anchor
                angle = math.degrees(math.atan2(vector[1], vector[0])) % 360
                error = (angle - bearing + 180) % 360 - 180
                angular_loss += (error / 1.005)**2
            prior_distance = float(np.linalg.norm(point - center)) if kind == 'cluster' else abs(float(np.linalg.norm(point)) - center)
            score = angular_loss + (prior_distance / max(20.0, width / 2))**2 + 0.001 * float(np.linalg.norm(point - self.position))
            choices.append((score, point, kind, support))
        if not choices:
            self.empirical_cache[key] = None
            return None
        score, point, kind, support = min(choices, key=lambda choice: choice[0])
        result = point.copy(), kind
        self.empirical_cache[key] = result
        self.empirical_events.append({'event': 'proposal', 'channel': state.channel, 'revision': state.revision,
                                      'kind': kind, 'supporting_successes': support, 'point': point.tolist(),
                                      'score': score, 'actual_belief_area': state.region.area})
        return result

    def center_radius(self, state):
        original, radius = super().center_radius(state)
        if self.empirical_mode == 'empirical_trial' or radius <= 40:
            return original, radius
        estimate = self.estimate(state)
        if estimate is None:
            return original, radius
        point, _kind = estimate
        enclosing = float(np.linalg.norm(hull_vertices(state.region) - point, axis=1).max())
        self.stats['empirical_centers'] += 1
        return point.copy(), enclosing

    def localize(self, state):
        _center, radius = super().center_radius(state)
        estimate = self.estimate(state)
        if (self.empirical_mode != 'empirical_route' and estimate is not None and radius > 60
                and state.channel not in self.tried_channels):
            point, kind = estimate
            self.tried_channels.add(state.channel)
            self.stats['empirical_trials'] += 1
            success = self.clear(state.channel, point)
            self.type_outcomes[kind]['hits' if success else 'failures'] += 1
            self.stats['empirical_hits'] += int(success)
            self.stats['empirical_failed_trials'] += int(not success)
            self.empirical_events.append({'event': 'trial', 'channel': state.channel, 'kind': kind,
                                          'point': point.tolist(), 'success': success, 'actual_clock': self.virtual_seconds})
            self.reuse_stop()
            if success:
                return
        return super().localize(state)
