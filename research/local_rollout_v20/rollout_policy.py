from __future__ import annotations

import math

import numpy as np
from local_model import conditional_samples, forecast
from plan_geometry import hull_vertices, small_clear_plan
from scan_policy import ScanEconomyPolicy


MODES = ('rollout', 'cost_clear', 'clear_first')


class LocalRolloutPolicy(ScanEconomyPolicy):
    def __init__(self, *args, mode='rollout'):
        if mode not in MODES:
            raise ValueError('Unknown local-rollout mode')
        super().__init__(*args, mode='station_only')
        self.rollout_mode = mode
        self.forecast_events = []
        self.stats.update({'rollout_targets': 0, 'rollout_probe_changes': 0,
                           'rollout_direct_clears': 0, 'rollout_prediction_failures': 0})

    def candidates(self, state, original):
        center, radius = self.center_radius(state)
        anchor = state.positives[-1][0]
        radial = center - anchor
        radial /= max(float(np.linalg.norm(radial)), 1e-9)
        tangent = np.array([-radial[1], radial[0]])
        vertices = hull_vertices(state.region)
        proposed = [self.position.copy()]
        proposed.extend(self.position + fraction * (center - self.position) for fraction in (0.25, 0.5, 0.75))
        proposed.append(center.copy())
        for fraction in (0.25, 0.65):
            offset = min(300.0, max(25.0, fraction * radius))
            proposed.extend(center + sign * offset * tangent for sign in (-1, 1))
        allowed = []
        for point in proposed:
            if any(np.linalg.norm(point - previous) < 25 for previous in state.measured_positions):
                continue
            if np.linalg.norm(vertices - point, axis=1).max() > 999.9:
                continue
            if np.linalg.norm(point - original) < 1e-5 or any(np.linalg.norm(point - prior) < 1e-5 for prior in allowed):
                continue
            allowed.append(point)

        def preliminary_cost(point):
            previous = center - anchor
            following = center - point
            sine = abs(previous[0] * following[1] - previous[1] * following[0])
            sine /= max(1e-8, np.linalg.norm(previous) * np.linalg.norm(following))
            residual = min(radius, math.tan(math.radians(1.005)) * np.linalg.norm(following) / max(sine, 0.08))
            return (np.linalg.norm(point - self.position) + np.linalg.norm(following)) / 5 + 2 * residual

        return [np.asarray(original).copy(), *sorted(allowed, key=preliminary_cost)[:7]]

    def choose_forecast(self, state, original, clear_plan):
        samples = conditional_samples(self.hypotheses(state))
        if not samples:
            return None
        options = []
        for point in self.candidates(state, original):
            prediction = forecast(self, state, samples, point=point)
            options.append(('probe', point, prediction))
        if self.rollout_mode == 'cost_clear' and clear_plan is not None:
            prediction = forecast(self, state, samples, clear_plan=clear_plan)
            options.append(('clear', clear_plan.centers[0], prediction))
        chosen = min(options, key=lambda option: option[2]['score'])
        if chosen[2]['score'] > options[0][2]['score'] - 2:
            chosen = options[0]
        self.stats['rollout_prediction_failures'] += sum(len(option[2]['failures']) for option in options)
        event = {'channel': state.channel, 'origin': self.position.tolist(), 'clock_before': self.virtual_seconds,
                 'baseline_point': np.asarray(original).tolist(), 'selected_kind': chosen[0],
                 'selected_point': chosen[1].tolist(), 'baseline_prediction': options[0][2],
                 'selected_prediction': chosen[2],
                 'options': [{'kind': kind, 'point': point.tolist(), **prediction} for kind, point, prediction in options]}
        return chosen, event

    def localize(self, state):
        plan = small_clear_plan(state.region, self.position, 4) if state.status == 'DETECTED' else None
        if state.status == 'NEAR' or (plan is not None and (plan.certificate_kind == 'mec' or len(plan.centers) <= 2)):
            return super().localize(state)
        if self.rollout_mode == 'clear_first':
            if plan is None:
                return super().localize(state)
            self.active_target = state.channel
            self.execute_clear_plan(state, plan)
            self.stats['rollout_direct_clears'] += 1
            self.active_target = None
            return None
        original = super().choose_probe(state)
        if original is None:
            return super().localize(state)
        decision = self.choose_forecast(state, original, plan)
        if decision is None:
            return super().localize(state)
        (kind, point, _prediction), event = decision
        self.stats['rollout_targets'] += 1
        self.active_target = state.channel
        if kind == 'clear':
            self.execute_clear_plan(state, plan)
            self.stats['rollout_direct_clears'] += 1
            self.active_target = None
        else:
            self.stats['rollout_probe_changes'] += int(not np.array_equal(point, original))
            response = self.measure(state.channel, point, active=True)
            self.probe_history[state.channel].append(np.asarray(point).copy())
            self.stats['close_probes'] += 1
            self.stats['close_no_signal'] += int(response['result'] == 'no_signal')
            self.reuse_stop()
            super().localize(state)
        event['clock_after'] = self.virtual_seconds
        event['actual_local_seconds'] = self.virtual_seconds - event['clock_before']
        event['actual_minus_predicted_mean'] = event['actual_local_seconds'] - event['selected_prediction']['mean']
        self.forecast_events.append(event)
