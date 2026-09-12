from __future__ import annotations

import numpy as np
from action_policy import ActionSchedulerPolicy
from directed_routes import directed_order, route_cost, transition_costs
from geometry import open_route
from plan_geometry import disk_polygon
from scan_policy import ScanEconomyPolicy
from task_forecasts import forecast_task


MODES = ('identity', 'center_exit', 'rollout_exit')


class ExitDistributionPolicy(ScanEconomyPolicy):
    plan_action = ActionSchedulerPolicy.plan_action

    def __init__(self, *args, mode='rollout_exit'):
        if mode not in MODES:
            raise ValueError('Unknown Q3 complete-task scheduling mode')
        super().__init__(*args, mode='station_only')
        if self.problem != 'q3' or self.options['target_radius'] != 2000:
            raise ValueError('Complete-task study requires the frozen Q3 admission policy')
        self.exit_mode = mode
        self.task_events = []
        self.stats.update({'task_forecasts': 0, 'task_forecast_failures': 0, 'task_route_fallbacks': 0,
                           'task_route_decisions': 0, 'task_exit_order_changes': 0})

    def first_action(self, state):
        return ActionSchedulerPolicy.next_local_action(self, state, 0)

    def select_task(self, stops):
        original = np.asarray([stop[2] for stop in stops])
        route = open_route(original, self.position)
        baseline_selected = int(np.argmin(np.linalg.norm(original - route[0], axis=1)))
        jobs, failed = [], False
        for kind, identifier, destination in stops:
            job = {'kind': kind, 'identifier': identifier, 'center_destination': np.asarray(destination).tolist()}
            if kind == 'station':
                job.update({'entry': np.asarray(destination).tolist(), 'exits': [np.asarray(destination).tolist()],
                            'weights': [1.0], 'first_kind': 'station'})
            else:
                state = self.states[identifier]
                action = self.first_action(state)
                job.update({'entry': action.position.tolist(), 'first_kind': action.kind, 'revision': state.revision})
                if self.exit_mode == 'center_exit':
                    job.update({'exits': [np.asarray(destination).tolist()], 'weights': [1.0]})
                else:
                    forecast = forecast_task(self, state, action)
                    self.stats['task_forecasts'] += 1
                    job['forecast'] = forecast
                    if not forecast['ok']:
                        self.stats['task_forecast_failures'] += 1
                        failed = True
                    else:
                        job.update({'exits': forecast['exits'], 'weights': forecast['weights']})
            jobs.append(job)
            if failed:
                break
        event = {'mode': self.exit_mode, 'start_position': self.position.tolist(), 'start_seconds': self.virtual_seconds,
                 'jobs': jobs, 'baseline_selected': baseline_selected, 'fallback': failed}
        if failed:
            self.stats['task_route_fallbacks'] += 1
            selected = baseline_selected
        else:
            initial, matrix = transition_costs([job['entry'] for job in jobs], [job['exits'] for job in jobs],
                                               [job['weights'] for job in jobs], self.position)
            order = directed_order(initial, matrix)
            selected = order[0]
            event.update({'planned_order': order, 'planned_route_cost_meters': route_cost(order, initial, matrix)})
        self.stats['task_route_decisions'] += 1
        self.stats['task_exit_order_changes'] += int(selected != baseline_selected)
        event['selected_index'] = selected
        return selected, event

    def run_tour(self):
        if self.exit_mode == 'identity':
            return super().run_tour()
        remaining = list(range(len(self.route)))
        self.scan_unknown(self.position, force=True)
        remaining.remove(0)
        self.stats['outer_first'] = False
        for _iteration in range(120):
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            if not remaining and not self.discovery_done:
                self.finish_discovery('full_route')
            if self.options.get('adaptive') and not self.discovery_done:
                irrelevant = [index for index in remaining if self.coverage.region.intersection(
                    disk_polygon(self.route[index], 999.9)).is_empty]
                remaining = [index for index in remaining if index not in irrelevant]
                self.stats['coverage_skips'] += len(irrelevant)
                if not remaining:
                    if not self.coverage.empty:
                        raise ValueError('Pruned search stations without completing coverage')
                    self.finish_discovery('coverage')
            stops = []
            for state in unresolved:
                center, radius = (state.near_position, 0) if state.status == 'NEAR' else self.center_radius(state)
                if radius <= self.options.get('target_radius', 130) or not remaining or self.discovery_done:
                    stops.append(('target', state.channel, center))
            if not self.discovery_done:
                stops.extend(('station', index, self.route[index]) for index in remaining)
            if not stops:
                return self.result()
            selected, event = self.select_task(stops)
            kind, identifier, destination = stops[selected]
            event.update({'selected_kind': kind, 'selected_identifier': identifier})
            if kind == 'target':
                action = self.first_action(self.states[identifier])
                event.update({'first_position': action.position.tolist(), 'first_kind': action.kind,
                              'active_before': self.stats['active_measures']})
                self.stats['joint_targets'] += 1
                self.localize(self.states[identifier])
                event['active_after'] = self.stats['active_measures']
            else:
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.reuse_directions()
                remaining.remove(identifier)
            event.update({'end_seconds': self.virtual_seconds, 'actual_exit': self.position.tolist()})
            self.task_events.append(event)
        raise ValueError('Tour iteration limit exceeded')
