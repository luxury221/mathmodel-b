from __future__ import annotations

import types

import numpy as np
from latency_route import DiscoveryRouter
from population import PopulationModel
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy


MODES = ('identity', 'distance', 'area', 'mixture')


class DiscoveryOrderMixin:
    reference_mode = None

    def __init__(self, port, problem, stations, variant, network, mode='mixture'):
        if mode not in MODES:
            raise ValueError('Unknown discovery order mode')
        super().__init__(port, problem, stations, variant, network, mode=self.reference_mode)
        self.order_mode = mode
        self.population = None if mode == 'identity' else PopulationModel(problem, mode)
        self.order_router = DiscoveryRouter(self, self.population, mode)
        self.scan_calibration = []
        self.stats.update({'latency_dispatch_calls': 0, 'latency_route_calls': 0,
                           'latency_first_changes': 0, 'latency_colocated_fallbacks': 0})

    def call_routed(self, method):
        original = method.__func__
        namespace = {**original.__globals__, 'open_route': self.order_router}
        cloned = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        cloned.__kwdefaults__ = original.__kwdefaults__
        return types.MethodType(cloned, self)()

    def run(self):
        return self.call_routed(super().run)

    def run_tour(self):
        return self.call_routed(super().run_tour)

    def measure(self, channel, position, active=False, opportunistic=False):
        was_unknown = self.states[channel].status == 'UNKNOWN'
        clock = self.virtual_seconds
        response = super().measure(channel, position, active, opportunistic)
        if self.population is not None and was_unknown and self.virtual_seconds > clock:
            self.population.observe(channel, position, response['result'] != 'no_signal')
        return response

    def clear(self, channel, position):
        was_unknown = self.states[channel].status == 'UNKNOWN'
        result = super().clear(channel, position)
        if self.population is not None and was_unknown:
            self.population.observe(channel, position, bool(result), optical=True)
        return result

    def scan_unknown(self, position, force=False):
        prediction = None
        if self.population is not None and force and self.unknown_channels() and not self.discovery_done:
            forecast = self.population.forecast(self.unknown_channels(), self.known_count())
            prediction = {'point': np.asarray(position).tolist(), 'clock_before': self.virtual_seconds,
                          'known_before': self.known_count(), 'unknown_before': len(self.unknown_channels()),
                          'expected_discoveries': float(forecast['masses'][self.population.visible(position)].sum()),
                          'family_probabilities': forecast['family_probabilities'],
                          'radio_observations_before': self.population.radio_observations}
        result = super().scan_unknown(position, force)
        if prediction is not None and self.population.radio_observations > prediction['radio_observations_before']:
            prediction['actual_discoveries'] = self.known_count() - prediction['known_before']
            prediction['clock_after'] = self.virtual_seconds
            prediction['radio_observations'] = self.population.radio_observations - prediction.pop('radio_observations_before')
            self.scan_calibration.append(prediction)
        return result


class DiscoveryOrderQ3(DiscoveryOrderMixin, ScanEconomyPolicy):
    reference_mode = 'station_only'


class DiscoveryOrderQ4(DiscoveryOrderMixin, ShapedProbePolicy):
    reference_mode = 'shaped_cost'
