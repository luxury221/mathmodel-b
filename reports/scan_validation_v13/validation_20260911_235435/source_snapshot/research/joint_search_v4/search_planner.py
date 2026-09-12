from __future__ import annotations

import numpy as np
from geometry import dual_ring_network, ring, seven_network


class WitnessPlanner:
    def __init__(self, problem):
        self.problem = problem
        axis = np.arange(-1750, 1751, 250)
        grid = np.array([(horizontal, vertical) for horizontal in axis for vertical in axis
                         if horizontal**2 + vertical**2 <= 1800**2])
        positions = np.vstack((grid, ring(1799.99, 72), ring(1000.01, 36), [0, 0]))
        if problem == 'q4':
            self.positions = np.repeat(positions, 36, axis=0)
            self.headings = np.tile(ring(1, 36), (len(positions), 1))
            self.candidates = np.vstack((dual_ring_network(), ring(1870, 12), ring(1350, 12), ring(500, 6), ring(950, 8)))
        else:
            self.positions = positions
            self.headings = None
            self.candidates = np.vstack((seven_network(), ring(1400, 12), ring(1750, 12)))
        self.alive = np.ones(len(self.positions), dtype=bool)
        self.matrix = np.array([self.visible(candidate) for candidate in self.candidates])
        self.observation_count = 0

    def visible(self, receiver):
        vectors = np.asarray(receiver) - self.positions
        visible = np.sum(vectors**2, axis=1) <= 999.9**2
        if self.headings is not None:
            visible &= np.sum(vectors * self.headings, axis=1) >= 0
        return visible

    def update(self, observations):
        for receiver in observations[self.observation_count:]:
            self.alive &= ~self.visible(receiver)
        self.observation_count = len(observations)

    def plan(self, start, target_positions):
        alive = self.alive.copy()
        selected = []
        anchors = np.vstack((np.asarray(start), target_positions)) if len(target_positions) else np.asarray(start)[None, :]
        for _iteration in range(24):
            if not alive.any():
                break
            gains = np.sum(self.matrix[:, alive], axis=1)
            distances = np.linalg.norm(self.candidates[:, None] - anchors[None, :], axis=2).min(axis=1)
            scores = gains / (600 + distances)
            index = int(np.argmax(scores))
            if gains[index] == 0:
                break
            selected.append(self.candidates[index])
            anchors = np.vstack((anchors, self.candidates[index]))
            alive &= ~self.matrix[index]
        return selected
