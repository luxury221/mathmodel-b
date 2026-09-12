from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from plan_geometry import hull_vertices


@dataclass
class AsymmetricProbe:
    anchor: np.ndarray
    candidates: np.ndarray
    vertices: np.ndarray
    direction: np.ndarray
    side: float
    maximum_squared_range_change: float
    minimum_forward_gap: float
    minimum_cone_slack: float


def certify_probe(region, anchor, first, mirror, bearing):
    anchor = np.asarray(anchor, dtype=float)
    candidates = np.asarray([first, mirror], dtype=float)
    vertices = hull_vertices(region)
    angle = math.radians(bearing)
    direction = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([-direction[1], direction[0]])
    relative = vertices - anchor
    displacements = candidates - anchor
    longitudinal = displacements @ direction
    side = float(np.sign(displacements[0] @ tangent))
    if side == 0 or np.min(longitudinal) <= 1e-6:
        return None
    gap = float(np.min(relative @ direction) - np.max(longitudinal))
    crosses = relative[:, 0, None] * displacements[None, :, 1] - relative[:, 1, None] * displacements[None, :, 0]
    slack = float(min(np.min(side * crosses[:, 0]), np.min(-side * crosses[:, 1])))
    old_squared = np.sum(relative**2, axis=1)
    new_squared = np.sum((candidates[:, None] - vertices[None, :])**2, axis=2)
    change = float(np.max(new_squared - old_squared[None, :]))
    if gap <= 1e-6 or slack <= 1e-6 or change >= -1e-6:
        return None
    return AsymmetricProbe(anchor.copy(), candidates.copy(), vertices.copy(), direction, side, change, gap, slack)


def solve_point(region, anchor, template, mirror, bearing, witness_positions, witness_headings):
    anchor = np.asarray(anchor, dtype=float)
    vertices = hull_vertices(region)
    relative = (vertices - anchor) / 1000
    previous_ranges = np.linalg.norm(relative, axis=1)
    angle = math.radians(bearing)
    direction = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([-direction[1], direction[0]])
    side = float(np.sign((np.asarray(template) - anchor) @ tangent))
    witness_positions = np.asarray(witness_positions, dtype=float).reshape(-1, 2)
    witness_headings = np.asarray(witness_headings, dtype=float).reshape(-1, 2)
    if len(witness_positions):
        distances = np.linalg.norm(vertices[:, None] - witness_positions[None, :], axis=2)
        if np.any(distances > 1000 * previous_ranges[:, None] + 999.8):
            return None, 'disjoint_range_balls'
    variable = cp.Variable(2)
    cone = relative[:, 0] * variable[1] - relative[:, 1] * variable[0]
    constraints = [cp.norm(variable[None, :] - relative, axis=1) <= previous_ranges - 1e-6,
                   side * cone >= 1e-8,
                   variable @ direction >= 1e-4,
                   variable @ direction <= float(np.min(relative @ direction)) - 1e-6]
    if len(witness_positions):
        witness_relative = (witness_positions - anchor) / 1000
        unique_positions = np.unique(witness_relative, axis=0)
        constraints.extend([cp.norm(variable[None, :] - unique_positions, axis=1) <= 0.9998,
                            cp.sum(cp.multiply(variable[None, :] - witness_relative, witness_headings), axis=1) >= 5e-5])
    objective = cp.Minimize(cp.sum_squares(variable - (np.asarray(template) - anchor) / 1000))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver='CLARABEL', max_iter=150, tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    except cp.error.SolverError:
        return None, 'solver_error'
    if problem.status != cp.OPTIMAL or variable.value is None:
        return None, str(problem.status)
    point = anchor + 1000 * np.asarray(variable.value)
    if np.linalg.norm(point - template) < 1e-4:
        point = np.asarray(template).copy()
    certificate = certify_probe(region, anchor, point, mirror, bearing)
    if certificate is None:
        return None, 'physical_probe_recheck_failed'
    if len(witness_positions):
        vectors = point - witness_positions
        if np.any(np.linalg.norm(vectors, axis=1) > 999.9) or np.any(np.sum(vectors * witness_headings, axis=1) < 0):
            return None, 'physical_witness_recheck_failed'
    return certificate, 'optimal'


def missing_heading(point, receivers):
    point = np.asarray(point)
    vectors = np.asarray(receivers) - point
    close = vectors[np.linalg.norm(vectors, axis=1) <= 999.9]
    if not len(close):
        return np.array([1.0, 0.0])
    angles = np.sort(np.mod(np.arctan2(close[:, 1], close[:, 0]), 2 * math.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
    largest = int(np.argmax(gaps))
    if gaps[largest] <= math.pi + 1e-9:
        return None
    heading = angles[largest] + gaps[largest] / 2
    direction = np.array([math.cos(heading), math.sin(heading)])
    return direction if np.all(close @ direction < 0) else None
