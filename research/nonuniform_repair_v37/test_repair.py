from __future__ import annotations

import numpy as np
import repair


def test_registered_starts_have_expected_counts_and_fixed_origin():
    layouts = list(repair.starting_layouts())
    assert [len(points) for _name, points in layouts] == [20, 20, 19]
    for _name, points in layouts:
        assert np.array_equal(points[0], np.zeros(2))
        assert np.max(np.linalg.norm(points, axis=1)) < 2000


def test_nearly_complete_twenty_point_start_is_not_misclassified():
    _name, points = next(repair.starting_layouts())
    coverage = repair.synthesis.certify(points)
    assert not coverage.empty
    assert coverage.area > 10000
    examples = repair.synthesis.counterexamples(coverage.region, points, limit=64)
    assert examples
    assert all(violation > 0 for violation, _position, _heading in examples)


def test_public_witnesses_exclude_the_assumed_optical_origin():
    positions, headings = repair.synthesis.initial_witnesses()
    assert len(positions) == len(headings)
    assert np.min(np.linalg.norm(positions, axis=1)) > 19.99
    assert np.allclose(np.linalg.norm(headings, axis=1), 1)
