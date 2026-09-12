from __future__ import annotations

from numbers import Integral


def normalize_points(points, point_count):
    values = list(points)
    if any(not isinstance(value, Integral) or isinstance(value, bool)
           or value < 0 or value >= point_count for value in values):
        raise ValueError('Every point index must be an integer inside the pool')
    return tuple(sorted(set(int(value) for value in values)))


def reduce_cover(groups, point_count, forced_true=(0,)):
    if not isinstance(point_count, Integral) or isinstance(point_count, bool) or point_count <= 0:
        raise ValueError('Point count must be a positive integer')
    fixed = normalize_points(forced_true, point_count)
    fixed_set = set(fixed)
    normalized = [normalize_points(group, point_count) for group in groups]
    ordered = sorted(range(len(normalized)), key=lambda index: (len(normalized[index]), normalized[index], index))
    implications = [None] * len(normalized)
    retained_masks = {}
    buckets = {}
    empty_index = None
    for original_index in ordered:
        group = normalized[original_index]
        fixed_members = fixed_set.intersection(group)
        if fixed_members:
            implications[original_index] = {'kind': 'fixed_true', 'point': min(fixed_members)}
            continue
        mask = sum(1 << member for member in group)
        witness = empty_index
        if witness is None:
            for member in group:
                for previous_index in buckets.get(member, ()):
                    previous_mask = retained_masks[previous_index]
                    if previous_mask & mask == previous_mask:
                        witness = previous_index
                        break
                if witness is not None:
                    break
        if witness is not None:
            implications[original_index] = {'kind': 'retained_subset', 'index': witness}
            continue
        retained_masks[original_index] = mask
        implications[original_index] = {'kind': 'retained_subset', 'index': original_index}
        if group:
            buckets.setdefault(group[0], []).append(original_index)
        else:
            empty_index = original_index
    retained = sorted(retained_masks)
    return {
        'point_count': int(point_count),
        'forced_true': list(fixed),
        'original_count': len(normalized),
        'retained_count': len(retained),
        'fixed_satisfied_count': sum(item['kind'] == 'fixed_true' for item in implications),
        'subsumed_count': sum(item['kind'] == 'retained_subset' for item in implications) - len(retained),
        'kept_indices': retained,
        'implications': implications,
    }
