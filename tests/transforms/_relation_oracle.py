"""Evaluates SpatialRelation chains on a point, to check recovered relations against the original chain.

This deliberately shares nothing with the transform machinery (no canonicalizer, no transforms): It applies each
relation by its documented meaning, one after the other, to a dict from axis name to value.

- Affine: `x_target = linear @ x_source + translation`; Translation: adds its values.
- PermutationTo / ProjectionTo / AxisRearrangementTo: Values follow their axis name. Dropped axes are gone, inserted
  axes start at 0.
- AxisRenaming: Values stay where they are, only the names change.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clearscale import Affine, AxisRenaming, SameSpace, SpatialRelation, Translation
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale.types import AxisKey

Point = Mapping[Any, float]  # Mapping is invariant in its keys: accept dicts keyed by str as well as by AxisKey


def evaluate_relation(relation: SpatialRelation, point: Point) -> Dict[AxisKey, float]:
    """The image of `point` (axis name -> value) under `relation`, with axes in the relation's target order."""
    if isinstance(relation, SpatialRelationSequence):
        for step in relation.relations:
            point = evaluate_relation(step, point)
        return dict(point)
    axes = tuple(point)
    targets = relation.target_axes(axes)  # also validates that the relation fits the axes
    if isinstance(relation, SameSpace):
        return dict(point)
    if isinstance(relation, Affine):
        return {
            target: sum(relation.linear[target][source] * point[source] for source in axes)
            + relation.translation[target]
            for target in targets
        }
    if isinstance(relation, Translation):
        return {target: point[target] + relation[target] for target in targets}
    if isinstance(relation, AxisRenaming):
        return {target: point[source] for source, target in zip(axes, targets)}
    # PermutationTo, ProjectionTo, AxisRearrangementTo
    return {target: point.get(target, 0.0) for target in targets}


def linear_part_of(
    relation: SpatialRelation, source_axes: Sequence[AxisKey]
) -> Tuple[List[List[float]], Tuple[AxisKey, ...]]:
    """The matrix (rows: target axes, columns: source axes) of the chain's linear part, read off by evaluating it at the
    origin and at the unit vectors, and the chain's target axes."""
    source_axes = tuple(source_axes)
    origin = evaluate_relation(relation, {a: 0.0 for a in source_axes})
    targets = tuple(origin)
    columns = []
    for axis in source_axes:
        image = evaluate_relation(relation, {a: 1.0 if a == axis else 0.0 for a in source_axes})
        columns.append([image[t] - origin[t] for t in targets])
    return [[column[i] for column in columns] for i in range(len(targets))], targets


def is_invertible(matrix: Sequence[Sequence[float]], relative_tolerance: float = 1e-9) -> bool:
    """Whether a square matrix has full rank (Gaussian elimination with partial pivoting, tolerance relative to the
    largest entry)."""
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        return False
    work = [list(map(float, row)) for row in matrix]
    scale: Optional[float] = max((abs(v) for row in work for v in row), default=0.0)
    if not scale or math.isnan(scale):
        return False
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(work[r][column]))
        if abs(work[pivot][column]) <= relative_tolerance * scale:
            return False
        work[column], work[pivot] = work[pivot], work[column]
        for row in range(column + 1, size):
            factor = work[row][column] / work[column][column]
            for k in range(column, size):
                work[row][k] -= factor * work[column][k]
    return True
