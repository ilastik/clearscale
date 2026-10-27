from typing import Tuple

import pytest

from clearscale import Affine, AxisRenaming, Factor, PermutationTo, ProjectionTo
from clearscale._spatial_relations import AxisRearrangementTo, SpatialRelationSequence, SpatialRelation
from clearscale._transforms import TransformSequence, relation_to_transform_canonic
from clearscale._transforms._to_from_spatial_relation import _flatten_atomic

from ._random_relations import chain_mixes_axes, random_relation_chain


def scaled(**factors: float) -> Affine:
    return Affine.from_factor(Factor(factors))


def transform_dependencies(transform, source_ndim) -> Tuple[Tuple[int, ...], ...]:
    transforms = _flatten_atomic(TransformSequence((transform.unbound(),)).canonicalized())
    # This below should basically be the implementation of TransformSequence._axis_dependencies,
    # but in prod we don't need sequence dependencies because we only ever ask dependencies of atomic transforms
    dependencies = tuple((i,) for i in range(source_ndim))
    for t in transforms:
        step = t._axis_dependencies(len(dependencies))
        assert step is not None
        dependencies = tuple(tuple(sorted({d for i in deps for d in dependencies[i]})) for deps in step)
    return dependencies


def relation_dependencies_by_index(relation, source_axes) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(source_axes.index(a) for a in deps) for deps in relation._axis_dependencies(source_axes))


@pytest.mark.parametrize(
    "relation, source",
    [
        (AxisRenaming(z="Z"), "zyx"),
        (AxisRenaming(y="x", x="y"), "zyx"),
        (scaled(z=1, y=2, x=2), "zyx"),
        (ProjectionTo("cyx"), "zyx"),
        (PermutationTo("xyz"), "zyx"),
        (AxisRearrangementTo("xcy"), "zyx"),
        (Affine.from_array(((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0)), "zyx"), "zyx"),
        (SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx"))), "zyx"),
    ],
)
def test_handpicked_relations(relation, source):
    transform = relation_to_transform_canonic(relation, source)
    assert transform_dependencies(transform, len(source)) == relation_dependencies_by_index(relation, source)


@pytest.mark.parametrize("seed", range(300))
def test_relation_dependencies_agree_with_those_of_its_transform(seed):
    source, relation = random_relation_chain(seed)
    transform = relation_to_transform_canonic(relation, source)
    from_transform = transform_dependencies(transform, len(source))
    from_relation = relation_dependencies_by_index(relation, source)
    if not chain_mixes_axes(relation, source):
        assert from_transform == from_relation
        return
    # Chains that mix axes combine their steps' dependencies without noticing numerical cancellation between steps:
    # The relation-level answer is an upper bound for the exact one that the combined transform has.
    assert all(set(exact) <= set(bound) for exact, bound in zip(from_transform, from_relation))
