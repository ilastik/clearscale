import pytest

from clearscale import Affine, AxisRenaming, Factor, PermutationTo, ProjectionTo
from clearscale._spatial_relations import AxisRearrangementTo, SpatialRelationSequence
from clearscale._transforms import IdentityTransform, TransformSequence, relation_to_transform_canonic
from clearscale._transforms._to_from_spatial_relation import _flatten_leaves, relation_to_transform

from ._random_relations import random_relation_chain


def scaled(**factors: float) -> Affine:
    return Affine.from_factor(Factor(factors))


def transform_dependencies_by_index(transform, source_ndim):
    """Fold the leaves' index-based dependencies into the dependencies of the entire transform."""
    leaves = _flatten_leaves(TransformSequence((transform.unbound(),)).canonicalized())
    dependencies = tuple((i,) for i in range(source_ndim))
    for leaf in leaves:
        step = leaf._axis_dependencies(len(dependencies))
        assert step is not None
        dependencies = tuple(tuple(sorted({d for i in deps for d in dependencies[i]})) for deps in step)
    return dependencies


def relation_dependencies_by_index(relation, source_axes):
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
    assert transform_dependencies_by_index(transform, len(source)) == relation_dependencies_by_index(relation, source)


def chain_mixes_axes(relation, source_axes) -> bool:
    relations = relation.relations if isinstance(relation, SpatialRelationSequence) else (relation,)
    axes = tuple(source_axes)
    for step in relations:
        if any(len(deps) > 1 for deps in step._axis_dependencies(axes)):
            return True
        axes = step.target_axes(axes)
    return False


@pytest.mark.parametrize("seed", range(300))
def test_relation_dependencies_agree_with_those_of_its_transform(seed):
    source, relation = random_relation_chain(seed)
    transform = relation_to_transform_canonic(relation, source)
    from_transform = transform_dependencies_by_index(transform, len(source))
    from_relation = relation_dependencies_by_index(relation, source)
    if not chain_mixes_axes(relation, source):
        assert from_transform == from_relation
        return
    # Chains that mix axes combine their steps' dependencies without noticing numerical cancellation between steps:
    # The relation-level answer is an upper bound for the exact one that the combined transform has.
    assert all(set(exact) <= set(bound) for exact, bound in zip(from_transform, from_relation))


def test_axis_renaming_converts_to_positional_identity():
    assert isinstance(relation_to_transform(AxisRenaming(z="Z"), "zyx"), IdentityTransform)
    assert relation_to_transform_canonic(AxisRenaming(z="Z"), "zyx") == IdentityTransform()


def test_axis_renaming_in_a_sequence_leaves_later_relations_to_use_the_new_names():
    relation = SpatialRelationSequence((AxisRenaming(y="Y"), scaled(z=1, Y=2, x=2)))
    transform = relation_to_transform_canonic(relation, "zyx")
    assert transform.scale == (1.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        relation_to_transform(SpatialRelationSequence((AxisRenaming(y="Y"), scaled(z=1, y=2, x=2))), "zyx")


def test_axis_renaming_is_incompatible_with_wrong_source_axes():
    with pytest.raises(ValueError):
        relation_to_transform(AxisRenaming(c="C"), "zyx")


def test_dependencies_of_mixing_chains_are_an_upper_bound_when_the_steps_cancel_out():
    # Two opposite shears. Each one mixes y and x, but together they are the identity.
    shear = Affine.from_array(((1, 1, 0), (0, 1, 0)), "yx")
    opposite = Affine.from_array(((1, -1, 0), (0, 1, 0)), "yx")
    assert shear._axis_dependencies("yx") == opposite._axis_dependencies("yx") == (("y", "x"), ("x",))
    chain = SpatialRelationSequence((shear, opposite))
    transform = relation_to_transform_canonic(chain, "yx")
    # The exact answer, from the composed matrix: Every axis depends on itself only
    assert transform_dependencies_by_index(transform, 2) == ((0,), (1,))
    # SpatialRelationSequence combines the dependencies step by step, and does not notice the cancellation
    assert chain._axis_dependencies("yx") == (("y", "x"), ("x",))
