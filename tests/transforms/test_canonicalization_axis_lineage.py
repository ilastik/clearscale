"""Canonicalization must not lose which output axis continues which input axis, where a matrix cannot tell."""

import pytest

from clearscale import Affine, PermutationTo, ProjectionTo, Translation
from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale._transforms import (
    AffineTransform,
    DisplacementsTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    TransformSequence,
    relation_to_transform_canonic,
    transform_to_relation,
)

TILT = Affine.from_array(((1.0, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), "zyx")
"""z = z + 0.5 * x"""


def atoms(transform):
    return list(transform.transforms) if isinstance(transform, TransformSequence) else [transform]


def canonical(relation, source_axes):
    return relation_to_transform_canonic(relation, source_axes)


def test_inserted_axis_that_depends_on_other_axes_is_inserted_where_it_is():
    """A tilted plane in 3D: z is inserted first, then z = z + 0.5 * x"""
    relation = SpatialRelationSequence((ProjectionTo("zyx"), TILT))
    transform = canonical(relation, "yx")
    assert atoms(transform) == [
        ProjectAxisTransform(inserts=(0,)),
        AffineTransform(affine=((1.0, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))),
    ]
    assert transform_to_relation(transform, "yx", "zyx") == relation


def test_offset_of_such_an_affine_stays_in_it():
    relation = SpatialRelationSequence((ProjectionTo("zyx"), TILT, Translation(z=2.0, y=0.0, x=0.0)))
    transform = canonical(relation, "yx")
    assert [type(a) for a in atoms(transform)] == [ProjectAxisTransform, AffineTransform]
    recovered = transform_to_relation(transform, "yx", "zyx")
    assert isinstance(recovered, SpatialRelationSequence) and recovered[0] == ProjectionTo("zyx")
    assert canonical(recovered, "yx") == transform


def test_axis_that_feeds_another_one_is_dropped_where_it_is_not_last():
    """y depends on z, which is then dropped: z is the first axis, not the last"""
    feeding = Affine.from_array(((1.0, 0.0, 0.0, 0.0), (0.5, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), "zyx")
    relation = SpatialRelationSequence((feeding, ProjectionTo("yx"), Translation(y=1.0, x=2.0)))
    transform = canonical(relation, "zyx")
    assert [type(a) for a in atoms(transform)] == [MapAxisTransform, AffineTransform, ProjectAxisTransform]
    recovered = transform_to_relation(transform, "zyx", "yx")
    assert isinstance(recovered, SpatialRelationSequence)
    assert recovered[0] == PermutationTo("yxz")  # z moves to the end, where it is dropped
    assert recovered[-1] == ProjectionTo("yx")
    assert canonical(recovered, "zyx") == transform


def test_affine_before_dropping_axes_is_padded_to_an_invertible_matrix():
    # The surviving rows each depend on y + x, so copying z as the final row leaves the square matrix singular.
    mixing = Affine.from_array(((0, 1, 0, 0), (1, 1, 1, 0), (2, 1, 1, 0)), "zyx")
    transform = canonical(SpatialRelationSequence((mixing, ProjectionTo("yx"))), "zyx")
    assert isinstance(transform, TransformSequence)
    reorder, affine, project = transform.transforms
    assert reorder == MapAxisTransform((1, 2, 0))
    assert isinstance(affine, AffineTransform) and affine.is_invertible
    assert project == ProjectAxisTransform(drops=(2,))
    assert transform.collapsed() == AffineTransform(((1, 1, 1, 0), (2, 1, 1, 0)))


def test_replaced_axis_is_kept_at_its_position_when_a_rotation_leaves_a_different_axis_zero():
    # Insert c in place of z, then c' = x and x' = -c. The zero row for x' does not make x the inserted axis.
    rotation = Affine.from_array(((0, 0, 1, 0), (0, 1, 0, 0), (-1, 0, 0, 0)), "cyx")
    relation = SpatialRelationSequence((ProjectionTo("cyx"), rotation))
    transform = canonical(relation, "zyx")
    recovered = transform_to_relation(transform, "zyx", "cyx")
    assert isinstance(recovered, SpatialRelationSequence)
    projection, affine = recovered
    assert projection == ProjectionTo("cyx")
    assert isinstance(affine, Affine)
    assert tuple(row[1:] for row in affine.to_tuples()) == ((0, 1, 0), (1, 0, 0), (0, 0, 0))
    assert canonical(recovered, "zyx") == transform


def test_replacing_an_axis_and_zeroing_a_continued_axis_are_distinct():
    zero_y = Affine.from_array(((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), "cyx")
    relation = SpatialRelationSequence((ProjectionTo("cyx"), zero_y))
    transform = canonical(relation, "zyx")
    assert transform_to_relation(transform, "zyx", "cyx") == relation


def test_inserting_reordering_and_mixing_together():
    mixing = Affine.from_array(((1.0, 0.5, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), "zxy")
    relation = SpatialRelationSequence((PermutationTo("xy"), ProjectionTo("zxy"), mixing))
    transform = canonical(relation, "yx")
    assert [type(a) for a in atoms(transform)] == [ProjectAxisTransform, MapAxisTransform, AffineTransform]
    assert transform_to_relation(transform, "yx", "zxy").target_axes("yx") == ("z", "x", "y")
    assert canonical(transform_to_relation(transform, "yx", "zxy"), "yx") == transform


def test_axes_of_a_path_over_several_transforms_are_tracked_across_them():
    edge = canonical(SpatialRelationSequence((ProjectionTo("zyx"), TILT)), "yx")
    shift = canonical(Translation(z=1.0, y=2.0, x=3.0), "zyx")
    path = TransformSequence((edge, shift)).canonicalized()
    assert [type(a) for a in atoms(path)] == [ProjectAxisTransform, AffineTransform]
    recovered = transform_to_relation(path, "yx", "zyx")
    assert isinstance(recovered, SpatialRelationSequence) and recovered[0] == ProjectionTo("zyx")


@pytest.mark.parametrize(
    "relation, source_axes",
    [
        (SpatialRelationSequence((ProjectionTo("zyx"), TILT)), "yx"),
        (SpatialRelationSequence((TILT, ProjectionTo("zy"))), "zyx"),
    ],
)
def test_canonicalization_is_idempotent(relation, source_axes):
    transform = canonical(relation, source_axes)
    assert TransformSequence((transform,)).canonicalized() == transform


def test_a_bare_affine_that_adds_axes_keeps_the_convention_that_new_axes_come_last():
    """There is nothing to take the position from, so it stays as it was"""
    bare = AffineTransform(affine=((0.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))  # 3 outputs from 2 inputs
    assert TransformSequence((bare,)).canonicalized() == bare.simplified()


def test_chains_with_transforms_that_are_not_affine_are_canonicalized_as_before():
    chain = TransformSequence((ProjectAxisTransform(inserts=(0,)), DisplacementsTransform(path="d")))
    assert chain.canonicalized() == chain.collapsed().simplified()


def test_a_new_axis_named_like_a_dropped_one_that_is_still_needed_cannot_be_expressed():
    """Known limitation: z feeds y, then z (and x) are dropped, then a *new* z is inserted. The canonical form
    inserts before the affine, where the old and the new z would exist at once, which no relations can name."""
    feeding = Affine.from_array(((1.0, 0.0, 0.0, 0.0), (0.5, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), "zyx")
    relation = SpatialRelationSequence((feeding, ProjectionTo("y"), ProjectionTo("zy")))
    transform = canonical(relation, "zyx")
    with pytest.raises(CannotConvertToSpatialRelationError, match="same name"):
        transform_to_relation(transform, "zyx", "zy")
