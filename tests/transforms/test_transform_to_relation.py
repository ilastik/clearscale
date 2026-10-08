import math

import pytest

from clearscale import Affine, AxisRenaming, Factor, Multiscale, Scale, Shape, Translation
from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._spatial_relations import (
    AxisRearrangementTo,
    PermutationTo,
    ProjectionTo,
    SpatialRelation,
    SpatialRelationSequence,
)
from clearscale._transforms import (
    AffineTransform,
    ByDimensionTransform,
    DisplacementsTransform,
    IdentityTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    ScaleTransform,
    TransformSequence,
    TranslationTransform,
    relation_to_transform_canonic,
)
from clearscale._transforms._to_from_spatial_relation import _flatten_leaves, transform_to_relation
from clearscale._services.ome_zarr import MultiscaleTransforms
from clearscale._transforms import _ByDimensionChild


def scaled(**factors: float) -> Affine:
    """The relation that was a Factor before Factor stopped being a relation"""
    return Affine.from_factor(Factor(factors))


from ._random_relations import chain_mixes_axes, random_relation_chain


def assert_same_leaves(actual, expected):
    """Compare canonical transforms by leaf type and payload, tolerating float rounding (1 / (1 / f))."""
    actual_leaves, expected_leaves = _flatten_leaves(actual), _flatten_leaves(expected)
    assert [type(t) for t in actual_leaves] == [type(t) for t in expected_leaves]
    for a, e in zip(actual_leaves, expected_leaves):
        if isinstance(e, ScaleTransform):
            assert a.scale == pytest.approx(e.scale)
        elif isinstance(e, TranslationTransform):
            assert a.translation == pytest.approx(e.translation)
        elif isinstance(e, MapAxisTransform):
            assert a.map_axis == e.map_axis
        elif isinstance(e, RotationTransform):
            assert [v for row in a.rotation for v in row] == pytest.approx([v for row in e.rotation for v in row])
        elif isinstance(e, AffineTransform):
            assert [v for row in a.affine for v in row] == pytest.approx([v for row in e.affine for v in row])
        else:
            assert (a.drops, a.inserts) == (e.drops, e.inserts)


def roundtrip(relation: SpatialRelation, source_axes: str):
    """relation -> canonical transform -> relation -> canonical transform must reproduce the transform"""
    transform = relation_to_transform_canonic(relation, source_axes)
    target_axes = relation.target_axes(source_axes)
    recovered = transform_to_relation(transform, source_axes, target_axes)
    assert recovered is not None
    assert recovered.target_axes(source_axes) == target_axes
    assert_same_leaves(relation_to_transform_canonic(recovered, source_axes), transform)
    return recovered


@pytest.mark.parametrize(
    "relation, source_axes",
    [
        (scaled(z=1, y=2, x=2), "zyx"),
        (scaled(c=1, y=2.5, x=0.5), "cyx"),
        (Translation(z=0.0, y=1.5, x=-2.0), "zyx"),
        (ProjectionTo("cyx"), "yx"),
        (ProjectionTo("yx"), "zyx"),
        (ProjectionTo("tzyx"), "yx"),
        (ProjectionTo("czx"), "zyx"),
        (PermutationTo("xyz"), "zyx"),
        (PermutationTo("yxz"), "zyx"),
        (AxisRearrangementTo("cxy"), "yx"),
        (AxisRearrangementTo("xz"), "zyx"),
        (AxisRearrangementTo("xyc"), "zyx"),
        (SpatialRelationSequence((scaled(y=2, x=2), Translation(y=1.0, x=1.0))), "yx"),
        (SpatialRelationSequence((scaled(z=1, y=2, x=2), ProjectionTo("cyx"))), "zyx"),
        (SpatialRelationSequence((PermutationTo("xyz"), scaled(x=2, y=4, z=1), Translation(x=1.0, y=0, z=0))), "zyx"),
        (SpatialRelationSequence((ProjectionTo("yx"), AxisRearrangementTo("xyc"))), "cyx"),
    ],
)
def test_relation_roundtrips_through_transform(relation, source_axes):
    roundtrip(relation, source_axes)


def test_simple_relations_are_recovered_as_themselves():
    assert roundtrip(scaled(z=1, y=2, x=2), "zyx") == scaled(z=1, y=2, x=2)
    assert roundtrip(Translation(y=1.5, x=2.0), "yx") == Translation(y=1.5, x=2.0)
    assert roundtrip(ProjectionTo("cyx"), "yx") == ProjectionTo("cyx")
    assert roundtrip(ProjectionTo("x"), "yx") == ProjectionTo("x")
    assert roundtrip(PermutationTo("xy"), "yx") == PermutationTo("xy")


def test_reordering_with_inserts_is_recovered_as_one_structural_relation():
    # Project + Map in the same run must come back as one relation, not as one per leaf
    recovered = roundtrip(AxisRearrangementTo("cxy"), "yx")
    assert recovered == AxisRearrangementTo("cxy")


def test_identity_is_none():
    assert transform_to_relation(IdentityTransform(), "zyx", "zyx") is None
    assert transform_to_relation(ScaleTransform((1.0, 1.0)), "yx", "yx") is None
    assert transform_to_relation(MapAxisTransform((0, 1)), "yx", "yx") is None
    assert transform_to_relation(TransformSequence((IdentityTransform(),)), "yx", "yx") is None


def test_non_canonical_inputs_are_canonicalized():
    two_scales = TransformSequence((ScaleTransform((0.5, 0.5)), ScaleTransform((0.5, 1.0))))
    assert transform_to_relation(two_scales, "yx", "yx") == scaled(y=4.0, x=2.0)
    affine = AffineTransform(affine=((0.5, 0.0, 3.0), (0.0, 0.5, 4.0)))
    assert transform_to_relation(affine, "yx", "yx") == SpatialRelationSequence(
        (scaled(y=2.0, x=2.0), Translation(y=-3.0, x=-4.0))
    )


def test_legacy_multiscale_transforms_are_flattened():
    legacy = MultiscaleTransforms((ScaleTransform((0.5, 0.5)), TranslationTransform((1.0, 2.0))))
    assert transform_to_relation(legacy, "yx", "yx") == SpatialRelationSequence(
        (scaled(y=2.0, x=2.0), Translation(y=-1.0, x=-2.0))
    )


def test_worked_example_zyx_to_cyx():
    transform = TransformSequence((ProjectAxisTransform(drops=(0,), inserts=(0,)), ScaleTransform((1.0, 0.5, 0.5))))
    assert transform_to_relation(transform, "zyx", "cyx") == SpatialRelationSequence(
        (ProjectionTo("cyx"), scaled(c=1.0, y=2.0, x=2.0))
    )


def test_dropped_and_inserted_axis_of_same_name_becomes_projection_chain():
    transform = ProjectAxisTransform(drops=(0,), inserts=(0,))
    relation = transform_to_relation(transform, "cyx", "cyx")
    assert relation == SpatialRelationSequence((ProjectionTo("yx"), ProjectionTo("cyx")))
    # And is faithful: a single ProjectionTo("cyx") would instead mean "keep c" (identity)
    assert_same_leaves(relation_to_transform_canonic(relation, "cyx"), transform)


def test_same_name_collision_with_reordering_uses_rearrangement_for_second_step():
    relation = SpatialRelationSequence((ProjectionTo("yx"), AxisRearrangementTo("xyc")))
    transform = relation_to_transform_canonic(relation, "cyx")
    assert transform_to_relation(transform, "cyx", "xyc") == relation


def test_collision_inside_longer_chain():
    transform = TransformSequence(
        (
            ScaleTransform((1.0, 2.0, 2.0)),
            ProjectAxisTransform(drops=(0,), inserts=(0,)),
            TranslationTransform((0, 1, 1)),
        )
    )
    relation = transform_to_relation(transform, "cyx", "cyx")
    assert relation is not None
    assert_same_leaves(relation_to_transform_canonic(relation, "cyx"), TransformSequence((transform,)).canonicalized())


def test_dropping_and_reinserting_every_axis_under_same_names_raises():
    with pytest.raises(CannotConvertToSpatialRelationError, match="re-inserted"):
        transform_to_relation(ProjectAxisTransform(drops=(0,), inserts=(0,)), "z", "z")


def test_different_axis_names_at_the_ends_become_a_final_renaming():
    assert transform_to_relation(IdentityTransform(), "zyx", "ZYX") == AxisRenaming(z="Z", y="Y", x="X")
    assert transform_to_relation(IdentityTransform(), "zyx", "zYx") == AxisRenaming(y="Y")
    assert transform_to_relation(ScaleTransform((2.0, 2.0)), "yx", "vu") == SpatialRelationSequence(
        (scaled(y=0.5, x=0.5), AxisRenaming(y="v", x="u"))
    )


def test_swapped_axis_names_become_one_simultaneous_renaming_not_a_permutation():
    assert transform_to_relation(IdentityTransform(), "yx", "xy") == AxisRenaming(y="x", x="y")
    # Moving the data, names stay positional: swap the data, then call the (now first) axis "y" again
    assert transform_to_relation(MapAxisTransform((1, 0)), "yx", "yx") == SpatialRelationSequence(
        (PermutationTo("xy"), AxisRenaming(y="x", x="y"))
    )
    assert transform_to_relation(MapAxisTransform((1, 0)), "yx", "xy") == PermutationTo("xy")


def test_renaming_comes_last_so_earlier_relations_speak_the_source_vocabulary():
    transform = TransformSequence((ScaleTransform((1.0, 0.5, 0.5)), TranslationTransform((0.0, 1.0, 1.0))))
    assert transform_to_relation(transform, "zyx", "zYX") == SpatialRelationSequence(
        (scaled(z=1.0, y=2.0, x=2.0), Translation(z=0.0, y=-1.0, x=-1.0), AxisRenaming(y="Y", x="X"))
    )


def test_inserted_axes_are_not_renamed_and_dropped_axes_need_no_renaming():
    transform = ProjectAxisTransform(drops=(0,), inserts=(0,))
    # z is dropped and c is a new axis; y is renamed
    assert transform_to_relation(transform, "zyx", "cYx") == SpatialRelationSequence(
        (ProjectionTo("cyx"), AxisRenaming(y="Y"))
    )


def test_name_reused_for_new_axis_while_the_original_lives_on_under_another_name_cannot_be_expressed():
    # The new axis is called "x" and the old "x" is called "b": Both are "x" until the final renaming.
    with pytest.raises(CannotConvertToSpatialRelationError, match="ambiguous"):
        transform_to_relation(ProjectAxisTransform(inserts=(0,)), "yx", ("x", "a", "b"))
    # Fine if the old x is dropped instead: Drop it, insert the new x, and rename y
    assert transform_to_relation(ProjectAxisTransform(drops=(1,), inserts=(0,)), "yx", ("x", "a")) == (
        SpatialRelationSequence((ProjectionTo("y"), ProjectionTo(("x", "y")), AxisRenaming(y="a")))
    )


def test_unsupported_transforms_raise():
    with pytest.raises(CannotConvertToSpatialRelationError, match="DisplacementsTransform"):
        transform_to_relation(DisplacementsTransform(path="d"), "yx", "yx")
    mixing_child = _ByDimensionChild(
        source_indices=(0, 1), target_indices=(0, 1), transform=RotationTransform(rotation=((0.8, -0.6), (0.6, 0.8)))
    )
    with pytest.raises(CannotConvertToSpatialRelationError, match="ByDimensionTransform"):
        transform_to_relation(ByDimensionTransform((mixing_child,)), "yx", "yx")
    with pytest.raises(CannotConvertToSpatialRelationError, match="path-backed"):
        transform_to_relation(TranslationTransform(translation=(), _ome_zarr_path="p"), "yx", "yx")
    with pytest.raises(CannotConvertToSpatialRelationError, match="no SpatialRelation equivalent|unknown"):
        transform_to_relation(RotationTransform(_ome_zarr_path="r"), "yx", "yx")


C, S = math.cos(0.3), math.sin(0.3)


@pytest.mark.parametrize(
    "transform",
    [
        RotationTransform(rotation=((C, -S), (S, C))),
        RotationTransform(rotation=((0.0, -1.0), (1.0, 0.0))),  # Maps axes one-to-one, but flips one: Not a permutation
        AffineTransform(affine=((1.0, 0.5, 3.0), (0.0, 1.0, 4.0))),  # shear
        AffineTransform(affine=((2.0, 1.0, 0.0), (-1.0, 3.0, 5.0))),
        ScaleTransform((-1.0, 2.0)),  # reflection
    ],
    ids=lambda t: type(t).__name__,
)
def test_matrix_transforms_become_the_inverse_affine_relation(transform):
    relation = transform_to_relation(transform, "yx", "yx")
    assert isinstance(relation, Affine) or (
        isinstance(relation, SpatialRelationSequence) and any(isinstance(r, Affine) for r in relation)
    )
    # Applying a relation to the image is the inverse of the coordinate transformation: Round trip must close
    assert_same_leaves(relation_to_transform_canonic(relation, "yx"), TransformSequence((transform,)).canonicalized())


def test_rotation_relation_has_the_transposed_matrix():
    relation = transform_to_relation(RotationTransform(rotation=((C, -S), (S, C))), "yx", "yx")
    assert isinstance(relation, Affine)
    assert [v for row in relation.to_tuples() for v in row] == pytest.approx([C, S, 0.0, -S, C, 0.0])


def test_non_invertible_matrix_raises_conversion_error():
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
        transform_to_relation(ScaleTransform((0.0, 1.0)), "yx", "yx")
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
        transform_to_relation(AffineTransform(affine=((1.0, 2.0, 0.0), (2.0, 4.0, 0.0))), "yx", "yx")


def test_affine_after_projection_uses_the_names_of_the_projected_axes():
    transform = TransformSequence(
        (ProjectAxisTransform(drops=(0,)), AffineTransform(affine=((1.0, 0.5, 0.0), (0.0, 1.0, 0.0))))
    )
    relation = transform_to_relation(transform, "zyx", "yx")
    assert isinstance(relation, SpatialRelationSequence)
    assert relation[0] == ProjectionTo("yx") and isinstance(relation[1], Affine)
    assert relation[1].axes == ("y", "x")


def test_path_backed_values_raise_conversion_error():
    with pytest.raises(CannotConvertToSpatialRelationError, match="path-backed"):
        transform_to_relation(ScaleTransform(scale=(), _ome_zarr_path="p"), "yx", "yx")


def test_transform_not_fitting_axes_raises_plain_value_error():
    for transform, source, target in (
        (ScaleTransform((2.0, 2.0)), "zyx", "zyx"),
        (ProjectAxisTransform(inserts=(0,)), "yx", "yx"),
    ):
        with pytest.raises(ValueError) as exc_info:
            transform_to_relation(transform, source, target)
        assert not isinstance(exc_info.value, CannotConvertToSpatialRelationError)


def test_bound_transform_from_multiscale_graph_roundtrips():
    ms = Multiscale.from_single(Scale(Shape(z=4, y=8, x=8), ome_zarr_axes="infer"))
    relations = SpatialRelationSequence((scaled(z=1, y=2, x=2), Translation(z=0.0, y=3.0, x=3.0), ProjectionTo("cyx")))
    with_system = ms.with_coordinate_system("other", reached_by=relations)
    (transform,) = with_system._transform_graph.transforms
    target_axes = transform.target.owner.axes
    assert target_axes == ("c", "y", "x")
    recovered = transform_to_relation(transform, transform.source.owner.axes, target_axes)
    assert_same_leaves(
        relation_to_transform_canonic(recovered, ms.axes), relation_to_transform_canonic(relations, ms.axes)
    )


def test_affine_that_changes_the_number_of_axes_is_split_into_projection_and_affine():
    # 1x2 matrix: y' = y + 0.5 x. Canonicalization pads it to a square matrix and splits off the dropped axis.
    shearing_drop = AffineTransform(affine=((1.0, 0.5, 0.0),))
    relation = transform_to_relation(shearing_drop, "yx", "y")
    assert isinstance(relation, SpatialRelationSequence) and len(relation) == 2
    assert isinstance(relation[0], Affine) and relation[1] == ProjectionTo("y")
    assert relation[0].axes == ("y", "x")
    assert_same_leaves(
        TransformSequence((relation_to_transform_canonic(relation, "yx"),)).canonicalized(),
        TransformSequence((shearing_drop,)).canonicalized(),
    )
    # A column that is zero is simply dropped first, leaving a scaling of the one remaining axis
    assert transform_to_relation(AffineTransform(affine=((2.0, 0.0, 0.0),)), "yx", "y") == SpatialRelationSequence(
        (ProjectionTo("y"), scaled(y=0.5))
    )


@pytest.mark.parametrize("seed", range(300))
def test_random_relation_chains_roundtrip_through_transforms(seed):
    source, relation = random_relation_chain(seed)
    target = relation.target_axes(source)
    transform = relation_to_transform_canonic(relation, source)
    mixes = chain_mixes_axes(relation, source)
    try:
        recovered = transform_to_relation(transform, source, target)
    except CannotConvertToSpatialRelationError:
        # Known limitation: An Affine that mixes axes, composed with relations that drop, insert or permute axes, is
        # collapsed into one matrix that may no longer be separable into those parts again.
        # Chains that don't mix axes always convert back.
        assert mixes, "chains that don't mix axes must always convert"
        return
    if recovered is None:
        assert_same_leaves(transform, IdentityTransform())
        assert target == source
        return
    assert recovered.target_axes(source) == target
    assert_same_leaves(relation_to_transform_canonic(recovered, source), transform)
    if not mixes:
        # Same axis lineage, whichever way it is expressed. (For chains that mix axes, both sides are only
        # upper bounds that may differ by numerical cancellation, see SpatialRelation._axis_dependencies.)
        assert recovered._axis_dependencies(source) == relation._axis_dependencies(source)
