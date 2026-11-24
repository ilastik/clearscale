import math

import pytest

from clearscale import Affine, AxisRenaming, Factor, Multiscale, SameSpace, Scale, Shape, Translation
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
    Transform,
    BijectionTransform,
    CoordinatesTransform,
    NodeRef,
)
from clearscale._transforms._to_from_spatial_relation import transform_to_relation, _flatten_atomic
from clearscale._services.ome_zarr import MultiscaleTransforms
from clearscale._transforms import _ByDimensionChild
from ._random_relations import random_relation_chain, chain_mixes_axes


def scaling_relation(**factors: float) -> Affine:
    # Maybe inline when the exact form of scaling relations is settled
    return Affine.from_factor(Factor(factors))


def shifted(**translation: float) -> Affine:
    """A pure translation is expressed as an Affine: transform_to_relation never returns Translation"""
    return Affine(translation=Translation(translation))


def scaled_and_shifted(factors: dict, translation: dict) -> Affine:
    return Affine(linear=scaling_relation(**factors).linear, translation=Translation(translation))


def parts_of(relation: SpatialRelation) -> list:
    return list(relation) if isinstance(relation, SpatialRelationSequence) else [relation]


def affines_in(relation: SpatialRelation) -> list:
    return [r for r in parts_of(relation) if isinstance(r, Affine)]


def assert_same_payloads(actual: Transform, expected: Transform):
    actual_atomic = _flatten_atomic(actual)
    expected_atomic = _flatten_atomic(expected)

    assert [type(t) for t in actual_atomic] == [type(t) for t in expected_atomic]
    for a, e in zip(actual_atomic, expected_atomic):
        if isinstance(e, ScaleTransform):
            assert isinstance(a, ScaleTransform)
            assert a.scale == pytest.approx(e.scale)
        elif isinstance(e, TranslationTransform):
            assert isinstance(a, TranslationTransform)
            assert a.translation == pytest.approx(e.translation)
        elif isinstance(e, RotationTransform):
            assert isinstance(a, RotationTransform)
            assert a.rotation is not None and e.rotation is not None
            assert [v for row in a.rotation for v in row] == pytest.approx([v for row in e.rotation for v in row])
        elif isinstance(e, AffineTransform):
            assert isinstance(a, AffineTransform)
            assert a.affine is not None and e.affine is not None
            assert [v for row in a.affine for v in row] == pytest.approx([v for row in e.affine for v in row])
        else:
            assert a == e


COS, SIN = math.cos(0.3), math.sin(0.3)


class TestRoundtrip:
    @staticmethod
    def roundtrip(relation: SpatialRelation, source_axes: str):
        """relation -> canonical transform -> relation -> canonical transform must reproduce the transform"""
        transform = relation_to_transform_canonic(relation, source_axes)
        recovered = transform_to_relation(transform, source_axes, relation.target_axes(source_axes))
        assert recovered is not None
        assert recovered.target_axes(source_axes) == relation.target_axes(source_axes)
        assert_same_payloads(relation_to_transform_canonic(recovered, source_axes), transform)
        return recovered

    @pytest.mark.parametrize(
        "relation, source_axes",
        [
            (scaling_relation(z=1, y=2, x=2), "zyx"),
            (scaling_relation(c=1, y=2.5, x=0.5), "cyx"),
            (ProjectionTo("x"), "yx"),
            (ProjectionTo("cyx"), "yx"),
            (ProjectionTo("yx"), "zyx"),
            (ProjectionTo("tzyx"), "yx"),
            (ProjectionTo("czx"), "zyx"),
            (PermutationTo("xyz"), "zyx"),
            (PermutationTo("yxz"), "zyx"),
            (AxisRearrangementTo("cxy"), "yx"),
            (AxisRearrangementTo("xz"), "zyx"),
            (AxisRearrangementTo("xyc"), "zyx"),
            (SpatialRelationSequence((ProjectionTo("yx"), AxisRearrangementTo("xyc"))), "cyx"),
        ],
    )
    def test_simple_relations_and_canonical_sequences_roundtrip_through_transform(self, relation, source_axes):
        assert self.roundtrip(relation, source_axes) == relation

    @pytest.mark.parametrize(
        "relation, source_axes, expected",
        [
            (Translation(z=0.0, y=1.5, x=-2.0), "zyx", shifted(z=0.0, y=1.5, x=-2.0)),
            (
                SpatialRelationSequence((scaling_relation(y=2, x=2), Translation(y=1.0, x=1.0))),
                "yx",
                scaled_and_shifted(dict(y=2, x=2), dict(y=1.0, x=1.0)),
            ),
            (
                SpatialRelationSequence(
                    (PermutationTo("xyz"), scaling_relation(x=2, y=4, z=1), Translation(x=1.0, y=0, z=0))
                ),
                "zyx",
                SpatialRelationSequence(
                    (PermutationTo("xyz"), scaled_and_shifted(dict(x=2, y=4, z=1), dict(x=1.0, y=0, z=0)))
                ),
            ),
        ],
    )
    def test_translations_roundtrip_as_affines(self, relation, source_axes, expected):
        assert self.roundtrip(relation, source_axes) == expected

    def test_arbitrary_sequences_may_roundtrip_equivalent_not_equal(self):
        # The transform can't express all information about the specific arrangement of relations it came from
        relation = SpatialRelationSequence((scaling_relation(z=1, y=2, x=2), ProjectionTo("cyx")))
        source_axes = "zyx"
        recovered = self.roundtrip(relation, source_axes)

        expected_equivalent = SpatialRelationSequence((ProjectionTo("cyx"), scaling_relation(c=1, y=2, x=2)))
        assert recovered == expected_equivalent, "equivalent but not equal"

    def test_reordering_with_inserts_is_recovered_as_one_structural_relation(self):
        recovered = self.roundtrip(AxisRearrangementTo("cxy"), "yx")
        assert recovered == AxisRearrangementTo("cxy"), "projectAxis+mapAxis should roundtrip as a single relation"

    @pytest.mark.parametrize(
        "transform",
        [
            RotationTransform(rotation=((COS, -SIN), (SIN, COS))),
            RotationTransform(
                rotation=((0.0, -1.0), (1.0, 0.0))
            ),  # Not a permutation: Maps axes one-to-one, but flips one
            AffineTransform(affine=((1.0, 0.5, 3.0), (0.0, 1.0, 4.0))),  # shear
            AffineTransform(affine=((2.0, 1.0, 0.0), (-1.0, 3.0, 5.0))),
            ScaleTransform((-1.0, 2.0)),  # reflection
        ],
        ids=lambda t: type(t).__name__,
    )
    def test_matrix_transforms_roundtrip_canonical(self, transform):
        relation = transform_to_relation(transform, "yx", "yx")
        assert isinstance(relation, Affine) or (
            isinstance(relation, SpatialRelationSequence) and any(isinstance(r, Affine) for r in relation)
        )
        recovered = relation_to_transform_canonic(relation, "yx")
        assert_same_payloads(recovered, TransformSequence((transform,)).canonicalized())

    def test_shearing_dropping_affine_roundtrips_canonical(self):
        # 1x2 linear component: y' = y + 0.5 x
        # Canonicalization pads it to a square matrix and splits off the dropped axis
        shearing_drop = AffineTransform(affine=((1.0, 0.5, 0.0),))
        relation = transform_to_relation(shearing_drop, "yx", "y")
        assert isinstance(relation, SpatialRelationSequence) and len(relation) == 2
        affine = relation[0]
        assert isinstance(affine, Affine) and affine.axes == ("y", "x")
        assert relation[1] == ProjectionTo("y")
        recovered = relation_to_transform_canonic(relation, "yx")
        assert_same_payloads(recovered, TransformSequence((shearing_drop,)).canonicalized())

    def test_bound_transform_from_multiscale_graph_roundtrips(self):
        ms = Multiscale.from_single(Scale(Shape(z=4, y=8, x=8), ome_zarr_axes="infer"))
        relations = SpatialRelationSequence(
            (scaling_relation(z=1, y=2, x=2), Translation(z=0.0, y=3.0, x=3.0), ProjectionTo("cyx"))
        )
        with_system = ms.with_coordinate_system("other", reached_by=relations)
        (transform,) = with_system._transform_graph.transforms
        assert isinstance(transform.source, NodeRef) and isinstance(transform.target, NodeRef)
        target_axes = transform.target.owner.axes
        assert target_axes == ("c", "y", "x")
        recovered = transform_to_relation(transform, transform.source.owner.axes, target_axes)
        assert recovered is not None
        assert_same_payloads(
            relation_to_transform_canonic(recovered, ms.axes), relation_to_transform_canonic(relations, ms.axes)
        )


class TestEdgeCases:
    def test_identity_is_same_space(self):
        assert transform_to_relation(IdentityTransform(), "zyx", "zyx") == SameSpace()
        assert transform_to_relation(ScaleTransform((1.0, 1.0)), "yx", "yx") == SameSpace()
        assert transform_to_relation(MapAxisTransform((0, 1)), "yx", "yx") == SameSpace()
        assert transform_to_relation(TransformSequence((IdentityTransform(),)), "yx", "yx") == SameSpace()
        assert (
            transform_to_relation(
                BijectionTransform(forward=IdentityTransform(), inverse=ScaleTransform((3.0,))), "yx", "yx"
            )
            == SameSpace()
        ), "should unwrap bijection.forward"

    def test_non_canonical_inputs_are_canonicalized(self):
        two_scales = TransformSequence((ScaleTransform((0.5, 0.5)), ScaleTransform((0.5, 1.0))))
        assert transform_to_relation(two_scales, "yx", "yx") == scaling_relation(y=0.25, x=0.5)

        affine = AffineTransform(affine=((0.5, 0.0, 3.0), (0.0, 0.5, 4.0)))
        assert transform_to_relation(affine, "yx", "yx") == scaled_and_shifted(dict(y=0.5, x=0.5), dict(y=3.0, x=4.0))

    def test_legacy_multiscale_transforms_are_flattened(self):
        legacy = MultiscaleTransforms((ScaleTransform((0.5, 0.5)), TranslationTransform((1.0, 2.0))))
        assert transform_to_relation(legacy, "yx", "yx") == scaled_and_shifted(dict(y=0.5, x=0.5), dict(y=1.0, x=2.0))

    def test_worked_example_zyx_to_cyx(self):
        transform = TransformSequence((ProjectAxisTransform(drops=(0,), inserts=(0,)), ScaleTransform((1.0, 0.5, 0.5))))
        assert transform_to_relation(transform, "zyx", "cyx") == SpatialRelationSequence(
            (ProjectionTo("cyx"), scaling_relation(c=1.0, y=0.5, x=0.5))
        )

    def test_dropped_and_inserted_axis_of_same_name_becomes_projection_chain(self):
        transform = ProjectAxisTransform(drops=(0,), inserts=(0,))
        relation = transform_to_relation(transform, "cyx", "cyx")
        assert relation is not None
        assert relation == SpatialRelationSequence((ProjectionTo("yx"), ProjectionTo("cyx")))
        # And is faithful: a single ProjectionTo("cyx") would instead mean "keep c" (identity)
        assert_same_payloads(relation_to_transform_canonic(relation, "cyx"), transform)

    def test_same_name_collision_with_reordering_uses_rearrangement_for_second_step(self):
        # c1,y,x -permute-> y,x -rearrange-> x,y,c2 is the simplest representation that distinguishes
        # c,y,x -(drop c, insert c, swap xy)-> x,y,c
        # from the permutation that preserves c:
        # c,y,x -(permute)-> x,y,c
        # so it should roundtrip exactly.
        relation = SpatialRelationSequence((ProjectionTo("yx"), AxisRearrangementTo("xyc")))
        transform = relation_to_transform_canonic(relation, "cyx")
        assert transform_to_relation(transform, "cyx", "xyc") == relation

    def test_collision_inside_longer_chain(self):
        transform = TransformSequence(
            (
                ScaleTransform((1.0, 2.0, 2.0)),
                ProjectAxisTransform(drops=(0,), inserts=(0,)),
                TranslationTransform((0, 1, 1)),
            )
        )
        relation = transform_to_relation(transform, "cyx", "cyx")
        assert relation is not None
        assert_same_payloads(
            relation_to_transform_canonic(relation, "cyx"), TransformSequence((transform,)).canonicalized()
        )

    def test_rotation_relation_has_the_same_matrix(self):
        relation = transform_to_relation(RotationTransform(rotation=((COS, -SIN), (SIN, COS))), "yx", "yx")
        assert isinstance(relation, Affine)
        assert relation.to_tuples() == ((COS, -SIN, 0.0), (SIN, COS, 0.0))

    def test_affine_after_projection_uses_the_names_of_the_projected_axes(self):
        transform = TransformSequence(
            (ProjectAxisTransform(drops=(0,)), AffineTransform(affine=((1.0, 0.5, 0.0), (0.0, 1.0, 0.0))))
        )
        relation = transform_to_relation(transform, "zyx", "yx")
        assert isinstance(relation, SpatialRelationSequence)
        assert len(relation) == 2
        assert relation[0] == ProjectionTo("yx")
        affine = relation[1]
        assert isinstance(affine, Affine) and affine.axes == ("y", "x")

    def test_dropping_scaling_affine_splits(self):
        # zero-col is simply dropped first, leaving a scaling of the one remaining axis
        dropping_scaling = AffineTransform(affine=((2.0, 0.0, 0.0),))
        assert transform_to_relation(dropping_scaling, "yx", "y") == SpatialRelationSequence(
            (ProjectionTo("y"), scaling_relation(y=2.0))
        )


class TestPhysicalRuns:
    """Consecutive physical transforms (scale, translation, rotation, affine) are one Affine, which includes the
    translation. Translation is never returned."""

    def test_scale_then_translation_is_one_affine(self):
        transform = TransformSequence((ScaleTransform((2.0, 3.0)), TranslationTransform((1.0, 1.0))))
        relation = transform_to_relation(transform, "yx", "yx")
        assert relation == scaled_and_shifted(dict(y=2.0, x=3.0), dict(y=1.0, x=1.0))
        assert isinstance(relation, Affine) and relation.translation == Translation(y=1.0, x=1.0)

    def test_translation_then_scale_scales_the_translation(self):
        transform = TransformSequence((TranslationTransform((1.0, 1.0)), ScaleTransform((2.0, 3.0))))
        assert transform_to_relation(transform, "yx", "yx") == scaled_and_shifted(
            dict(y=2.0, x=3.0), dict(y=2.0, x=3.0)
        )

    def test_rotation_scale_translation_is_one_affine(self):
        transform = TransformSequence(
            (
                RotationTransform(rotation=((COS, -SIN), (SIN, COS))),
                ScaleTransform((2.0, 2.0)),
                TranslationTransform((5.0, 6.0)),
            )
        )
        relation = transform_to_relation(transform, "yx", "yx")
        assert isinstance(relation, Affine)
        assert [v for row in relation.to_tuples() for v in row] == pytest.approx(
            [2 * COS, -2 * SIN, 5.0, 2 * SIN, 2 * COS, 6.0]
        )

    def test_pure_translation_is_an_affine_with_identity_linear(self):
        relation = transform_to_relation(TranslationTransform((1.0, 2.0)), "yx", "yx")
        assert isinstance(relation, Affine)
        assert relation.linear.is_identity()
        assert relation.translation == Translation(y=1.0, x=2.0)

    def test_structural_relations_stay_separate_from_the_affine(self):
        transform = TransformSequence(
            (MapAxisTransform((2, 1, 0)), ScaleTransform((2.0, 1.0, 1.0)), TranslationTransform((1.0, 0.0, 0.0)))
        )
        relation = transform_to_relation(transform, "zyx", "xyz")
        assert relation == SpatialRelationSequence(
            (PermutationTo("xyz"), scaled_and_shifted(dict(x=2.0, y=1.0, z=1.0), dict(x=1.0, y=0.0, z=0.0)))
        )

    def test_offset_stays_in_the_affine_when_it_uses_an_axis_that_is_dropped(self):
        # z depends on x, which is dropped afterwards, so the affine has to act on all three axes before the drop
        shear = Affine.from_array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], "zyx")
        chain = SpatialRelationSequence((shear, ProjectionTo("zy"), Translation(z=1.0, y=2.0)))
        transform = relation_to_transform_canonic(chain, "zyx")
        relation = transform_to_relation(transform, "zyx", "zy")
        assert isinstance(relation, SpatialRelationSequence) and len(relation) == 2
        affine, projection = relation
        assert projection == ProjectionTo("zy")
        assert isinstance(affine, Affine) and affine.axes == ("z", "y", "x")
        assert affine.to_tuples() == ((1.0, 0.0, 0.5, 1.0), (0.0, 1.0, 0.0, 2.0), (0.0, 0.0, 1.0, 0.0))
        assert_same_payloads(relation_to_transform_canonic(relation, "zyx"), transform)

    def test_translation_after_a_lone_projection_is_an_affine_after_it(self):
        chain = SpatialRelationSequence((ProjectionTo("yx"), Translation(y=1.0, x=2.0)))
        transform = relation_to_transform_canonic(chain, "zyx")
        assert transform_to_relation(transform, "zyx", "yx") == SpatialRelationSequence(
            (ProjectionTo("yx"), shifted(y=1.0, x=2.0))
        )

    def test_one_affine_with_axis_changes_before_and_after(self):
        shear = Affine.from_array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.0, 0.0]], "xyz")
        chain = SpatialRelationSequence(
            (PermutationTo("xyz"), shear, ProjectionTo("xy"), Translation(x=2.0, y=1.0), PermutationTo("yx"))
        )
        transform = relation_to_transform_canonic(chain, "zyx")
        relation = transform_to_relation(transform, "zyx", "yx")
        assert len(affines_in(relation)) == 1
        assert not any(isinstance(r, Translation) for r in parts_of(relation))
        assert_same_payloads(relation_to_transform_canonic(relation, "zyx"), transform)


class TestErrors:
    def test_dropping_and_reinserting_every_axis_under_same_names_raises(self):
        with pytest.raises(CannotConvertToSpatialRelationError, match="re-inserted"):
            transform_to_relation(ProjectAxisTransform(drops=(0,), inserts=(0,)), "z", "z")

    def test_unsupported_transforms_raise(self):
        with pytest.raises(
            CannotConvertToSpatialRelationError, match="CoordinatesTransform has no SpatialRelation equivalent"
        ):
            transform_to_relation(CoordinatesTransform(path="d"), "yx", "yx")
        with pytest.raises(
            CannotConvertToSpatialRelationError, match="DisplacementsTransform has no SpatialRelation equivalent"
        ):
            transform_to_relation(DisplacementsTransform(path="d"), "yx", "yx")
        mixing_child = _ByDimensionChild(
            source_indices=(0, 1),
            target_indices=(0, 1),
            transform=RotationTransform(rotation=((0.8, -0.6), (0.6, 0.8))),
        )
        with pytest.raises(
            CannotConvertToSpatialRelationError, match="ByDimensionTransform has no SpatialRelation equivalent"
        ):
            transform_to_relation(ByDimensionTransform((mixing_child,)), "yx", "yx")
        path_backed_matrix_error = (
            r"values are not loaded \(path-backed\), so it cannot determine how input axes correspond to outputs"
        )
        with pytest.raises(CannotConvertToSpatialRelationError, match=path_backed_matrix_error):
            transform_to_relation(RotationTransform(_ome_zarr_path="r"), "yx", "yx")
        with pytest.raises(CannotConvertToSpatialRelationError, match=path_backed_matrix_error):
            transform_to_relation(AffineTransform(_ome_zarr_path="r"), "yx", "yx")
        # scale and transform can still determine axis correspondence when path-backed
        path_backed_equal_axes_error = r"values are not loaded \(path-backed\); please request if you need this"
        with pytest.raises(CannotConvertToSpatialRelationError, match=path_backed_equal_axes_error):
            transform_to_relation(ScaleTransform(scale=(), _ome_zarr_path="p"), "yx", "yx")
        with pytest.raises(CannotConvertToSpatialRelationError, match=path_backed_equal_axes_error):
            transform_to_relation(TranslationTransform(translation=(), _ome_zarr_path="p"), "yx", "yx")

    @pytest.mark.parametrize(
        "transform", [ScaleTransform((0.0, 1.0)), AffineTransform(affine=((1.0, 2.0, 0.0), (2.0, 4.0, 0.0)))]
    )
    def test_non_invertible_matrices_can_convert_to_affine(self, transform):
        relation = transform_to_relation(transform, "yx", "yx")
        assert relation is not None
        assert isinstance(relation, Affine)
        assert relation.to_tuples() == transform._to_affine_transform(source_ndim=2, target_ndim=2).affine

    def test_transform_not_fitting_axes_raises_plain_value_error(self):
        for transform, source, target in (
            (ScaleTransform((2.0, 2.0)), "zyx", "zyx"),
            (ProjectAxisTransform(inserts=(0,)), "yx", "yx"),
        ):
            with pytest.raises(ValueError) as exc_info:
                transform_to_relation(transform, source, target)
            assert not isinstance(exc_info.value, CannotConvertToSpatialRelationError)


@pytest.mark.parametrize("seed", range(1000))
def test_random_relation_chains_roundtrip_through_transforms(seed):
    source, relation = random_relation_chain(seed)
    target = relation.target_axes(source)
    transform = relation_to_transform_canonic(relation, source)
    mixes = chain_mixes_axes(relation, source)
    assert TransformSequence((transform,)).canonicalized() == transform, "canonicalization is idempotent"
    try:
        recovered = transform_to_relation(transform, source, target)
    except CannotConvertToSpatialRelationError as error:
        # The one thing that relations cannot express: Axes are dropped, and others with the same names are inserted
        # (the dropped ones may be the only ones that were left, or live on under another name).
        assert "same name" in str(error), f"only reused names are inexpressible, but: {error}"
        return
    if recovered == SameSpace():
        assert isinstance(transform, IdentityTransform)
        assert target == source
        return
    assert recovered.target_axes(source) == target
    assert_same_payloads(relation_to_transform_canonic(recovered, source), transform)
    assert not any(isinstance(r, Translation) for r in parts_of(recovered)), "translations are part of the Affine"
    assert len(affines_in(recovered)) <= 1, "all physical transformation is combined in one Affine"
    if not mixes:
        # Same axis lineage, whichever way it is expressed. (For chains that mix axes, both sides are only
        # upper bounds that may differ by numerical cancellation, see SpatialRelationSequence._axis_links.)
        assert recovered._axis_links(source).sources == relation._axis_links(source).sources


class TestRenaming:
    def test_different_axis_names_at_the_ends_become_a_final_renaming(self):
        assert transform_to_relation(IdentityTransform(), "zyx", "ZYX") == AxisRenaming(z="Z", y="Y", x="X")
        assert transform_to_relation(IdentityTransform(), "zyx", "zYx") == AxisRenaming(y="Y")
        assert transform_to_relation(TranslationTransform((2.0, 2.0)), "yx", "vu") == SpatialRelationSequence(
            (shifted(y=2.0, x=2.0), AxisRenaming(y="v", x="u"))
        )

    def test_swapped_axis_names_become_one_simultaneous_renaming_not_a_permutation(self):
        assert transform_to_relation(IdentityTransform(), "yx", "xy") == AxisRenaming(y="x", x="y")
        # Moving the data, names stay positional: swap the data, then call the (now first) axis "y" again
        assert transform_to_relation(MapAxisTransform((1, 0)), "yx", "yx") == SpatialRelationSequence(
            (PermutationTo("xy"), AxisRenaming(y="x", x="y"))
        )
        assert transform_to_relation(MapAxisTransform((1, 0)), "yx", "xy") == PermutationTo("xy")

    def test_inserted_axes_are_not_renamed_and_dropped_axes_need_no_renaming(self):
        transform = ProjectAxisTransform(drops=(0,), inserts=(0,))
        # z is dropped and c is a new axis; y is renamed
        assert transform_to_relation(transform, "zyx", "cYx") == SpatialRelationSequence(
            (ProjectionTo("cyx"), AxisRenaming(y="Y"))
        )

    def test_name_reused_for_new_axis_while_the_original_lives_on_under_another_name_cannot_be_expressed(self):
        # The new axis is called "x" and the old "x" is called "b": Both are "x" until the final renaming.
        with pytest.raises(CannotConvertToSpatialRelationError, match="ambiguous"):
            transform_to_relation(ProjectAxisTransform(inserts=(0,)), "yx", ("x", "a", "b"))
        # Fine if the old x is dropped instead: Drop it, insert the new x, and rename y
        assert transform_to_relation(ProjectAxisTransform(drops=(1,), inserts=(0,)), "yx", ("x", "a")) == (
            SpatialRelationSequence((ProjectionTo("y"), ProjectionTo(("x", "y")), AxisRenaming(y="a")))
        )
