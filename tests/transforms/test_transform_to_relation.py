import math

import pytest

from clearscale import Affine, Factor, Multiscale, Scale, Shape, Translation
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


def scaling_relation(**factors: float) -> Affine:
    # Maybe inline when the exact form of scaling relations is settled
    return Affine.from_factor(Factor(factors))


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
            (Translation(z=0.0, y=1.5, x=-2.0), "zyx"),
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
            (SpatialRelationSequence((scaling_relation(y=2, x=2), Translation(y=1.0, x=1.0))), "yx"),
            (
                SpatialRelationSequence(
                    (PermutationTo("xyz"), scaling_relation(x=2, y=4, z=1), Translation(x=1.0, y=0, z=0))
                ),
                "zyx",
            ),
            (SpatialRelationSequence((ProjectionTo("yx"), AxisRearrangementTo("xyc"))), "cyx"),
        ],
    )
    def test_simple_relations_and_canonical_sequences_roundtrip_through_transform(self, relation, source_axes):
        assert self.roundtrip(relation, source_axes) == relation

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
    def test_identity_is_none(self):
        assert transform_to_relation(IdentityTransform(), "zyx", "zyx") is None
        assert transform_to_relation(ScaleTransform((1.0, 1.0)), "yx", "yx") is None
        assert transform_to_relation(MapAxisTransform((0, 1)), "yx", "yx") is None
        assert transform_to_relation(TransformSequence((IdentityTransform(),)), "yx", "yx") is None
        assert (
            transform_to_relation(
                BijectionTransform(forward=IdentityTransform(), inverse=ScaleTransform((3.0,))), "yx", "yx"
            )
            is None
        ), "should unwrap bijection.forward"

    def test_non_canonical_inputs_are_canonicalized(self):
        two_scales = TransformSequence((ScaleTransform((0.5, 0.5)), ScaleTransform((0.5, 1.0))))
        assert transform_to_relation(two_scales, "yx", "yx") == scaling_relation(y=4.0, x=2.0)

        affine = AffineTransform(affine=((0.5, 0.0, 3.0), (0.0, 0.5, 4.0)))
        assert transform_to_relation(affine, "yx", "yx") == SpatialRelationSequence(
            (scaling_relation(y=2.0, x=2.0), Translation(y=-3.0, x=-4.0))
        )

    def test_legacy_multiscale_transforms_are_flattened(self):
        legacy = MultiscaleTransforms((ScaleTransform((0.5, 0.5)), TranslationTransform((1.0, 2.0))))
        assert transform_to_relation(legacy, "yx", "yx") == SpatialRelationSequence(
            (scaling_relation(y=2.0, x=2.0), Translation(y=-1.0, x=-2.0))
        )

    def test_worked_example_zyx_to_cyx(self):
        transform = TransformSequence((ProjectAxisTransform(drops=(0,), inserts=(0,)), ScaleTransform((1.0, 0.5, 0.5))))
        assert transform_to_relation(transform, "zyx", "cyx") == SpatialRelationSequence(
            (ProjectionTo("cyx"), scaling_relation(c=1.0, y=2.0, x=2.0))
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

    def test_rotation_relation_has_the_transposed_matrix(self):
        relation = transform_to_relation(RotationTransform(rotation=((COS, -SIN), (SIN, COS))), "yx", "yx")
        assert isinstance(relation, Affine)
        assert [v for row in relation.to_tuples() for v in row] == pytest.approx([COS, SIN, 0.0, -SIN, COS, 0.0])

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
            (ProjectionTo("y"), scaling_relation(y=0.5))
        )


class TestErrors:
    def test_dropping_and_reinserting_every_axis_under_same_names_raises(self):
        with pytest.raises(CannotConvertToSpatialRelationError, match="re-inserted"):
            transform_to_relation(ProjectAxisTransform(drops=(0,), inserts=(0,)), "z", "z")

    def test_relabeling_raises(self):
        with pytest.raises(CannotConvertToSpatialRelationError, match="relabeled"):
            transform_to_relation(IdentityTransform(), "zyx", "ZYX")
        with pytest.raises(CannotConvertToSpatialRelationError, match="relabeled"):
            transform_to_relation(ScaleTransform((2.0, 2.0)), "yx", "vu")

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

    def test_non_invertible_matrix_raises_conversion_error(self):
        with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
            transform_to_relation(ScaleTransform((0.0, 1.0)), "yx", "yx")
        with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
            transform_to_relation(AffineTransform(affine=((1.0, 2.0, 0.0), (2.0, 4.0, 0.0))), "yx", "yx")

    def test_transform_not_fitting_axes_raises_plain_value_error(self):
        for transform, source, target in (
            (ScaleTransform((2.0, 2.0)), "zyx", "zyx"),
            (ProjectAxisTransform(inserts=(0,)), "yx", "yx"),
        ):
            with pytest.raises(ValueError) as exc_info:
                transform_to_relation(transform, source, target)
            assert not isinstance(exc_info.value, CannotConvertToSpatialRelationError)
