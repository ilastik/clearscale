import math

import pytest

from clearscale import (
    Affine,
    AxisRearrangementTo,
    AxisRenaming,
    CannotConvertToSpatialRelationError,
    Factor,
    Multiscale,
    NoSuchCoordinateSystemError,
    PermutationTo,
    PixelSize,
    ProjectionTo,
    Scale,
    Shape,
    Translation,
    Unit,
    ome_zarr,
)
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale._transforms import CoordinateSystem, DisplacementsTransform, RotationTransform, TransformGraph

Axis = ome_zarr.Axis


def scaled(**factors: float) -> Affine:
    return Affine.from_factor(Factor(factors))


# Values are exactly representable, so that factors survive 1 / (1 / f) and round trips compare equal


def _volume() -> Multiscale:
    return Multiscale.from_single(
        Scale(Shape(z=4, y=8, x=8), PixelSize(z=2.0, y=1.0, x=1.0), Unit(z="um", y="um", x="um"))
    )


def _plane(axes: str = "yx") -> Multiscale:
    return Multiscale.from_single(
        Scale(Shape({a: 8 for a in axes}), ome_zarr_axes={a: Axis(type="space") for a in axes})
    )


RELATIONS_ON_ZYX = [
    scaled(z=1, y=2, x=4),
    Translation(z=0.0, y=1.5, x=-2.0),
    ProjectionTo("cyx"),
    ProjectionTo("yx"),
    PermutationTo("xyz"),
    AxisRearrangementTo("cxy"),
    AxisRenaming(z="Z"),
    AxisRenaming(y="x", x="y"),
    SpatialRelationSequence((scaled(z=1, y=2, x=2), Translation(z=0.0, y=1.0, x=1.0), ProjectionTo("cyx"))),
    SpatialRelationSequence((AxisRenaming(z="Z"), PermutationTo("yxZ"), scaled(y=2, x=2, Z=1))),
]


@pytest.mark.parametrize("relation", RELATIONS_ON_ZYX, ids=repr)
def test_relation_to_coordinate_system_inverts_with_coordinate_system(relation):
    original = _volume().with_coordinate_system("other", reached_by=relation)
    retrieved = original.relation_to_coordinate_system("other")
    assert retrieved is not None
    assert retrieved.target_axes(original.axes) == tuple(original.coordinate_system_ome_zarr_axes("other"))
    rebuilt = _volume().with_coordinate_system(
        "other", reached_by=retrieved, ome_zarr_axes=original.coordinate_system_ome_zarr_axes("other")
    )
    assert rebuilt == original


def test_relation_to_coordinate_system_is_none_for_the_same_space_under_another_name():
    ms = _volume().with_coordinate_system("same")
    assert ms.relation_to_coordinate_system("same") is None
    assert ms.relation_from_coordinate_system("same") is None


def test_with_coordinate_system_reached_by_a_chain_comes_back_as_canonical_chain():
    relation = [Translation(z=0.0, y=1.0, x=1.0), scaled(z=1, y=2, x=2)]
    ms = _volume().with_coordinate_system("other", reached_by=relation)
    # Composed to scale + translation; relations are re-expressed in the order scale, translation
    assert ms.relation_to_coordinate_system("other") == SpatialRelationSequence(
        (scaled(z=1.0, y=2.0, x=2.0), Translation(z=0.0, y=0.5, x=0.5))
    )


@pytest.mark.parametrize(
    "relation, derived",
    [
        (scaled(z=1, y=2, x=2), None),
        (Translation(z=0.0, y=4.0, x=8.0), None),
        (ProjectionTo("yx"), "yx"),
        (ProjectionTo("cyx"), "cyx"),
        (AxisRearrangementTo("xy"), "xy"),
        (PermutationTo("xyz"), None),
        (SpatialRelationSequence((scaled(z=1, y=2, x=2), ProjectionTo("yx"))), "yx"),
        (AxisRenaming(z="Z", y="Y", x="X"), "ZYX"),
    ],
    ids=repr,
)
def test_relation_from_coordinate_system_inverts_as_derived_from(relation, derived):
    volume = _volume()
    axes = derived or "".join(relation.target_axes(volume.axes))
    result = _cube(axes)
    derived_ms = result.as_derived_from(volume, by=relation)
    (source_name,) = derived_ms.coordinate_systems
    retrieved = derived_ms.relation_from_coordinate_system(source_name)
    assert retrieved is not None
    assert retrieved.target_axes(volume.axes) == derived_ms.axes
    rebuilt = result.as_derived_from(volume, by=retrieved)
    assert rebuilt == derived_ms


def _cube(axes: str) -> Multiscale:
    return Multiscale.from_single(
        Scale(Shape({a: 4 for a in axes}), ome_zarr_axes={a: Axis(type="space") for a in axes})
    )


def test_derivation_that_drops_axes_has_a_relation_from_but_none_to_the_source():
    volume = _volume()
    derived = _plane("yx").as_derived_from(volume, by=ProjectionTo("yx"))
    (name,) = derived.coordinate_systems
    assert derived.relation_from_coordinate_system(name) == ProjectionTo("yx")
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
        derived.relation_to_coordinate_system(name)


def test_invertible_derivation_has_the_inverse_relation_in_the_other_direction():
    volume = _volume()
    derived = _cube("zyx").as_derived_from(volume, by=scaled(z=1, y=2, x=2))
    (name,) = derived.coordinate_systems
    assert derived.relation_from_coordinate_system(name) == scaled(z=1.0, y=2.0, x=2.0)
    assert derived.relation_to_coordinate_system(name) == scaled(z=1.0, y=0.5, x=0.5)


def test_with_coordinate_system_is_retrievable_in_both_directions_when_invertible():
    ms = _volume().with_coordinate_system("other", reached_by=scaled(z=1, y=2, x=2))
    assert ms.relation_to_coordinate_system("other") == scaled(z=1.0, y=2.0, x=2.0)
    assert ms.relation_from_coordinate_system("other") == scaled(z=1.0, y=0.5, x=0.5)


def test_with_coordinate_system_that_drops_axes_only_has_the_to_direction():
    ms = _volume().with_coordinate_system("plane", reached_by=ProjectionTo("yx"))
    assert ms.relation_to_coordinate_system("plane") == ProjectionTo("yx")
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
        ms.relation_from_coordinate_system("plane")


@pytest.mark.parametrize("retrieve", ["relation_to_coordinate_system", "relation_from_coordinate_system"])
def test_unknown_names_and_the_multiscales_own_system_raise(retrieve):
    ms = _volume().with_coordinate_system("other")
    own_name = ms._intrinsic_ref.name
    for name in ("nope", own_name):
        with pytest.raises(NoSuchCoordinateSystemError):
            getattr(ms, retrieve)(name)
    assert issubclass(NoSuchCoordinateSystemError, ValueError)
    with pytest.raises(NoSuchCoordinateSystemError):
        getattr(_volume(), retrieve)("other")


def _with_connection(ms: Multiscale, name: str, transform) -> Multiscale:
    """Multiscale with `transform` (not necessarily created by a relation) from its own space to a new system"""
    system = CoordinateSystem(ms.ome_zarr_axes)._as_ref(name)
    graph = TransformGraph((transform.bound(ms._intrinsic_ref, system),), system_refs=ms._transform_graph.system_refs)
    return Multiscale(ms.items(), _transform_graph=graph, _intrinsic_ref=ms._intrinsic_ref)


def test_rotations_from_other_tools_are_retrievable_as_affine_relations():
    c, s = math.cos(0.3), math.sin(0.3)
    rotated = _with_connection(_plane("yx"), "rotated", RotationTransform(rotation=((c, -s), (s, c))))
    to_system = rotated.relation_to_coordinate_system("rotated")
    from_system = rotated.relation_from_coordinate_system("rotated")
    assert isinstance(to_system, Affine) and isinstance(from_system, Affine)
    # The relation is the inverse of the coordinate transformation, and the other direction is its inverse
    assert [v for row in to_system.to_tuples() for v in row] == pytest.approx([c, s, 0.0, -s, c, 0.0])
    assert [v for row in from_system.to_tuples() for v in row] == pytest.approx([c, -s, 0.0, s, c, 0.0])


def test_connections_without_relation_equivalent_raise_conversion_error():
    displaced = _with_connection(_plane("yx"), "displaced", DisplacementsTransform(path="field"))
    with pytest.raises(CannotConvertToSpatialRelationError, match="DisplacementsTransform"):
        displaced.relation_to_coordinate_system("displaced")
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
        displaced.relation_from_coordinate_system("displaced")


def test_legacy_global_transforms_are_retrievable_through_their_synthetic_system():
    ms = Multiscale.from_ome_zarr(
        {
            "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
            "coordinateTransformations": [
                {"type": "scale", "scale": [2.0, 2.0]},
                {"type": "translation", "translation": [4.0, 8.0]},
            ],
        },
        shape_source="singletons",
    )
    (name,) = ms.coordinate_systems
    assert ms.relation_to_coordinate_system(name) == SpatialRelationSequence(
        (scaled(y=0.5, x=0.5), Translation(y=-4.0, x=-8.0))
    )


def test_system_carried_over_a_dropping_derivation_is_retrievable_from_but_not_to():
    # The example in docs/space.md
    source = Multiscale.from_single(
        Scale(Shape(z=40, y=512, x=512), PixelSize(z=2.0, y=0.5, x=0.5), Unit(z="um", y="um", x="um"))
    ).with_coordinate_system("stage", reached_by=Translation(z=100.0, y=250.0, x=250.0).inverted())
    projection = Multiscale.from_single(
        Scale(Shape(y=512, x=512), PixelSize(y=0.5, x=0.5), Unit(y="um", x="um")), scale_key="mip"
    )
    result = projection.as_derived_from(source, by=ProjectionTo("yx"))
    assert "stage" in result.coordinate_systems
    assert result.relation_from_coordinate_system("stage") == SpatialRelationSequence(
        (ProjectionTo("yx"), Translation(y=250.0, x=250.0))
    )
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be inverted"):
        result.relation_to_coordinate_system("stage")


def test_system_carried_over_an_invertible_derivation_is_retrievable_in_both_directions():
    source = _volume().with_coordinate_system("stage", reached_by=Translation(z=100.0, y=250.0, x=250.0).inverted())
    derived = _cube("zyx").as_derived_from(source, by=scaled(z=1, y=2, x=2))
    to_stage = derived.relation_to_coordinate_system("stage")
    from_stage = derived.relation_from_coordinate_system("stage")
    assert to_stage == SpatialRelationSequence((scaled(z=1.0, y=0.5, x=0.5), Translation(z=-100.0, y=-250.0, x=-250.0)))
    assert from_stage == SpatialRelationSequence((scaled(z=1.0, y=2.0, x=2.0), Translation(z=100.0, y=125.0, x=125.0)))
