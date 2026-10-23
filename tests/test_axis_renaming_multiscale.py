import pytest

from clearscale import (
    Affine,
    AxisRearrangementTo,
    AxisRenaming,
    Multiscale,
    OmeZarrGroup,
    PixelSize,
    ProjectionTo,
    Scale,
    Shape,
    Translation,
    Unit,
    ome_zarr,
)
from clearscale._spatial_relations import SpatialRelationSequence

Axis = ome_zarr.Axis


def _volume() -> Multiscale:
    return Multiscale.from_single(
        Scale(
            Shape(z=4, y=8, x=8),
            PixelSize(z=2.0, y=1.0, x=1.0),
            Unit(z="um", y="um", x="um"),
            ome_zarr_axes={
                "z": Axis(type="space", long_name="Depth"),
                "y": Axis(type="space"),
                "x": Axis(type="space"),
            },
        )
    )


def test_properties_follow_a_renamed_axis_in_a_new_coordinate_system():
    ms = _volume().with_coordinate_system("renamed", reached_by=AxisRenaming(z="Z"))
    axes = ms.coordinate_system_ome_zarr_axes("renamed")
    assert list(axes) == ["Z", "y", "x"]
    assert axes["Z"] == Axis(name="Z", type="space", unit="um", long_name="Depth")
    assert axes["y"] == Axis(name="y", type="space", unit="um")


def test_properties_follow_swapped_names():
    source = Multiscale.from_single(
        Scale(
            Shape(y=8, x=8),
            unit=Unit(y="um", x="um"),
            ome_zarr_axes={"y": Axis(type="space", long_name="rows"), "x": Axis(type="space", long_name="cols")},
        )
    )
    swapped = source.with_coordinate_system("swapped", reached_by=AxisRenaming(y="x", x="y"))
    axes = swapped.coordinate_system_ome_zarr_axes("swapped")
    assert list(axes) == ["x", "y"]
    assert axes["x"] == Axis(name="x", type="space", unit="um", long_name="rows")
    assert axes["y"] == Axis(name="y", type="space", unit="um", long_name="cols")


def test_explicit_properties_override_carried_ones_and_fill_blanks_from_them():
    ms = _volume().with_coordinate_system(
        "renamed", reached_by=AxisRenaming(z="Z"), ome_zarr_axes={"Z": Axis(long_name="Tiefe")}
    )
    axes = ms.coordinate_system_ome_zarr_axes("renamed")
    assert axes["Z"] == Axis(name="Z", type="space", unit="um", long_name="Tiefe")


def test_explicit_unit_is_merged_over_carried_unit_only_for_axes_it_mentions():
    ms = _volume().with_coordinate_system("renamed", reached_by=AxisRenaming(z="Z"), unit={"Z": "mm"})
    axes = ms.coordinate_system_ome_zarr_axes("renamed")
    assert (axes["Z"].unit, axes["y"].unit) == ("mm", "um")


def test_a_dropped_and_reinserted_axis_does_not_inherit_properties_through_the_renaming():
    relations = SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx")))
    axes = _volume().with_coordinate_system("fresh", reached_by=relations).coordinate_system_ome_zarr_axes("fresh")
    assert axes["z"] == Axis(name="z", type="space"), "new axis: type inferred from its key, nothing carried"
    assert axes["y"].unit == "um"


def test_inserted_axis_next_to_renamed_axis_gets_inferred_type():
    relations = SpatialRelationSequence((AxisRenaming(z="Z"), AxisRearrangementTo("cZyx")))
    axes = _volume().with_coordinate_system("n", reached_by=relations).coordinate_system_ome_zarr_axes("n")
    assert axes["c"] == Axis(name="c", type="channel")
    assert axes["Z"].unit == "um"


def test_renaming_alone_does_not_change_the_multiscale_itself():
    ms = _volume()
    renamed = ms.with_coordinate_system("renamed", reached_by=AxisRenaming(z="Z"))
    assert renamed.axes == ms.axes
    assert list(renamed["s0"].shape) == ["z", "y", "x"]
    assert renamed.coordinate_systems == ("renamed",)


def test_renamed_coordinate_system_roundtrips_through_ome_zarr_0_6():
    ms = _volume().with_coordinate_system("renamed", reached_by=[AxisRenaming(z="Z"), Translation(Z=1.0, y=2.0, x=3.0)])
    attrs = OmeZarrGroup.from_single(ms).to_attrs(version="0.6")
    system = next(s for s in attrs["ome"]["multiscales"][0]["coordinateSystems"] if s["name"] == "renamed")
    assert [a["name"] for a in system["axes"]] == ["Z", "y", "x"]
    assert system["axes"][0]["unit"] == "um"
    assert ms.lowest_lossless_ome_zarr_version == "0.6"


def test_as_derived_from_carries_properties_across_a_renaming():
    source = _volume()
    derived = Multiscale.from_single(
        Scale(
            Shape(Z=4, Y=8, X=8),
            ome_zarr_axes={"Z": Axis(type="space"), "Y": Axis(type="space"), "X": Axis(type="space")},
        )
    ).as_derived_from(source, by=AxisRenaming(z="Z", y="Y", x="X"))
    assert derived.unit == Unit(Z="um", Y="um", X="um")
    assert derived.ome_zarr_axes["Z"] == Axis(name="Z", type="space", unit="um", long_name="Depth")


def test_as_derived_from_without_matching_axes_needs_the_renaming_to_say_so():
    derived = Multiscale.from_single(Scale(Shape(Z=4, y=8, x=8), ome_zarr_axes={a: Axis(type="space") for a in "Zyx"}))
    with pytest.raises(ValueError, match="Cannot transfer"):
        derived.as_derived_from(_volume())
    assert derived.as_derived_from(_volume(), by=AxisRenaming(z="Z")).unit["Z"] == "um"


def test_as_derived_from_does_not_carry_through_a_dropped_and_reinserted_axis():
    derived = Multiscale.from_single(Scale(Shape(z=4, y=8, x=8), ome_zarr_axes={a: Axis(type="space") for a in "zyx"}))
    result = derived.as_derived_from(_volume(), by=SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx"))))
    assert result.unit == Unit(z="", y="um", x="um")


def test_incompatible_renaming_is_rejected():
    with pytest.raises(ValueError):
        _volume().with_coordinate_system("n", reached_by=AxisRenaming(c="C"))


def test_axis_depending_on_several_axes_keeps_the_properties_they_agree_on():
    # Shearing y by x: y now depends on y and x. Both are space axes in um, so the sheared y still is one.
    # (The long names differ, so the sheared y has none.)
    source = Multiscale.from_single(
        Scale(
            Shape(y=8, x=8),
            unit=Unit(y="um", x="um"),
            ome_zarr_axes={"y": Axis(type="space", long_name="rows"), "x": Axis(type="space", long_name="cols")},
        )
    )
    shear = Affine.from_array(((1, 0.5, 0), (0, 1, 0)), "yx")
    axes = source.with_coordinate_system("sheared", reached_by=shear).coordinate_system_ome_zarr_axes("sheared")
    assert axes["y"] == Axis(name="y", type="space", unit="um")
    assert axes["x"] == Axis(name="x", type="space", unit="um", long_name="cols")


def test_axis_depending_on_axes_that_disagree_is_blank_where_they_do_not_agree():
    source = Multiscale.from_single(
        Scale(Shape(y=8, x=8), unit=Unit(y="um", x="mm"), ome_zarr_axes={a: Axis(type="space") for a in "yx"})
    )
    shear = Affine.from_array(((1, 0.5, 0), (0, 1, 0)), "yx")
    axes = source.with_coordinate_system("sheared", reached_by=shear).coordinate_system_ome_zarr_axes("sheared")
    assert axes["y"] == Axis(name="y", type="space"), "type agrees, units do not"
    assert axes["x"] == Axis(name="x", type="space", unit="mm")
