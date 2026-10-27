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


def _multiscale_zyx() -> Multiscale:
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


class TestWithCoordinateSystem:
    def test_renamed_axis_carries_over_properties(self):
        ms = _multiscale_zyx().with_coordinate_system("renamed", reached_by=AxisRenaming(z="Z"))
        axes = ms.coordinate_system_ome_zarr_axes("renamed")
        assert list(axes) == ["Z", "y", "x"]
        assert axes["Z"] == Axis(name="Z", type="space", unit="um", long_name="Depth")
        assert axes["y"] == Axis(name="y", type="space", unit="um")

    def test_swapped_names_swap_properties(self):
        source = Multiscale.from_single(
            Scale(
                Shape(y=8, x=8),
                unit=Unit(y="um", x="um"),
                ome_zarr_axes={"y": Axis(type="space", long_name="rows"), "x": Axis(type="space", long_name="cols")},
            )
        )
        with_swapped = source.with_coordinate_system("swapped", reached_by=AxisRenaming(y="x", x="y"))
        axes = with_swapped.coordinate_system_ome_zarr_axes("swapped")
        assert list(axes) == ["x", "y"]
        assert axes["x"] == Axis(name="x", type="space", unit="um", long_name="rows")
        assert axes["y"] == Axis(name="y", type="space", unit="um", long_name="cols")

    def test_explicit_properties_override_carryover_and_fill_blanks_from_them(self):
        ms = _multiscale_zyx().with_coordinate_system(
            "renamed", reached_by=AxisRenaming(z="Z"), ome_zarr_axes={"Z": Axis(long_name="Tiefe")}
        )
        axes = ms.coordinate_system_ome_zarr_axes("renamed")
        assert axes["Z"] == Axis(name="Z", type="space", unit="um", long_name="Tiefe")

    def test_renamed_dropped_reinserted_axis_key_does_not_carry_over(self):
        relations = SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx")))
        with_system = _multiscale_zyx().with_coordinate_system("fresh", reached_by=relations)
        axes = with_system.coordinate_system_ome_zarr_axes("fresh")
        assert axes["z"].long_name is None, "should not carry over from _volume.ome_zarr_axes[z]"
        assert axes["z"] == Axis(name="z", type="space"), "new axis: type inferred from its key"
        assert axes["y"].unit == "um"

    def test_inserted_axis_next_to_renamed_infers_type(self):
        relations = SpatialRelationSequence((AxisRenaming(z="Z"), AxisRearrangementTo("cZyx")))
        axes = _multiscale_zyx().with_coordinate_system("n", reached_by=relations).coordinate_system_ome_zarr_axes("n")
        assert axes["c"] == Axis(name="c", type="channel")
        assert axes["Z"].unit == "um"

    def test_axis_depending_on_several_axes_keeps_properties_they_agree_on(self):
        # Shearing y by x: y' now depends on y and x. Both are space axes in um, so the sheared y' is the same.
        # (The long names differ, so the sheared y' has none.)
        source = Multiscale.from_single(
            Scale(
                Shape(y=8, x=8),
                unit=Unit(y="um", x="um"),
                ome_zarr_axes={"y": Axis(type="space", long_name="rows"), "x": Axis(type="space", long_name="cols")},
            )
        )
        shear = Affine.from_array(((1, 0.5, 0), (0, 1, 0)), "yx")
        axes = source.with_coordinate_system("sheared", reached_by=shear).coordinate_system_ome_zarr_axes("sheared")
        assert axes["y"].long_name is None, "should not carry over disagreeing property"
        assert axes["y"] == Axis(name="y", type="space", unit="um")
        assert axes["x"] == Axis(name="x", type="space", unit="um", long_name="cols")

    def test_renaming_does_not_change_original_instance(self):
        ms = _multiscale_zyx()
        with_renamed = ms.with_coordinate_system("renamed", reached_by=AxisRenaming(z="Z"))
        assert with_renamed.axes == ms.axes
        assert list(with_renamed["s0"].shape) == ["z", "y", "x"]
        assert with_renamed.coordinate_systems == ("renamed",)

    def test_renamed_coordinate_system_roundtrips_through_ome_zarr_0_6(self):
        ms = _multiscale_zyx().with_coordinate_system(
            "renamed", reached_by=[AxisRenaming(z="Z"), Translation(Z=1.0, y=2.0, x=3.0)]
        )
        assert ms.lowest_lossless_ome_zarr_version == "0.6"

        attrs = OmeZarrGroup.from_single(ms).to_attrs(version="0.6")
        system = next(s for s in attrs["ome"]["multiscales"][0]["coordinateSystems"] if s["name"] == "renamed")
        assert [a["name"] for a in system["axes"]] == ["Z", "y", "x"]
        assert system["axes"][0]["unit"] == "um"

        recovered_group = OmeZarrGroup.from_attrs(attrs, shape_source="singletons")
        assert len(recovered_group.multiscales) == 1

        recovered = recovered_group.multiscales[0]
        assert recovered.coordinate_systems == ("renamed",)
        assert list(recovered.coordinate_system_ome_zarr_axes("renamed")) == ["Z", "y", "x"]


class TestAsDerivedFrom:
    def test_as_derived_from_carries_properties_across_a_renaming(self):
        source = _multiscale_zyx()
        derived = Multiscale.from_single(
            Scale(
                Shape(Z=4, Y=8, X=8),
                ome_zarr_axes={"Z": Axis(type="space"), "Y": Axis(type="space"), "X": Axis(type="space")},
            )
        ).as_derived_from(source, by=AxisRenaming(z="Z", y="Y", x="X"))
        assert derived.unit == Unit(Z="um", Y="um", X="um")
        assert derived.ome_zarr_axes["Z"] == Axis(name="Z", type="space", unit="um", long_name="Depth")

    def test_as_derived_from_renamed_dropped_reinserted_axis_key_does_not_carry_over(self):
        derived = Multiscale.from_single(
            Scale(Shape(z=4, y=8, x=8), ome_zarr_axes={a: Axis(type="space") for a in "zyx"})
        )
        result = derived.as_derived_from(
            _multiscale_zyx(), by=SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx")))
        )
        assert result.unit == Unit(z="", y="um", x="um")


class TestErrors:
    def test_unit_override_without_affine_raises_through_renaming(self):
        with pytest.raises(
            ValueError, match=r"The units of 'z' and 'Z' differ.*, but no relation converts between them"
        ):
            _ = _multiscale_zyx().with_coordinate_system("renamed", reached_by=AxisRenaming(z="Z"), unit={"Z": "mm"})

    def test_incompatible_renaming_is_rejected(self):
        with pytest.raises(ValueError, match=r"Cannot rename axes.*not present in source axes"):
            _multiscale_zyx().with_coordinate_system("n", reached_by=AxisRenaming(c="C"))
        with pytest.raises(ValueError, match=r"Cannot rename axes.*not present in source axes"):
            Multiscale.from_single(Scale(Shape(Z=4, y=8, x=8))).as_derived_from(
                _multiscale_zyx(), by=AxisRenaming(c="C")
            )

    def test_as_derived_from_with_mismatching_axes_requires_renaming(self):
        derived = Multiscale.from_single(
            Scale(Shape(Z=4, y=8, x=8), ome_zarr_axes={a: Axis(type="space") for a in "Zyx"})
        )
        with pytest.raises(ValueError, match="Cannot transfer coordinate systems"):
            derived.as_derived_from(_multiscale_zyx())
        assert derived.as_derived_from(_multiscale_zyx(), by=AxisRenaming(z="Z")).unit["Z"] == "um"
