"""A Multiscale is always valid regarding the axis requirements of multiscale formats (2-5 axes, 2-3 of type space),
no matter how it was constructed."""

import pytest

from clearscale import AxisRearrangementTo, GroupKind, Multiscale, OmeZarrGroup, Scale, Shape, ome_zarr

SPACE = ome_zarr.Axis(type="space")


def _typed_yx() -> Multiscale:
    return Multiscale.from_single(Scale(Shape(y=4, x=4), ome_zarr_axes={"y": SPACE, "x": SPACE}))


def _0_4_metadata(axes):
    return {
        "version": "0.4",
        "axes": axes,
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}]}],
    }


def _0_6_metadata(system_axes):
    return {
        "version": "0.6",
        "coordinateSystems": [
            {"name": "physical", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
            {"name": "other", "axes": system_axes},
        ],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0], "input": {"path": "s0"}, "output": {"name": "physical"}}
                ],
            }
        ],
        "coordinateTransformations": [{"type": "identity", "input": {"name": "physical"}, "output": {"name": "other"}}],
    }


class TestConstruction:
    @pytest.mark.parametrize("axes", ["x", "tczyxa"])
    def test_rejects_wrong_ndim(self, axes):
        scale = Scale.from_lists(axes, ome_zarr_axes=[SPACE for _ in axes])
        with pytest.raises(ValueError, match="2 to 5 axes"):
            Multiscale.from_single(scale)

    def test_rejects_one_space_axis(self):
        axes = [ome_zarr.Axis(type="channel"), ome_zarr.Axis(type="channel"), SPACE]
        with pytest.raises(ValueError, match="has 1"):
            Multiscale.from_single(Scale.from_lists("cdx", ome_zarr_axes=axes))

    def test_rejects_four_space_axes(self):
        with pytest.raises(ValueError, match="has 4"):
            Multiscale.from_single(Scale.from_lists("wzyx", ome_zarr_axes=[SPACE] * 4))

    @pytest.mark.parametrize("axes", ["yx", "zyx", "czyx", "tczyx"])
    def test_accepts_inferred_standard_axes(self, axes):
        Multiscale.from_single(Scale.from_lists(axes))


class TestReading:
    @pytest.mark.parametrize(
        "axes",
        [
            [{"name": "y"}, {"name": "x"}],
            [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}, {"name": "x", "type": "channel"}],
            [{"name": "x", "type": "space"}],
        ],
        ids=["untyped", "one-space-axis", "one-axis"],
    )
    def test_from_ome_zarr_rejects_metadata_that_violates_the_spec(self, axes):
        with pytest.raises(ValueError, match="require"):
            Multiscale.from_ome_zarr(_0_4_metadata(axes), shape_source="singletons")

    def test_from_ome_zarr_rejects_invalid_additional_coordinate_system(self):
        system = [{"name": "y", "type": "space"}, {"name": "c", "type": "channel"}]
        with pytest.raises(ValueError, match="coordinate system 'other'"):
            Multiscale.from_ome_zarr(_0_6_metadata(system), shape_source="singletons")

    def test_from_ome_zarr_accepts_valid_additional_coordinate_system(self):
        system = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
        ms = Multiscale.from_ome_zarr(_0_6_metadata(system), shape_source="singletons")
        assert ms.coordinate_systems == ("other",)

    def test_group_reports_invalid_multiscale_as_invalid_with_reason(self):
        group = OmeZarrGroup.from_attrs(
            {"multiscales": [_0_4_metadata([{"name": "y"}, {"name": "x"}])]}, shape_source="singletons"
        )
        assert group.kind is GroupKind.INVALID and group.multiscales == ()
        assert group.version == "0.4"
        assert "'space'" in group.invalid_objects[0].error


class TestCoordinateSystems:
    def test_with_coordinate_system_rejects_too_few_axes(self):
        with pytest.raises(ValueError, match="2 to 5 axes, but coordinate system 'w'"):
            _typed_yx().with_coordinate_system("w", reached_by=AxisRearrangementTo("x"))

    def test_with_coordinate_system_rejects_too_many_axes(self):
        with pytest.raises(ValueError, match="2 to 5 axes"):
            _typed_yx().with_coordinate_system("w", reached_by=AxisRearrangementTo("tczyxa"))

    def test_with_coordinate_system_rejects_too_few_space_axes(self):
        with pytest.raises(ValueError, match="'space', but coordinate system 'w' has 1"):
            _typed_yx().with_coordinate_system("w", reached_by=AxisRearrangementTo("yc"))

    def test_with_coordinate_system_rejects_untyped_axes_taking_the_place_of_space_axes(self):
        with pytest.raises(ValueError, match="Axes without a type"):
            _typed_yx().with_coordinate_system("w", reached_by=AxisRearrangementTo("ix"))

    def test_with_coordinate_system_accepts_untyped_extra_axes_alongside_two_space_axes(self):
        ms = _typed_yx().with_coordinate_system("w", reached_by=AxisRearrangementTo("cyx"))
        assert ms.coordinate_systems == ("w",)


@pytest.mark.parametrize("key", ["", "/s0", "../s0", "s 0", "a//b"])
def test_invalid_scale_keys_are_rejected_on_every_path(key):
    scale = Scale.from_lists("yx")
    with pytest.raises(ValueError, match="not a valid relative path"):
        Multiscale.from_single(scale, scale_key=key)
    with pytest.raises(ValueError, match="not a valid relative path"):
        Multiscale.from_ome_zarr(
            {
                "version": "0.4",
                "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
                "datasets": (
                    [{"path": key or "x", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]}]
                    if key
                    else [{"path": "../x", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]}]
                ),
            },
            shape_source="singletons",
        )
