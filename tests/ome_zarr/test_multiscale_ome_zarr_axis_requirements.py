"""
Tests OME-Zarr requirements for axes:
Spec (identical for 0.4, 0.5 and 0.6):
    The length of “axes” must be between 2 and 5 (...).
    The “axes” MUST contain 2 or 3 entries of “type:space”.
"""

from typing import List

import pytest

from clearscale import AxisRearrangementTo, Multiscale, OmeZarrGroup, Scale, Shape, ome_zarr

SPACE = ome_zarr.Axis(type="space")


def _untyped(axes="zyx") -> Multiscale:
    return Multiscale.from_single(Scale.from_lists(axes))


def _typed_yx() -> Multiscale:
    return Multiscale.from_single(Scale(Shape(y=4, x=4), ome_zarr_axes={"y": SPACE, "x": SPACE}))


def _axis_types(written, version):
    axes = written["axes"] if version != "0.6" else written["coordinateSystems"][0]["axes"]
    return [a.get("type") for a in axes]


@pytest.mark.parametrize("version", ome_zarr.SUPPORTED_OME_ZARR_VERSIONS_WRITE)
class TestToOmeZarr:
    """
    The general principle is "the public API should not enable creation of states that cannot
    be serialized to any format" (and so far OME-Zarr is the only format we serialize to).
    But
    1. OME-Zarr's axis concept isn't defined beyond "there are 'time', 'channel' and 'space' axes".
    2. Many consumers probably don't work with axis "types" on their side
    3. There are use-cases for Multiscale objects that never require serialization
    4. We can still make useful Multiscale objects in from_ome_zarr even if the input attrs are
       technically invalid (e.g. due to not defining axis types)
    Hence the exception from the general principle for axis types, and write-time enforcement instead.
    As an escape hatch for invalid Multiscales, to_ome_zarr takes the missing_axis_types param to fix them up.
    """

    def test_rejects_untyped_axes_naming_the_fix(self, version):
        with pytest.raises(ValueError, match="type 'space'.*ome_zarr_axes='infer'"):
            _untyped().to_ome_zarr(version=version)

    @pytest.mark.parametrize("axes", ["x", "tczyxa"])
    def test_rejects_wrong_ndim(self, version, axes):
        # TODO: ndim requirements should actually be enforced already during Multiscale construction.
        # There is no way to fix a 1D or 5D+ Multiscale by the time to_ome_zarr is called, so their
        # existence should be prevented.
        ms = Multiscale.from_single(Scale.from_lists(axes, ome_zarr_axes=[SPACE for _ in axes]))
        with pytest.raises(ValueError, match="2 to 5 axes"):
            ms.to_ome_zarr(version=version)

    @pytest.mark.parametrize("n_space", [0, 1, 4, 5])
    def test_rejects_one_space_axis(self, version, n_space):
        ome_axes: List[ome_zarr.Axis] = ([SPACE] * n_space) + ([ome_zarr.Axis()] * (5 - n_space))
        ms = Multiscale.from_single(Scale.from_lists("abcde", ome_zarr_axes=ome_axes))
        with pytest.raises(ValueError, match=f"has {n_space}"):
            ms.to_ome_zarr(version=version)

    def test_accepts_inferred_types(self, version):
        written = Multiscale.from_single(Scale.from_lists("zyx", ome_zarr_axes="infer")).to_ome_zarr(version=version)
        assert _axis_types(written, version) == ["space"] * 3


@pytest.mark.parametrize("version", ome_zarr.SUPPORTED_OME_ZARR_VERSIONS_WRITE)
class TestMissingAxisTypes:
    """
    We allow construction of untyped Multiscales because we expect intermediate states might
    need to be able to exist without valid typing.
    And because the type concept is too specific to OME-Zarr, and underspecified.
    The missing_axis_types kwarg is the escape hatch for fixing untyped Multiscales at write time.
    """

    def test_infer(self, version):
        written = _untyped("czyx").to_ome_zarr(version=version, missing_axis_types="infer")
        assert _axis_types(written, version) == ["channel", "space", "space", "space"]

    def test_mapping(self, version):
        types = {"a": "space", "b": "space"}
        written = _untyped("ab").to_ome_zarr(version=version, missing_axis_types=types)
        assert _axis_types(written, version) == ["space", "space"]

    def test_ignores_inapplicable_entries(self, version):
        types = {"some_other_key": "cookies", "x": "space", "y": "space"}
        written = _untyped("xy").to_ome_zarr(version=version, missing_axis_types=types)
        assert _axis_types(written, version) == ["space", "space"]

    def test_does_not_override_existing(self, version):
        # z would be inferred as type="space" if not already defined
        ms = Multiscale.from_single(
            Scale.from_lists("zyx", ome_zarr_axes=[ome_zarr.Axis(type="custom"), ome_zarr.Axis(), ome_zarr.Axis()])
        )
        written = ms.to_ome_zarr(version=version, missing_axis_types="infer")
        assert _axis_types(written, version) == ["custom", "space", "space"]

    def test_does_not_modify_the_multiscale(self, version):
        ms = _untyped()
        ms.to_ome_zarr(version=version, missing_axis_types="infer")
        assert all(ax.type is None for ax in ms.ome_zarr_axes.values())

    def test_revalidates_type_order_after_fill(self, version):
        with pytest.raises(ValueError, match="time-channel-others"):
            _untyped("zyxc").to_ome_zarr(version=version, missing_axis_types="infer")

    def test_rejects_bad_parameter(self, version):
        with pytest.raises(ValueError, match="must be 'infer' or"):
            _untyped().to_ome_zarr(version=version, missing_axis_types="guess")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty strings"):
            _untyped().to_ome_zarr(version=version, missing_axis_types={"z": ""})

    def test_ome_zarr_group_forwards(self, version):
        attrs = OmeZarrGroup.from_single(_untyped()).to_attrs(version=version, missing_axis_types="infer")
        assert attrs
        with pytest.raises(ValueError, match="type 'space'"):
            OmeZarrGroup.from_single(_untyped()).to_attrs(version=version)


class TestValidateForOmeZarr:
    def test_passes_valid_and_returns_none(self):
        assert _typed_yx().validate_for_ome_zarr() is None

    def test_raises_like_to_ome_zarr(self):
        with pytest.raises(ValueError, match="type 'space'"):
            _untyped().validate_for_ome_zarr()

    def test_mirrors_missing_axis_types(self):
        _untyped().validate_for_ome_zarr(missing_axis_types="infer")
        with pytest.raises(ValueError, match="time-channel-others"):
            _untyped("zyxc").validate_for_ome_zarr(missing_axis_types="infer")

    def test_rejects_invalid_scale_keys(self):
        ms = Multiscale.from_single(Scale.from_lists("yx", ome_zarr_axes="infer"), scale_key="../bad")
        with pytest.raises(ValueError, match="relative filesystem path"):
            ms.validate_for_ome_zarr()


class TestWithCoordinateSystem:
    def test_rejects_too_few_axes(self):
        with pytest.raises(ValueError, match="2 to 5 axes"):
            _typed_yx().with_coordinate_system("other", reached_by=AxisRearrangementTo("x"))

    def test_rejects_too_many_axes(self):
        with pytest.raises(ValueError, match="2 to 5 axes"):
            _typed_yx().with_coordinate_system("other", reached_by=AxisRearrangementTo("tczyxa"))

    def test_rejects_when_typed_non_space_axes_leave_too_few(self):
        ms = Multiscale.from_single(Scale.from_lists("cyx", ome_zarr_axes="infer"))
        with pytest.raises(ValueError, match="cannot have 2 or 3 axes of type 'space'"):
            ms.with_coordinate_system("w", reached_by=AxisRearrangementTo("cx"))

    def test_allows_untyped_axes_whose_types_can_still_be_assigned(self):
        ms = _untyped("zyx").with_coordinate_system("other", reached_by=AxisRearrangementTo("yx"))
        assert ms.coordinate_systems == ("other",)


class TestCoordinateSystemsToOmeZarr:
    @staticmethod
    def _with_invalid_satellite() -> Multiscale:
        # Passes the construction check (an untyped axis could still become 'space'), but is invalid once written
        return _typed_yx().with_coordinate_system("other", reached_by=AxisRearrangementTo("yc"))

    def test_validation_checks_all_systems(self):
        with pytest.raises(ValueError, match="coordinate system 'other'"):
            self._with_invalid_satellite().validate_for_ome_zarr()

    def test_write_validates_satellites_only_for_relevant_versions(self):
        ms = self._with_invalid_satellite()
        with pytest.raises(ValueError, match="coordinate system 'other'"):
            ms.to_ome_zarr(version="0.6")
        for version in ("0.4", "0.5"):
            # 0.4 and 0.5 can't express satellites.
            # Invalid satellites are fine here, because they're not serialised anyway.
            assert ms.to_ome_zarr(version=version)

    def test_missing_axis_types_fill_satellite_axes(self):
        ms = _typed_yx().with_coordinate_system("satellite", reached_by=AxisRearrangementTo("cyx"))
        written = ms.to_ome_zarr(version="0.6", missing_axis_types="infer")

        satellite_types = [
            ax.get("type") for sys in written["coordinateSystems"] if sys["name"] == "satellite" for ax in sys["axes"]
        ]
        assert satellite_types == ["channel", "space", "space"]
