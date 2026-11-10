"""Ensure axis property validation is consistent across the public SpatialRelation API.
Every method that uses `require_consistent_axis_properties` should be included in these tests,
and every (public) method that accepts SpatialRelations should use the validator function."""

import pytest

from clearscale import Affine, Multiscale, Scale, Scene, Shape, Unit

MICROMETER = Unit(y="micrometer", x="micrometer")
AFFINE_SCALING_X = Affine.from_factor({"y": 1, "x": 1000})  # y is kept as it is, x is scaled


def _ms_yx(unit: Unit) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(y=4, x=4), unit=unit)})


def _ms_with_coordinate_system(source_unit: Unit, target_unit: Unit, relation=AFFINE_SCALING_X):
    _ms_yx(source_unit).with_coordinate_system("target", reached_by=relation, unit=target_unit)


def _ms_as_derived_from(source_unit: Unit, target_unit: Unit, relation=AFFINE_SCALING_X):
    _ms_yx(source_unit).as_derived_from(_ms_yx(target_unit), reached_by=relation)


def _scene_from_multiscale_pair(source_unit: Unit, target_unit: Unit, relation=AFFINE_SCALING_X):
    Scene.from_multiscale_pair(a=_ms_yx(source_unit), b=_ms_yx(target_unit), a_to_b=relation, a_path="a", b_path="b")


def _scene_from_shared_coordinate_system(source_unit: Unit, target_unit: Unit, relation=AFFINE_SCALING_X):
    Scene.from_shared_coordinate_system(
        coordinate_system="target", connections=[(_ms_yx(source_unit), relation, "a")], unit=target_unit
    )


ENTRY_POINTS = [
    _ms_with_coordinate_system,
    _ms_as_derived_from,
    _scene_from_multiscale_pair,
    _scene_from_shared_coordinate_system,
]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
class TestUnitsAcrossScalingAffine:
    def test_accepts_the_scaled_axis_in_a_different_unit(self, entry_point):
        entry_point(MICROMETER, Unit(y="micrometer", x="nanometer"))

    def test_accepts_unspecified_units(self, entry_point):
        entry_point(MICROMETER, Unit(y="", x=""))
        entry_point(Unit(y="", x=""), Unit(y="nanometer", x="nanometer"))

    def test_rejects_a_different_unit_for_the_axis_that_is_kept_as_it_is(self, entry_point):
        with pytest.raises(ValueError, match=r"differ \('y': 'micrometer' vs 'nanometer'\)"):
            entry_point(MICROMETER, Unit(y="nanometer", x="nanometer"))


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
class TestUnitConversionFactor:
    NANO = Unit(y="nanometer", x="nanometer")

    def test_accepts_the_conversion_from_unit_conversion(self, entry_point):
        entry_point(MICROMETER, self.NANO, Affine.from_unit_conversion(MICROMETER, self.NANO))

    def test_accepts_other_scales_for_the_same_unit(self, entry_point):
        entry_point(MICROMETER, MICROMETER, Affine.from_factor({"y": 2, "x": 0.5}))

    def test_rejects_a_wrong_factor_naming_the_axis(self, entry_point):
        with pytest.raises(
            ValueError, match=r"differ \('x': 'micrometer' vs 'nanometer' \(scaled by 100, but .* 1000\)\)"
        ):
            entry_point(MICROMETER, self.NANO, Affine.from_array(((1000, 0, 0), (0, 100, 0)), "yx"))
