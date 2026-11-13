from typing import get_args

import pytest

from clearscale import BlueprintFactors, BlueprintShapes, Factor, Multiscale, PixelSize, Scale, Shape, Translation
from clearscale import characterization
from clearscale._services import translating as _translating
from clearscale.types import TranslatingMethodName

ALL_NAMES = ["half_pixel_space_preservation", "discrete_bin_center", "first_value_decimation"]


class TestRegistry:
    def test_names_match_the_literal_type(self):
        assert sorted(_translating.SHIFT_FUNCTIONS_BY_NAME) == sorted(get_args(TranslatingMethodName))

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_names_are_the_function_names(self, name):
        assert _translating.SHIFT_FUNCTIONS_BY_NAME[name].__name__ == name

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_characterization_reexports_the_same_function_objects(self, name):
        assert getattr(characterization, name) is _translating.SHIFT_FUNCTIONS_BY_NAME[name]

    def test_known_shift_functions_cover_all_names(self):
        assert [f.__name__ for f in characterization.known_shift_functions] == ALL_NAMES


class TestResolve:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_resolves_name_to_function(self, name):
        assert _translating.resolve_translating(name) is getattr(characterization, name)

    def test_callables_pass_through(self):
        def custom(base, target):
            raise NotImplementedError

        assert _translating.resolve_translating(custom) is custom

    def test_none_is_rejected_with_a_hint_for_the_old_no_shift_meaning(self):
        with pytest.raises(TypeError, match="Use translating='first_value_decimation' for no shift"):
            _translating.resolve_translating(None)  # type: ignore[arg-type]

    def test_unknown_name_lists_valid_names(self):
        with pytest.raises(ValueError, match="Unknown translating method 'half_pixel'.*half_pixel_space_preservation"):
            _translating.resolve_translating("half_pixel")  # type: ignore[arg-type]

    def test_rejects_other_types(self):
        with pytest.raises(TypeError, match="translating must be a function or one of"):
            _translating.resolve_translating(3)  # type: ignore[arg-type]


class TestAliasesInPublicMethods:
    shapes = BlueprintShapes({"s0": Shape(y=8, x=8), "s1": Shape(y=4, x=2)})
    base = Scale(shape=Shape(y=8, x=8), pixel_size=PixelSize(y=2.0, x=3.0), translation=Translation(y=10.0, x=-5.0))

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_shapes_apply_to_scale_matches_passing_the_function(self, name):
        by_name = self.shapes.apply_to_scale(self.base, pixel_sizing="shape_ratio", translating=name)
        by_function = self.shapes.apply_to_scale(
            self.base, pixel_sizing="shape_ratio", translating=getattr(characterization, name)
        )
        assert by_name == by_function

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_factors_apply_to_scale_matches_passing_the_function(self, name):
        factors = BlueprintFactors({"s0": Factor(y=1.0, x=1.0), "s1": Factor(y=2.0, x=4.0)})
        by_name = factors.apply_to_scale(self.base, rounding="ceil", pixel_sizing="shape_ratio", translating=name)
        by_function = factors.apply_to_scale(
            self.base, rounding="ceil", pixel_sizing="shape_ratio", translating=getattr(characterization, name)
        )
        assert by_name == by_function

    def test_factors_exact_factor_path_resolves_names(self):
        factors = BlueprintFactors({"s0": Factor(y=1.0, x=4.0)})
        base = Scale(shape=Shape(y=4, x=11), pixel_size=PixelSize(y=1.0, x=1.0), translation=Translation(y=0.0, x=0.0))
        ms = factors.apply_to_scale(
            base, rounding="ceil", pixel_sizing="exact_factor", translating="half_pixel_space_preservation"
        )
        # exact_factor pixel size along x = 4.0; half-pixel shift = 0.5 * (4.0 - 1.0) = 1.5. Along y: no scaling, no shift
        assert ms["s0"].translation == Translation(y=0.0, x=1.5)

    def test_unknown_name_is_rejected_even_without_scales_to_compute(self):
        # Validation happens at the entry point, not lazily per target scale
        single = BlueprintShapes({"s0": Shape(y=8, x=8)})
        with pytest.raises(ValueError, match="Unknown translating method"):
            single.apply_to_scale(self.base, pixel_sizing="shape_ratio", translating="nope")  # type: ignore[arg-type]

    def test_from_single_accepts_names(self):
        ms = Multiscale.from_single(
            self.base, blueprint=self.shapes, pixel_sizing="shape_ratio", translating="half_pixel_space_preservation"
        )
        assert ms["s1"].translation == Translation(y=11.0, x=-0.5)

    def test_derive_accepts_names(self):
        parent = Multiscale.from_single(self.base)
        derived = parent.derive(
            "s0", blueprint=self.shapes, pixel_sizing="shape_ratio", translating="half_pixel_space_preservation"
        )
        assert derived["s1"].translation == Translation(y=11.0, x=-0.5)


def _scale(pixel_size_items):
    """Shape should be irrelevant for all calculations.
    Pixel size will always be the precise source of information for what scaling was done."""
    ps = PixelSize(pixel_size_items)
    sh = Shape.all_singletons(ps)
    return Scale(shape=sh, pixel_size=ps)


class TestShiftFunctions:
    def test_half_pixel_space_preservation_computes_axis_wise_shift(self):
        base = _scale([("t", 12.0), ("z", 0.4), ("y", 0.6), ("x", 2.0)])
        target = _scale([("t", 12.0), ("z", 0.2), ("y", 1.5), ("x", 6.0)])

        shift = _translating.half_pixel_space_preservation(base, target)

        assert shift == Translation([("t", 0.0), ("z", -0.1), ("y", 0.45), ("x", 2.0)])

    def test_discrete_bin_center_computes_center_of_first_implicit_bin(self):
        base = _scale([("z", 0.4), ("y", 0.6), ("x", 2.0)])
        target = _scale([("z", 0.2), ("y", 1.5), ("x", 6.0)])

        shift = _translating.discrete_bin_center(base, target)

        assert shift == Translation([("z", 0.0), ("y", 0.6), ("x", 2.0)])

    @pytest.mark.parametrize(
        ("target_pixel_size", "expected_shift"),
        [
            (0.2, 0.0),
            (0.4, 0.0),
            (0.8, 0.2),
            (1.0, 0.4),
            (1.01, 0.4),
            (1.21, 0.6000000000000001),
        ],
    )
    def test_discrete_bin_center_uses_ceiled_implicit_bin_size(self, target_pixel_size, expected_shift):
        base = _scale([("x", 0.4)])
        target = _scale([("x", target_pixel_size)])

        shift = _translating.discrete_bin_center(base, target)

        assert shift == Translation(x=expected_shift)

    def test_first_value_decimation_returns_identity_translation(self):
        base = _scale([("cookies", 0.5), ("y", 1.0), ("x", 2.0)])
        target = _scale([("cookies", 0.0000212), ("y", 126.0), ("x", 4.0)])

        shift = _translating.first_value_decimation(base, target)

        assert shift == Translation([("cookies", 0.0), ("y", 0.0), ("x", 0.0)])

    @pytest.mark.parametrize("shift_function", _translating.known_shift_functions)
    @pytest.mark.parametrize(
        "target",
        [
            _scale([("x", 4.0), ("y", 2.0)]),
            _scale([("y", 2.0), ("z", 4.0)]),
        ],
    )
    def test_all_shift_functions_reject_axis_mismatches(self, shift_function, target):
        base = _scale([("y", 1.0), ("x", 2.0)])

        with pytest.raises(ValueError, match="Axis mismatch"):
            shift_function(base, target)
