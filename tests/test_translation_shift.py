import math

import pytest

from clearscale import (
    PixelSize,
    Scale,
    Shape,
    Translation,
    ScalingMethodCharacterization,
    characterize_shape_scaling_method,
    characterize_shape_factor_scaling_method,
    characterize_step_factor_scaling_method,
    discrete_bin_center,
    first_value_decimation,
    half_pixel_space_preservation,
)
from clearscale._translation_shift import known_shift_functions


def _scale(pixel_size_items):
    """Shape should be irrelevant for all calculations.
    Pixel size will always be the precise source of information for what scaling was done."""
    ps = PixelSize(pixel_size_items)
    sh = Shape.all_singletons(ps)
    return Scale(shape=sh, pixel_size=ps)


def _linear_scale_values(*, first_coordinate, spacing, length):
    return [first_coordinate + spacing * index for index in range(length)]


def _linear_scale_values_starting_at(*, first_coordinate, target_length, source_length):
    """Mock scaling implementation with an artificial offset. Spacing == source/target,
    i.e. exactly the shape_ratio convention."""
    return _linear_scale_values(
        first_coordinate=first_coordinate, spacing=source_length / target_length, length=target_length
    )


class TestShiftFunctions:
    def test_half_pixel_space_preservation_computes_axis_wise_shift(self):
        base = _scale([("t", 12.0), ("z", 0.4), ("y", 0.6), ("x", 2.0)])
        target = _scale([("t", 12.0), ("z", 0.2), ("y", 1.5), ("x", 6.0)])

        shift = half_pixel_space_preservation(base, target)

        assert shift == Translation([("t", 0.0), ("z", -0.1), ("y", 0.45), ("x", 2.0)])

    def test_discrete_bin_center_computes_center_of_first_implicit_bin(self):
        base = _scale([("z", 0.4), ("y", 0.6), ("x", 2.0)])
        target = _scale([("z", 0.2), ("y", 1.5), ("x", 6.0)])

        shift = discrete_bin_center(base, target)

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

        shift = discrete_bin_center(base, target)

        assert shift == Translation(x=expected_shift)

    def test_first_value_decimation_returns_identity_translation(self):
        base = _scale([("cookies", 0.5), ("y", 1.0), ("x", 2.0)])
        target = _scale([("cookies", 0.0000212), ("y", 126.0), ("x", 4.0)])

        shift = first_value_decimation(base, target)

        assert shift == Translation([("cookies", 0.0), ("y", 0.0), ("x", 0.0)])

    @pytest.mark.parametrize("shift_function", known_shift_functions)
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


class TestCharacterizeShapeScaler:
    @pytest.mark.parametrize(
        ("expected_function", "first_coordinate_of"),
        [
            (half_pixel_space_preservation, lambda src, tgt: 0.5 * (src / tgt - 1.0)),
            (discrete_bin_center, lambda src, tgt: 0.5 * (math.ceil(src / tgt) - 1)),
            (first_value_decimation, lambda src, tgt: 0.0),
        ],
    )
    def test_matches_known_translating_conventions(self, expected_function, first_coordinate_of):
        def scaling_function(source, target_length):
            return _linear_scale_values_starting_at(
                first_coordinate=first_coordinate_of(len(source), target_length),
                target_length=target_length,
                source_length=len(source),
            )

        characterization = characterize_shape_scaling_method(scaling_function)

        assert characterization.translating is expected_function
        assert characterization.translating_error < 1e-9
        assert characterization.pixel_sizing == "shape_ratio"
        assert characterization.pixel_sizing_error < 1e-9
        assert characterization.rounding is None
        assert characterization.rounding_error is None

    def test_is_resistant_to_edge_errors(self):
        def scaling_function(source, target_length):
            source_length = len(source)
            first_coordinate = 0.5 * (source_length / target_length - 1.0)
            values = _linear_scale_values_starting_at(
                first_coordinate=first_coordinate, target_length=target_length, source_length=source_length
            )
            margin = target_length // 4
            for index in range(margin):
                values[index] = -1000.0
            for index in range(target_length - margin, target_length):
                values[index] = 1000.0
            return values

        characterization = characterize_shape_scaling_method(scaling_function)

        assert characterization.translating is half_pixel_space_preservation
        assert characterization.translating_error < 1e-9

    def test_reports_error_from_closest_known_convention(self):
        # characterize_shape_scaling_method uses input-len 1025, target-len 257
        # shape ratio = 3.9883268482490272373540856031128
        # predictions:
        #   half-pixel: shape ratio - 1 * 0.5 = 1.4941634241245136186770428015564
        #   bin-center: (ceil(shape ratio) - 1) * 0.5 = 1.5
        #   decimation: 0
        # Error tolerance between half-pixel and bin-center = (1.5 - 1.49416...) / 2 = 0.00291...
        artificial_error = 0.0025

        def scaling_function(source, target_length):
            source_length = len(source)
            correct_first_coordinate = 0.5 * (source_length / target_length - 1.0)
            return _linear_scale_values_starting_at(
                first_coordinate=correct_first_coordinate + artificial_error,
                target_length=target_length,
                source_length=source_length,
            )

        characterization = characterize_shape_scaling_method(scaling_function)

        assert characterization.translating is half_pixel_space_preservation
        assert abs(characterization.translating_error - artificial_error) < 1e-9

    @pytest.mark.parametrize(
        ("scaling_function", "expected_error"),
        [
            (lambda source, target_length: [[0.0] * 200], "must return a one-dimensional array"),
            (
                lambda source, target_length: [float(index) for index in range(len(source))],
                "did not change the array length",
            ),
            (lambda source, target_length: [float(index) for index in range(149)], "at least 150"),
        ],
    )
    def test_rejects_invalid_scaling_function_outputs(self, scaling_function, expected_error):
        with pytest.raises(ValueError, match=expected_error):
            characterize_shape_scaling_method(scaling_function)

    def test_ilastik_op_resize_is_half_pixel_space_preserving(self):
        op_resize = pytest.importorskip("lazyflow.operators.opResize")
        graph = pytest.importorskip("lazyflow.graph")
        vigra = pytest.importorskip("vigra")
        np = pytest.importorskip("numpy")

        def resize_with_op(x, target_length):
            op_scale = op_resize.OpResize(
                graph=graph.Graph(),
                RawImage=vigra.taggedView(np.asarray(x), "x"),
                TargetShape=(target_length,),
                InterpolationOrder=1,
            )
            return op_scale.ResizedImage[:].wait()

        characterization = characterize_shape_scaling_method(resize_with_op)

        assert characterization.translating is half_pixel_space_preservation
        assert characterization.translating_error < 1e-9
        assert characterization.rounding is None


def _mock_factor_scaling_function(*, rounding_rule, pixel_sizing, shift_fn, divisor_convention):
    """
    rounding_rule: (n, factor) -> target_length, using the convention's own direction
      (n*factor for multiplier, n/factor for divisor).
    pixel_sizing: "shape_ratio" | "corner_ratio" | "exact_factor"
    divisor_convention: True for step/divisor (factor>1 shrinks), False for multiplier.
    """

    def scaling_function(source, factor):
        n = len(source)
        target_length = rounding_rule(n, factor)
        if pixel_sizing == "exact_factor":
            spacing = float(factor) if divisor_convention else 1.0 / factor
        elif pixel_sizing == "shape_ratio":
            spacing = n / target_length
        else:
            spacing = (n - 1) / (target_length - 1)
        base = _scale([("x", 1.0)])
        target = _scale([("x", spacing)])
        first_coordinate = shift_fn(base, target)["x"]
        return _linear_scale_values(first_coordinate=first_coordinate, spacing=spacing, length=target_length)

    return scaling_function


class TestCharacterizeShapeFactorScaler:
    def test_detects_exact_factor_and_rounding(self):
        # Uses the full official probe set, so this incidentally exercises both the
        # downsampling (factor<1) and upsampling (factor>1) rounding probes at once.
        scaling_function = _mock_factor_scaling_function(
            rounding_rule=lambda n, s: math.ceil(n * s),
            pixel_sizing="exact_factor",
            shift_fn=first_value_decimation,
            divisor_convention=False,
        )

        characterization = characterize_shape_factor_scaling_method(scaling_function)

        assert characterization.rounding == "ceil"
        assert characterization.rounding_error >= 1
        assert characterization.pixel_sizing == "exact_factor"
        assert characterization.pixel_sizing_error < 1e-9
        assert characterization.translating is first_value_decimation
        assert characterization.translating_error < 1e-9

    def test_detects_shape_ratio_and_round_half_up(self):
        scaling_function = _mock_factor_scaling_function(
            rounding_rule=lambda n, s: math.floor(n * s + 0.5),
            pixel_sizing="shape_ratio",
            shift_fn=half_pixel_space_preservation,
            divisor_convention=False,
        )

        characterization = characterize_shape_factor_scaling_method(scaling_function)

        assert characterization.rounding == "round_half_up"
        assert characterization.pixel_sizing == "shape_ratio"
        assert characterization.translating is half_pixel_space_preservation

    def test_falls_back_when_primary_probe_factor_rejected(self):
        """Method rejects the literal probe factor 0.37 (but accepts the 0.375 fallback, and
        accepts every rounding probe except the one that happens to also use 0.37)."""
        base_mock = _mock_factor_scaling_function(
            rounding_rule=lambda n, s: math.ceil(n * s),
            pixel_sizing="exact_factor",
            shift_fn=first_value_decimation,
            divisor_convention=False,
        )

        def scaling_function(source, factor):
            if factor == 0.37:
                raise ValueError("this method refuses factor=0.37 specifically")
            return base_mock(source, factor)

        characterization = characterize_shape_factor_scaling_method(scaling_function)

        assert characterization.rounding == "ceil"
        assert characterization.pixel_sizing == "exact_factor"
        assert characterization.pixel_sizing_error < 1e-9

    def test_detects_inconsistent_rounding_behaviour(self):
        """Simulates erratic/non-deterministic-looking output: the same nominal rule doesn't
        hold across different probe sizes. This is the actual safety net for both genuine
        inconsistency and (to the extent it manifests as inconsistency across differing
        probe inputs) non-determinism -- we can't detect true call-to-call non-determinism
        for identical inputs, since each probe uses distinct (n, factor) pairs."""

        def erratic_scaling_function(source, factor):
            n = len(source)
            target_length = math.ceil(n * factor) if n % 2 == 0 else math.floor(n * factor)
            spacing = 1.0 / factor
            return _linear_scale_values(first_coordinate=0.0, spacing=spacing, length=target_length)

        characterization = characterize_shape_factor_scaling_method(erratic_scaling_function)

        assert characterization.rounding == "indeterminate"
        assert characterization.pixel_sizing == "exact_factor"
        assert characterization.pixel_sizing_error < 1e-9

    def test_raises_when_only_one_probe_succeeds(self):
        def picky_scaling_function(source, factor):
            n = len(source)
            if (n, factor) != (1001, 0.5):
                raise ValueError("only supports this exact probe")
            return [0.0, 1.0, 2.0]

        with pytest.raises(ValueError, match="failed to execute more than one attempted parameter combination"):
            characterize_shape_factor_scaling_method(picky_scaling_function)


class TestCharacterizeStepFactorScaler:
    def test_detects_shape_ratio_and_floor(self):
        scaling_function = _mock_factor_scaling_function(
            rounding_rule=lambda n, s: math.floor(n / s),
            pixel_sizing="shape_ratio",
            shift_fn=discrete_bin_center,
            divisor_convention=True,
        )

        characterization = characterize_step_factor_scaling_method(scaling_function)

        assert characterization.rounding == "floor"
        assert characterization.pixel_sizing == "shape_ratio"
        assert characterization.translating is discrete_bin_center

    def test_detects_exact_factor_and_ceil(self):
        scaling_function = _mock_factor_scaling_function(
            rounding_rule=lambda n, s: math.ceil(n / s),
            pixel_sizing="exact_factor",
            shift_fn=first_value_decimation,
            divisor_convention=True,
        )

        characterization = characterize_step_factor_scaling_method(scaling_function)

        assert characterization.rounding == "ceil"
        assert characterization.pixel_sizing == "exact_factor"
        assert characterization.translating is first_value_decimation


class TestCharacterizeExactDivisionOnlyFunction:
    """Special case: The supplied scaling function only accepts params that exactly divide the shape provided"""

    def test_characterize_shape_factor_scaling_method_identifies_exact_division_requirement(self):
        """Some scaling methods may only accept factors that evenly divide the provided shape.
        More likely with step-factors than shape-factors, but possible."""

        def exact_division_only(source, factor):
            n = len(source)
            if (n * factor) % 1 != 0:
                raise ValueError("only supports exact division")
            return _linear_scale_values(first_coordinate=0.0, spacing=1.0 / factor, length=int(n * factor))

        characterization = characterize_shape_factor_scaling_method(exact_division_only)

        assert characterization.rounding == "error_on_round"
        assert characterization.rounding_error == math.inf
        assert characterization.pixel_sizing == "shape_ratio"
        assert characterization.translating is first_value_decimation

    def test_characterize_step_factor_scaling_method_identifies_exact_division_requirement(self):
        """Same reasoning as above"""

        def exact_division_only(source, factor):
            n = len(source)
            if n % factor != 0:
                raise ValueError("only supports exact division")
            target_length = n // factor
            return _linear_scale_values(first_coordinate=0.0, spacing=float(factor), length=target_length)

        characterization = characterize_step_factor_scaling_method(exact_division_only)

        assert characterization.rounding == "error_on_round"
        assert characterization.rounding_error == math.inf
        assert characterization.pixel_sizing == "shape_ratio"
        assert characterization.translating is first_value_decimation

    def test_characterize_shape_factor_scaling_method_still_raises_when_genuinely_broken(self):
        def always_broken(_source, _factor):
            raise RuntimeError("wrong signature, or just broken")

        with pytest.raises(ValueError, match="failed to execute more than one attempted parameter combination"):
            characterize_shape_factor_scaling_method(always_broken)
