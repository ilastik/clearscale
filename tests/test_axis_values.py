import math
import pytest
from typing import cast

from clearscale import Factor, PixelOffset, PixelSize, Shape, Translation


def test_shape_non_singleton_axes():
    shape = Shape(c=5, x=4, t=1, y=3, z=1)
    assert shape.non_singleton_axes() == ("c", "x", "y")
    assert shape.non_singleton_axes("yx") == ("x", "y"), "maintain shape's order, not param order"
    shape2 = Shape(z=1, y=1, x=1)
    assert shape2.non_singleton_axes() == ()


def assert_axis_values(value, expected_type, expected_items):
    assert type(value) is expected_type
    assert list(value.items()) == expected_items


def test_factor_multiplies_factor_axis_wise():
    result = Factor([("y", 2.0), ("x", 4.0)]) * Factor([("y", 3.0), ("x", 0.5)])

    assert_axis_values(result, Factor, [("y", 6.0), ("x", 2.0)])


def test_factor_divides_factor_axis_wise():
    result = Factor([("y", 6.0), ("x", 3.0)]) / Factor([("y", 2.0), ("x", 1.5)])

    assert_axis_values(result, Factor, [("y", 3.0), ("x", 2.0)])


def test_factor_and_pixel_size_multiply_using_pixel_size_scaling_rules():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor(x=2.0)

    assert_axis_values(factor * pixel_size, PixelSize, [("y", 2.0), ("x", 8.0)])
    assert_axis_values(pixel_size * factor, PixelSize, [("y", 2.0), ("x", 8.0)])


def test_pixel_size_rejects_zero_values():
    with pytest.raises(ValueError, match="Pixel size cannot be 0 or negative"):
        PixelSize(c=0.0)


def test_pixel_size_rejects_negative_values():
    with pytest.raises(ValueError, match="Pixel size cannot be 0 or negative"):
        PixelSize(c=-1.0)


def test_pixel_size_divides_by_factor_using_inverted_scaling_rules():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor(x=2.0)

    assert_axis_values(pixel_size / factor, PixelSize, [("y", 2.0), ("x", 2.0)])


def test_pixel_size_mul_factor_rejects_extra_factor_axes():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base pixel size"):
        _ = factor * pixel_size


def test_factor_mul_pixel_size_rejects_extra_factor_axes():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base pixel size"):
        _ = pixel_size * factor


def test_pixel_size_div_factor_rejects_extra_factor_axes():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base pixel size"):
        _ = pixel_size / factor


def test_pixel_size_divides_pixel_size_to_factor():
    target_pixel_size = PixelSize([("y", 4.0), ("x", 1.0)])
    base_pixel_size = PixelSize([("y", 2.0), ("x", 0.5)])

    result = target_pixel_size / base_pixel_size

    assert_axis_values(result, Factor, [("y", 2.0), ("x", 2.0)])


def test_translation_adds_and_subtracts_translation_axis_wise():
    left = Translation([("y", 1.0), ("x", 5.0)])
    right = Translation([("y", 2.0), ("x", 3.0)])

    assert_axis_values(left + right, Translation, [("y", 3.0), ("x", 8.0)])
    assert_axis_values(left - right, Translation, [("y", -1.0), ("x", 2.0)])


def test_translation_mul_factor_rejects_extra_factor_axes():
    translation = Translation([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base translation"):
        _ = factor * translation


def test_pixel_offset_adds_and_subtracts_pixel_offset_axis_wise():
    left = PixelOffset([("y", 10), ("x", 3)])
    right = PixelOffset([("y", 4), ("x", 8)])

    assert_axis_values(left + right, PixelOffset, [("y", 14), ("x", 11)])
    assert_axis_values(left - right, PixelOffset, [("y", 6), ("x", -5)])


def test_pixel_offset_and_pixel_size_multiply_to_translation_with_extra_pixel_size_axes_ignored():
    offset = PixelOffset([("y", 2), ("x", 4)])
    pixel_size = PixelSize([("z", 10.0), ("y", 0.5), ("x", 2.0)])

    assert_axis_values(offset * pixel_size, Translation, [("y", 1.0), ("x", 8.0)])
    assert_axis_values(pixel_size * offset, Translation, [("y", 1.0), ("x", 8.0)])


def test_offset_mul_pixel_size_multiplication_requires_offset_axes():
    offset = PixelOffset([("y", 2), ("x", 4)])
    pixel_size = PixelSize(y=0.5)

    with pytest.raises(ValueError, match="must contain all axes"):
        _ = offset * pixel_size


def test_pixel_size_mul_offset_multiplication_requires_offset_axes():
    offset = PixelOffset([("y", 2), ("x", 4)])
    pixel_size = PixelSize(y=0.5)

    with pytest.raises(ValueError, match="must contain all axes"):
        _ = pixel_size * offset


def test_shape_divides_shape_to_factor_using_factor_to_semantics():
    original_shape = Shape([("y", 100), ("x", 50)])
    resized_shape = Shape([("y", 25), ("x", 100)])

    result = original_shape / resized_shape

    assert_axis_values(result, Factor, [("y", 4.0), ("x", 0.5)])


@pytest.mark.parametrize(
    ("translation", "axes", "expected"),
    [
        pytest.param(Translation.identity(("x", "y")), ("x", "y"), True, id="identity"),
        pytest.param(Translation({"x": 1.0, "y": 0.0}), ("y",), True, id="identity along subset"),
        pytest.param(Translation({"x": 1.0, "y": 0.0}), ("x",), False, id="non-identity along subset"),
        pytest.param(Translation({"x": 1.0, "y": 2.0}), ("x", "y"), False, id="non-identity"),
        pytest.param(Translation({"x": 1.0, "y": 2.0}), (), True, id="empty axes"),
    ],
)
def test_translation_is_identity_along(translation, axes, expected):
    assert cast(Translation, translation).is_identity_along(axes) is expected


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        ("ceil", [("y", 3), ("x", 2)]),
        ("floor", [("y", 2), ("x", 1)]),
        ("round", [("y", 3), ("x", 2)]),
    ],
)
def test_translation_to_pixel_offset_rounds_and_ignores_extra_pixel_size_axes(rounding, expected):
    translation = Translation([("y", 2.6), ("x", 3.6)])
    pixel_size = PixelSize([("z", 100.0), ("y", 1.0), ("x", 2.0)])

    result = translation.to_pixel_offset(pixel_size, rounding=rounding)

    assert_axis_values(result, PixelOffset, expected)


def test_translation_to_pixel_offset_requires_translation_axes():
    translation = Translation([("y", 2.6), ("x", 3.6)])

    with pytest.raises(ValueError, match="must contain all axes"):
        translation.to_pixel_offset(PixelSize(y=1.0), rounding="round")


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (lambda left, right: left * right, Factor(y=2.0, x=3.0), Factor(x=4.0, y=5.0)),
        (lambda left, right: left / right, Factor(y=2.0, x=3.0), Factor(x=4.0, y=5.0)),
        (lambda left, right: left / right, PixelSize(y=2.0, x=3.0), PixelSize(x=4.0, y=5.0)),
        (lambda left, right: left + right, Translation(y=2.0, x=3.0), Translation(x=4.0, y=5.0)),
        (lambda left, right: left - right, Translation(y=2.0, x=3.0), Translation(x=4.0, y=5.0)),
        (lambda left, right: left + right, PixelOffset(y=2, x=3), PixelOffset(x=4, y=5)),
        (lambda left, right: left - right, PixelOffset(y=2, x=3), PixelOffset(x=4, y=5)),
        (lambda left, right: left / right, Shape(y=20, x=30), Shape(x=40, y=50)),
    ],
)
def test_same_kind_operations_reject_incompatible_axis_order(operation, left, right):
    with pytest.raises(ValueError, match="Incompatible axes"):
        operation(left, right)


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (lambda left, right: left * right, Factor(y=2.0, x=3.0), Factor(y=4.0, z=5.0)),
        (lambda left, right: left / right, Factor(y=2.0, x=3.0), Factor(y=4.0, z=5.0)),
        (lambda left, right: left / right, PixelSize(y=2.0, x=3.0), PixelSize(y=4.0, z=5.0)),
        (lambda left, right: left + right, Translation(y=2.0, x=3.0), Translation(y=4.0, z=5.0)),
        (lambda left, right: left - right, Translation(y=2.0, x=3.0), Translation(y=4.0, z=5.0)),
        (lambda left, right: left + right, PixelOffset(y=2, x=3), PixelOffset(y=4, z=5)),
        (lambda left, right: left - right, PixelOffset(y=2, x=3), PixelOffset(y=4, z=5)),
        (lambda left, right: left / right, Shape(y=20, x=30), Shape(y=40, z=50)),
    ],
)
def test_same_kind_operations_reject_incompatible_axes(operation, left, right):
    with pytest.raises(ValueError, match="Incompatible axes"):
        operation(left, right)


def test_implemented_dunders_return_not_implemented_for_unsupported_operands():
    assert Factor(y=2.0).__mul__(Shape(y=2)) is NotImplemented  # type: ignore[reportArgumentType]
    assert Factor(y=2.0).__truediv__(PixelSize(y=1.0)) is NotImplemented
    assert PixelSize(y=1.0).__mul__(PixelSize(y=2.0)) is NotImplemented
    assert PixelSize(y=1.0).__truediv__(Shape(y=2)) is NotImplemented
    assert Translation(y=1.0).__add__(PixelOffset(y=1)) is NotImplemented
    assert Translation(y=1.0).__sub__(PixelOffset(y=1)) is NotImplemented
    assert Translation(y=1.0).__mul__(PixelSize(y=1)) is NotImplemented
    assert PixelOffset(y=1).__add__(Shape(y=1)) is NotImplemented
    assert PixelOffset(y=1).__sub__(Shape(y=1)) is NotImplemented
    assert PixelOffset(y=1).__mul__(Factor(y=1.0)) is NotImplemented
    assert Shape(y=1).__truediv__(Factor(y=1.0)) is NotImplemented


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Shape(y=4) * Factor(y=2.0),  # type: ignore[operator]
        lambda: Shape(y=4) / Factor(y=2.0),  # type: ignore[operator]
        lambda: Translation(y=1.0) / PixelSize(y=1.0),  # type: ignore[operator]
        lambda: Shape(y=4) * PixelSize(y=1.0),  # type: ignore[operator]
        lambda: PixelOffset(y=1) + Shape(y=1),  # type: ignore[operator]
        lambda: Shape(y=1) + PixelOffset(y=1),  # type: ignore[operator]
        lambda: Factor(y=1.0) + Factor(y=1.0),  # type: ignore[operator]
        lambda: Factor(y=1.0) - Factor(y=1.0),  # type: ignore[operator]
        lambda: PixelSize(y=1.0) + PixelSize(y=1.0),  # type: ignore[operator]
        lambda: PixelSize(y=1.0) - PixelSize(y=1.0),  # type: ignore[operator]
        lambda: Shape(y=1) + Shape(y=1),  # type: ignore[operator]
        lambda: Shape(y=1) - Shape(y=1),  # type: ignore[operator]
        lambda: PixelOffset(y=1) / Factor(y=1.0),  # type: ignore[operator]
    ],
)
def test_unsupported_arithmetic_raises_type_error(operation):
    with pytest.raises(TypeError, match="unsupported operand type"):
        operation()


def test_axis_values_allow_keys_with_the_same_str():
    """Not necessarily desirable or intended behaviour; this test just documents an existing edge case.
    Multiscale does enforce unique str, but on the primitives it's not per se a problem."""
    shape = Shape({1: 4, "1": 5})
    assert tuple(shape.items()) == ((1, 4), ("1", 5))


class TestShapeMatches:
    def test_equal_shapes_match(self):
        assert Shape(y=2, x=3).matches(Shape(y=2, x=3))

    def test_differing_shared_axis_does_not_match(self):
        assert not Shape(y=2, x=3).matches(Shape(y=2, x=4))

    def test_only_shared_axes_are_compared(self):
        assert Shape(y=2, x=3).matches(Shape(x=3, z=99))
        assert Shape(x=3, z=99).matches(Shape(y=2, x=3))

    def test_axis_order_does_not_matter(self):
        assert Shape(y=2, x=3).matches(Shape(x=3, y=2))

    def test_shapes_without_shared_axes_match_vacuously(self):
        assert Shape(y=2).matches(Shape(x=3))

    def test_accepts_plain_mapping(self):
        assert Shape(y=2, x=3).matches({"x": 3})
        assert not Shape(y=2, x=3).matches({"x": 4})

    def test_only_restricts_comparison_to_given_axes(self):
        left, right = Shape(y=2, x=3), Shape(y=2, x=4)

        assert left.matches(right, only="y")
        assert not left.matches(right, only="x")
        assert not left.matches(right, only=("y", "x"))

    def test_only_with_axes_that_are_not_shared_matches_vacuously(self):
        assert Shape(y=2, x=3).matches(Shape(y=5, x=4), only="z")


class TestShapeScaledBy:
    def test_missing_factor_axes_default_to_no_change(self):
        assert Shape(y=10, x=10).scaled_by(Factor(x=2.0), rounding="floor") == Shape(y=10, x=5)

    def test_factor_axis_order_does_not_matter(self):
        shape = Shape(y=10, x=20)

        assert shape.scaled_by(Factor(x=2.0, y=5.0), rounding="floor") == Shape(y=2, x=10)

    def test_rejects_extra_factor_axes_like_pixel_size_and_translation(self):
        with pytest.raises(ValueError, match="Attempted to scale axes with no base shape"):
            Shape(x=10).scaled_by(Factor(x=2.0, z=2.0), rounding="floor")

    def test_never_scales_below_one(self):
        assert Shape(x=2).scaled_by(Factor(x=10.0), rounding="floor") == Shape(x=1)

    def test_empty_only_restricts_to_nothing_like_with_values(self):
        assert Shape(y=2, x=3).matches(Shape(y=5, x=4), only=())


class TestFactorMagnitude:
    def test_is_the_product_over_all_axes(self):
        assert Factor(y=2.0, x=3.0).magnitude() == 6.0

    def test_identity_has_magnitude_one(self):
        assert Factor.identity("zyx").magnitude() == 1.0

    def test_upscaling_axes_reduce_it(self):
        assert Factor(y=0.5, x=4.0).magnitude() == 2.0


class TestFactorIsIdentity:
    def test_true_if_all_axes_are_one(self):
        assert Factor(y=1.0, x=1.0).is_identity()

    def test_false_if_any_axis_differs(self):
        assert not Factor(y=1.0, x=2.0).is_identity()

    def test_false_for_up_and_down_scaling_that_cancels_in_magnitude(self):
        assert not Factor(y=2.0, x=0.5).is_identity()


class TestFactorIsDownscaling:
    def test_true_if_product_above_one(self):
        assert Factor(y=1.0, x=2.0).is_downscaling()

    def test_false_for_identity_and_upscaling(self):
        assert not Factor(y=1.0, x=1.0).is_downscaling()
        assert not Factor(y=1.0, x=0.5).is_downscaling()

    def test_judges_by_product_over_axes_so_mixed_factors_can_be_neither(self):
        mixed = Factor(y=2.0, x=0.5)

        assert not mixed.is_downscaling()
        assert not mixed.is_upscaling()


class TestFactorIsUpscaling:
    def test_true_if_product_below_one(self):
        assert Factor(y=1.0, x=0.5).is_upscaling()

    def test_false_for_identity_and_downscaling(self):
        assert not Factor(y=1.0, x=1.0).is_upscaling()
        assert not Factor(y=1.0, x=2.0).is_upscaling()


class TestFactorWithIdentity:
    def test_resets_given_axes_to_one_and_returns_a_factor(self):
        result = Factor(z=4.0, y=2.0, x=2.0).with_identity("zy")

        assert type(result) is Factor
        assert result == Factor(z=1.0, y=1.0, x=2.0)

    def test_with_identity_except_resets_all_other_axes(self):
        assert Factor(z=4.0, y=2.0, x=2.0).with_identity_except("x") == Factor(z=1.0, y=1.0, x=2.0)


class TestFactorToPhysical:
    def test_multiplies_the_base_pixel_size(self):
        result = Factor(y=2.0, x=4.0).to_physical(PixelSize(y=0.5, x=0.25))

        assert type(result) is PixelSize
        assert result == PixelSize(y=1.0, x=1.0)

    def test_axes_missing_from_the_factor_keep_the_base_pixel_size(self):
        result = Factor(y=2.0, x=2.0).to_physical(PixelSize(z=3.0, y=0.5, x=0.25))

        assert result == PixelSize(z=3.0, y=1.0, x=0.5)

    def test_rejects_factor_axes_missing_from_the_base(self):
        with pytest.raises(ValueError, match="no base pixel size"):
            Factor(y=2.0, z=2.0).to_physical(PixelSize(y=1.0, x=1.0))


class TestPixelSizeIsIdentity:
    def test_true_if_all_axes_are_one(self):
        assert PixelSize(y=1.0, x=1.0).is_identity()

    def test_false_if_any_axis_differs(self):
        assert not PixelSize(y=1.0, x=0.5).is_identity()


class TestPixelSizeWithIdentityExcept:
    def test_resets_all_other_axes_to_one(self):
        result = PixelSize(z=4.0, y=2.0, x=0.5).with_identity_except("x")

        assert type(result) is PixelSize
        assert result == PixelSize(z=1.0, y=1.0, x=0.5)

    def test_with_identity_resets_given_axes(self):
        assert PixelSize(z=4.0, y=2.0, x=0.5).with_identity("zy") == PixelSize(z=1.0, y=1.0, x=0.5)


class TestPixelSizeScaledBy:
    def test_factor_multiplies_per_axis(self):
        assert PixelSize(y=0.5, x=0.25).scaled_by(Factor(y=2.0, x=4.0)) == PixelSize(y=1.0, x=1.0)

    def test_mapping_is_accepted_like_a_factor(self):
        assert PixelSize(y=0.5, x=0.25).scaled_by({"x": 4.0}) == PixelSize(y=0.5, x=1.0)

    def test_scalar_applies_to_all_axes(self):
        assert PixelSize(y=0.5, x=0.25).scaled_by(2) == PixelSize(y=1.0, x=0.5)
        assert PixelSize(y=0.5, x=0.25).scaled_by(2.0) == PixelSize(y=1.0, x=0.5)

    def test_rejects_extra_factor_axes(self):
        with pytest.raises(ValueError, match="no base pixel size"):
            PixelSize(x=1.0).scaled_by(Factor(x=2.0, z=2.0))


class TestTranslationScaledBy:
    def test_factor_multiplies_per_axis(self):
        assert Translation(y=1.0, x=-2.0).scaled_by(Factor(y=2.0, x=4.0)) == Translation(y=2.0, x=-8.0)

    def test_scalar_applies_to_all_axes(self):
        assert Translation(y=1.0, x=-2.0).scaled_by(3) == Translation(y=3.0, x=-6.0)

    def test_missing_factor_axes_keep_the_value(self):
        assert Translation(y=1.0, x=-2.0).scaled_by({"x": 0.5}) == Translation(y=1.0, x=-1.0)

    def test_rejects_extra_factor_axes(self):
        with pytest.raises(ValueError, match="no base translation"):
            Translation(x=1.0).scaled_by(Factor(x=2.0, z=2.0))


class TestTranslationTrueDivision:
    def test_divides_by_the_factor(self):
        assert Translation(y=4.0, x=-3.0) / Factor(y=2.0, x=0.5) == Translation(y=2.0, x=-6.0)

    def test_is_the_inverse_of_multiplying(self):
        translation, factor = Translation(y=4.0, x=-3.0), Factor(y=2.0, x=4.0)

        assert translation * factor / factor == translation

    def test_other_operand_types_are_not_supported(self):
        with pytest.raises(TypeError):
            Translation(x=1.0) / 2.0  # type: ignore[operator]


class FakeAxisInfo:
    def __init__(self, key, resolution=0.0):
        self.key = key
        self.resolution = resolution


class FakeAxisTags:
    """Duck-types the parts of vigra.AxisTags that PixelSize uses."""

    def __init__(self, *infos):
        self._infos = list(infos)

    def __iter__(self):
        return iter(self._infos)

    def __len__(self):
        return len(self._infos)

    def setResolution(self, key, resolution):
        next(info for info in self._infos if info.key == key).resolution = resolution


class TestPixelSizeFromVigra:
    def test_takes_axis_keys_and_resolutions_in_order(self):
        tags = FakeAxisTags(FakeAxisInfo("y", 0.5), FakeAxisInfo("x", 0.25))

        assert list(PixelSize.from_vigra(tags).items()) == [("y", 0.5), ("x", 0.25)]

    def test_vigra_default_resolution_zero_becomes_one(self):
        tags = FakeAxisTags(FakeAxisInfo("y"), FakeAxisInfo("x", 0.25))

        assert PixelSize.from_vigra(tags) == PixelSize(y=1.0, x=0.25)


class TestPixelSizeToVigra:
    @pytest.fixture
    def fake_vigra(self, monkeypatch):
        import sys
        import types

        module = types.ModuleType("vigra")
        module.defaultAxistags = lambda keys: FakeAxisTags(*(FakeAxisInfo(k) for k in keys))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vigra", module)

    def test_creates_axistags_with_the_resolutions(self, fake_vigra):
        tags = PixelSize(y=0.5, x=0.25).to_vigra(None)

        assert [(t.key, t.resolution) for t in tags] == [("y", 0.5), ("x", 0.25)]

    def test_modifies_given_axistags_only_for_shared_axes(self, fake_vigra):
        existing = FakeAxisTags(FakeAxisInfo("c", 7.0), FakeAxisInfo("x", 1.0))

        result = PixelSize(y=0.5, x=0.25).to_vigra(existing)

        assert result is existing
        assert [(t.key, t.resolution) for t in result] == [("c", 7.0), ("x", 0.25)]

    def test_roundtrips_through_from_vigra(self, fake_vigra):
        original = PixelSize(y=0.5, x=0.25)

        assert PixelSize.from_vigra(original.to_vigra(None)) == original

    def test_explains_how_to_install_missing_vigra(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "vigra", None)

        with pytest.raises(ImportError, match='requires the package "vigra"'):
            PixelSize(x=1.0).to_vigra(None)


@pytest.mark.parametrize("rounding", [math.ceil, lambda value: int(value), None, 3, "unknown"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda rounding: Shape(x=8).scaled_by(Factor(x=2.0), rounding=rounding),
        lambda rounding: Translation(x=2.5).to_pixel_offset(PixelSize(x=1.0), rounding=rounding),
    ],
)
def test_rounding_rejects_functions_and_invalid_rules(operation, rounding):
    with pytest.raises(ValueError if isinstance(rounding, str) else TypeError, match="rounding"):
        operation(rounding)
