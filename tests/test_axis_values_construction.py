import pytest

from clearscale import Factor, PixelOffset, PixelSize, Shape, Translation, Unit, Coefficient

ALL_CLASSES = [Shape, Factor, PixelSize, Translation, Unit, PixelOffset, Coefficient]

VALID_EXAMPLE = {
    Shape: (2, 3),
    Factor: (2.0, 3.0),
    PixelSize: (2.0, 3.0),
    Translation: (2.0, 3.0),
    Unit: ("nm", "um"),
    PixelOffset: (2, 3),
    Coefficient: (2.0, 3.0),
}


@pytest.fixture(params=ALL_CLASSES, ids=lambda cls: cls.__name__)
def cls(request):
    return request.param


class TestCommonConstruction:
    def test_from_kwargs_mapping_and_pairs_are_equivalent(self, cls):
        y, x = VALID_EXAMPLE[cls]

        from_kwargs = cls(y=y, x=x)
        from_mapping = cls({"y": y, "x": x})
        from_pairs = cls([("y", y), ("x", x)])

        assert from_kwargs == from_mapping == from_pairs

    def test_mapping_can_be_combined_with_kwargs(self, cls):
        y, x = VALID_EXAMPLE[cls]

        assert cls({"y": y}, x=x) == cls(y=y, x=x)

    def test_preserves_axis_order(self, cls):
        y, x = VALID_EXAMPLE[cls]

        assert list(cls(x=x, y=y).keys()) == ["x", "y"]
        assert list(cls(y=y, x=x).keys()) == ["y", "x"]

    def test_empty_is_rejected(self, cls):
        with pytest.raises(ValueError, match=f"Empty {cls.__name__}"):
            cls()

    def test_none_key_is_rejected(self, cls):
        y, _ = VALID_EXAMPLE[cls]

        with pytest.raises(ValueError, match="None keys not allowed"):
            cls({None: y})

    def test_none_value_is_rejected(self, cls):
        with pytest.raises(ValueError, match="None values not allowed"):
            cls(x=None)

    def test_has_no_item_assignment(self, cls):
        y, x = VALID_EXAMPLE[cls]

        with pytest.raises(TypeError, match="does not support item assignment"):
            cls(y=y, x=x)["x"] = y  # type: ignore[index]

    def test_equal_values_hash_equal(self, cls):
        y, x = VALID_EXAMPLE[cls]

        assert hash(cls(y=y, x=x)) == hash(cls(y=y, x=x))

    def test_is_usable_as_dict_key(self, cls):
        y, x = VALID_EXAMPLE[cls]

        assert {cls(y=y, x=x): "found"}[cls(y=y, x=x)] == "found"

    def test_repr_names_class_and_shows_items_in_order(self, cls):
        y, x = VALID_EXAMPLE[cls]

        assert repr(cls(y=y, x=x)) == f"{cls.__name__}({{'y': {y!r}, 'x': {x!r}}})"

    def test_different_classes_with_same_content_are_not_equal(self):
        assert Factor(x=1.0) != PixelSize(x=1.0)
        assert Translation(x=1.0) != PixelSize(x=1.0)


class TestBoolsAreNotNumbers:
    @pytest.mark.parametrize("cls", [Shape, Factor, PixelSize, Translation, PixelOffset, Coefficient])
    @pytest.mark.parametrize("value", [True, False])
    def test_bools_are_rejected(self, cls, value):
        with pytest.raises(TypeError, match="All values must be .* Got bool for axis 'x'"):
            cls(x=value)


class TestFloatValuesMustBeFinite:
    @pytest.mark.parametrize("cls", [Factor, PixelSize, Translation, Coefficient])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_are_rejected(self, cls, value):
        with pytest.raises(ValueError, match="All values must be finite"):
            cls(x=value)


class TestShapeConstruction:
    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_values_below_one(self, value):
        with pytest.raises(ValueError, match=r"Shape cannot be lower than 1 \(got .* for axis 'x'\)"):
            Shape(x=value)

    @pytest.mark.parametrize("value", [1.5, 2.0, "2"])
    def test_rejects_non_integers_even_if_integral_valued(self, value):
        with pytest.raises(TypeError, match="All values must be integer"):
            Shape(x=value)


class TestScalingsConstruction:
    @pytest.mark.parametrize("cls", [Factor, PixelSize, Coefficient])
    def test_converts_integers_to_float(self, cls):
        result = cls(x=2)

        assert result["x"] == 2.0
        assert isinstance(result["x"], float)

    @pytest.mark.parametrize("cls", [Factor, PixelSize, Coefficient])
    def test_rejects_strings(self, cls):
        with pytest.raises(TypeError, match="All values must be float"):
            cls(x="a")

    @pytest.mark.parametrize("cls", [Factor, PixelSize])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_zero_and_negative(self, cls, value):
        with pytest.raises(ValueError, match="cannot be 0 or negative"):
            cls(x=value)

    def test_coefficient_accepts_zero_and_negative(self):
        Coefficient(x=0.0)
        Coefficient(x=-5.0)


class TestTranslationConstruction:
    def test_accepts_zero_and_negative(self):
        assert Translation(x=0.0, y=-1.5) == {"x": 0.0, "y": -1.5}

    def test_converts_integers_to_float(self):
        assert isinstance(Translation(x=2)["x"], float)


class TestPixelOffsetConstruction:
    def test_accepts_zero_and_negative(self):
        assert PixelOffset(x=0, y=-3) == {"x": 0, "y": -3}

    @pytest.mark.parametrize("value", [1.5, 2.0])
    def test_rejects_floats(self, value):
        with pytest.raises(TypeError, match="All values must be integer"):
            PixelOffset(x=value)


class TestUnitConstruction:
    def test_rejects_non_strings(self):
        with pytest.raises(TypeError, match="All values must be strings"):
            Unit(x=1)

    def test_accepts_empty_string_as_unspecified(self):
        assert Unit(x="").is_default()


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_with_axes_reordering_preserves_values_and_ordered_equality(cls):
    y, x = VALID_EXAMPLE[cls]
    original = cls(y=y, x=x)
    reordered = original.with_axes("xy")
    assert tuple(reordered.items()) == (("x", x), ("y", y))
    assert reordered != original
    assert reordered.with_axes("yx") == original
    assert hash(reordered.with_axes("yx")) == hash(original)
    assert tuple(original.items()) == (("y", y), ("x", x))
