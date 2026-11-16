import pytest

from clearscale import Factor, PixelOffset, PixelSize, Shape, Translation, Unit


class TestFromKeys:
    def test_fills_each_class_with_its_default(self):
        assert Shape.fromkeys("yx") == {"y": 1, "x": 1}
        assert Factor.fromkeys("yx") == {"y": 1.0, "x": 1.0}
        assert PixelSize.fromkeys("yx") == {"y": 1.0, "x": 1.0}
        assert Translation.fromkeys("yx") == {"y": 0.0, "x": 0.0}
        assert PixelOffset.fromkeys("yx") == {"y": 0, "x": 0}
        assert Unit.fromkeys("yx") == {"y": "", "x": ""}

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="Empty Shape"):
            Shape.fromkeys("")


class TestWithAxes:
    def test_inserts_missing_axes_with_default_and_orders_like_given(self):
        assert list(Shape(y=2).with_axes("zyx").items()) == [("z", 1), ("y", 2), ("x", 1)]

    def test_drops_axes_not_given(self):
        assert Shape(y=2, x=3).with_axes("x") == Shape(x=3)

    def test_empty_axes_are_rejected(self):
        with pytest.raises(ValueError, match="Cannot create empty Shape"):
            Shape(y=2).with_axes("")


class TestWithAxesOrder:
    def test_reorders(self):
        assert list(Shape(y=2, x=3).with_axes_order("xy").items()) == [("x", 3), ("y", 2)]

    def test_ignores_given_axes_that_are_not_present(self):
        assert list(Shape(y=2, x=3).with_axes_order("zxy").items()) == [("x", 3), ("y", 2)]

    def test_refuses_to_drop_axes(self):
        with pytest.raises(ValueError, match=r"This would drop: \('y',\)"):
            Shape(y=2, x=3).with_axes_order("x")

    def test_non_string_axis_keys_against_string_axes_are_reported_as_dropped(self):
        with pytest.raises(ValueError, match="This would drop: "):
            Shape({1: 2, "x": 3}).with_axes_order("x")


class TestWithoutAxes:
    def test_removes_given_axes(self):
        assert Shape(y=2, x=3).without_axes("y") == Shape(x=3)

    def test_refuses_to_remove_all_axes(self):
        with pytest.raises(ValueError, match="would leave no axes"):
            Shape(y=2).without_axes("y")

    def test_without_axes_except_keeps_only_given_axes_in_own_order(self):
        assert list(Shape(y=2, x=3, z=4).without_axes_except("xy").items()) == [("y", 2), ("x", 3)]

    def test_without_axes_except_refuses_if_none_present(self):
        with pytest.raises(ValueError, match="None of the specified axes"):
            Shape(y=2).without_axes_except("z")


class TestDefaults:
    def test_with_default_resets_given_axes(self):
        assert Shape(y=2, x=3).with_default("y") == Shape(y=1, x=3)
        assert Translation(y=2.0, x=3.0).with_default("y") == Translation(y=0.0, x=3.0)

    def test_with_default_except_resets_all_other_axes(self):
        assert Shape(y=2, x=3).with_default_except("y") == Shape(y=2, x=1)

    def test_without_default_values_removes_axes_at_default(self):
        assert Shape(y=1, x=3).without_singletons() == Shape(x=3)

    def test_without_singletons_refuses_to_remove_all_axes(self):
        with pytest.raises(ValueError, match="would leave no axes"):
            Shape(y=1).without_singletons()

    def test_is_default(self):
        assert Shape(y=1, x=1).is_default()
        assert not Shape(y=1, x=2).is_default()

    def test_shape_with_ones_resets_given_axes(self):
        assert Shape(y=2, x=3).with_ones("y") == Shape(y=1, x=3)


class TestWithValues:
    def test_replaces_given_values(self):
        assert Shape(y=2, x=3).with_values({"y": 5}) == Shape(y=5, x=3)

    def test_only_restricts_which_axes_are_replaced(self):
        assert Shape(y=2, x=3).with_values({"y": 5, "x": 7}, only="x") == Shape(y=2, x=7)

    def test_ignores_axes_not_present(self):
        assert Shape(y=2).with_values({"z": 5}) == Shape(y=2)

    def test_skips_none_values(self):
        assert Shape(y=2, x=3).with_values({"y": None}) == Shape(y=2, x=3)  # type: ignore[reportArgumentType]

    def test_empty_replacements_return_self(self):
        shape = Shape(y=2)

        assert shape.with_values({}) is shape

    def test_empty_only_replaces_nothing(self):
        shape = Shape(y=2, x=3)

        assert shape.with_values({"y": 5}, only=()) == shape

    def test_converts_int_to_float_for_float_valued_classes(self):
        result = Factor(y=2.0).with_values({"y": 5})

        assert result == Factor(y=5.0)
        assert isinstance(result["y"], float)

    def test_refuses_lossy_float_to_int_cast(self):
        with pytest.raises(TypeError, match="Cannot cast 5.0 of type float to int"):
            Shape(y=2).with_values({"y": 5.0})

    def test_refuses_int_for_unit(self):
        with pytest.raises(TypeError, match="Cannot cast 5 of type int to str"):
            Unit(y="nm").with_values({"y": 5})

    def test_validates_new_values(self):
        with pytest.raises(ValueError, match="Shape cannot be lower than 1"):
            Shape(y=2).with_values({"y": 0})

    def test_requires_a_mapping(self):
        with pytest.raises(TypeError, match=r"Pass \{axis: value\} mapping"):
            Shape(y=2).with_values([("y", 5)])  # type: ignore[arg-type]


class TestKeysAccessors:
    def test_keys_tuple_is_an_ordered_tuple(self):
        assert Shape(y=2, x=3).keys_tuple() == ("y", "x")

    def test_keys_list_is_an_ordered_list(self):
        assert Shape(y=2, x=3).keys_list() == ["y", "x"]

    def test_keys_list_is_a_copy(self):
        shape = Shape(y=2, x=3)

        shape.keys_list().append("z")

        assert shape.keys_list() == ["y", "x"]
