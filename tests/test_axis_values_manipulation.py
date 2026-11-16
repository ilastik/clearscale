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


@pytest.mark.parametrize(
    "factory",
    [
        Shape.all_singletons,
        Factor.identity,
        PixelSize.identity,
        Translation.identity,
        Unit.empty,
        lambda axes: Factor.uniform(axes, 2),
    ],
)
def test_default_factories_reject_duplicate_axes(factory):
    with pytest.raises(ValueError, match="Duplicate axes"):
        factory("xyx")


def test_duplicate_selectors_remain_set_like():
    shape = Shape(y=2, x=3)
    assert shape.without_axes("xx") == Shape(y=2)
    assert shape.without_axes_except("xx") == Shape(x=3)
    assert shape.with_default("xx") == Shape(y=2, x=1)
    assert shape.with_default_except("xx") == Shape(y=1, x=3)


@pytest.mark.parametrize("channel_present", [False, True])
def test_selectors_allow_optional_channel_axis(channel_present):
    shape = Shape(y=2, x=3)
    if channel_present:
        shape = shape.with_axes("cyx").with_values({"c": 4})
    assert shape.without_axes("ct") == Shape(y=2, x=3)
    assert shape.without_axes_except("tzyx") == Shape(y=2, x=3)
    assert shape.with_default("ct")["x"] == 3
    assert shape.with_default_except("tzyx").get("c", 1) == 1
    assert shape.with_values({"x": 5}, only="tcx")["x"] == 5


@pytest.mark.parametrize("selector", ["cxy", ("c", "x", "y")])
def test_string_selectors_match_individual_keys(selector):
    shape = Shape({"xy": 2, "x": 3, "y": 4, 1: 5, "": 6})
    assert shape.without_axes(selector) == Shape({"xy": 2, 1: 5, "": 6})
    assert shape.without_axes_except(selector) == Shape(x=3, y=4)
    assert shape.with_default(selector) == Shape({"xy": 2, "x": 1, "y": 1, 1: 5, "": 6})
    assert shape.with_default_except(selector) == Shape({"xy": 1, "x": 3, "y": 4, 1: 1, "": 1})
    assert shape.with_values({"xy": 9, "x": 9, 1: 9, "": 9}, only=selector) == Shape(
        {"xy": 2, "x": 9, "y": 4, 1: 5, "": 6}
    )
    assert shape.non_singleton_axes(selector) == ("x", "y")
    assert shape.without_axes_except(("xy", 1)) == Shape({"xy": 2, 1: 5})


@pytest.mark.parametrize("cls", [Factor, PixelSize, Translation])
def test_identity_aliases_use_individual_keys_and_allow_supersets(cls):
    value = cls({"xy": 2, "x": 3, 1: 4})
    assert value.with_identity("cxx")["xy"] == 2
    assert value.with_identity("cxx")["x"] == cls.fromkeys("x")["x"]
    assert value.with_identity_except("cxx")["x"] == 3
    assert value.with_identity_except("cxx")["xy"] == cls.fromkeys(("xy",))["xy"]


def test_translation_identity_query_uses_set_like_selectors():
    translation = Translation({"xy": 2, "x": 0, 1: 3})
    assert translation.is_identity_along("cxxy")
    assert translation.is_identity_along(())
    assert not translation.is_identity_along(("xy",))
    assert not translation.is_identity_along((1,))


def test_empty_selectors_keep_existing_filtering_contract():
    shape = Shape(y=2, x=3)
    assert shape.without_axes(()) == shape
    assert shape.with_default(()) == shape
    assert shape.with_default_except(()) == Shape(y=1, x=1)
    with pytest.raises(ValueError, match="None of the specified axes"):
        shape.without_axes_except(())
