import typing

import clearscale
import clearscale.str_axis_keys as str_keys

SPECIALIZED = {"Factor", "PixelOffset", "PixelSize", "Shape", "Translation", "Unit"}


def test_exports_same_names_as_clearscale():
    assert sorted(str_keys.__all__) == sorted(clearscale.__all__)


def test_axis_value_classes_are_specialized_to_str():
    for name in SPECIALIZED:
        alias = getattr(str_keys, name)
        assert typing.get_origin(alias) is getattr(clearscale, name)
        assert typing.get_args(alias) == (str,)


def test_everything_else_is_the_same_object():
    for name in set(clearscale.__all__) - SPECIALIZED:
        assert getattr(str_keys, name) is getattr(clearscale, name)


def test_specialized_aliases_construct_regular_instances():
    shape = str_keys.Shape(z=1, x=2)
    assert isinstance(shape, clearscale.Shape)
    assert shape == clearscale.Shape(z=1, x=2)
