"""Static typing check for consumers that only use str axis keys.

Runs pyright over the functions below. The functions themselves are never
called; they only need to type-check. Intended to be removed again before merging if typing is not tested upstream.
"""

import shutil
import subprocess
import sys
from collections.abc import Mapping

import pytest

import clearscale.str_axis_keys as str_cs
from clearscale import Factor, PixelSize, Shape, Unit


def _takes_int_mapping(m: Mapping[str, int]) -> None: ...
def _takes_float_mapping(m: Mapping[str, float]) -> None: ...
def _takes_str_mapping(m: Mapping[str, str]) -> None: ...


def _annotated_str_keys(shape: Shape[str], pixel_size: PixelSize[str], unit: Unit[str]) -> None:
    _takes_int_mapping(shape)
    _takes_float_mapping(pixel_size)
    _takes_str_mapping(unit)
    # operations keep the key type
    _takes_int_mapping(shape.with_axes("tzyx"))
    _takes_int_mapping(shape.scaled_by(Factor(z=2.0), rounding="floor"))
    _takes_float_mapping(pixel_size.scaled_by(Factor(z=2.0)))
    _takes_int_mapping(Shape[str].all_singletons("zyx"))


def _bare_annotation(shape: Shape, pixel_size: PixelSize) -> None:
    # existing code annotating the bare class keeps working
    _takes_int_mapping(shape)
    _takes_float_mapping(pixel_size)


def _constructed_from_kwargs() -> None:
    _takes_int_mapping(Shape(z=1, x=2))
    _takes_float_mapping(PixelSize(z=1.0, x=0.5))
    _takes_float_mapping(Factor(z=2.0) * PixelSize(z=1.0))


def _str_axis_keys_module(shape: str_cs.Shape, pixel_size: str_cs.PixelSize) -> None:
    # no [str] needed when using the str_axis_keys namespace
    _takes_int_mapping(shape)
    _takes_float_mapping(pixel_size)
    _takes_int_mapping(str_cs.Shape.all_singletons("zyx"))
    _takes_int_mapping(str_cs.Shape([("z", 1)]))
    _takes_float_mapping(str_cs.PixelSize(z=1.0).scaled_by(str_cs.Factor(z=2.0)))


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_str_key_consumer_type_checks():
    result = subprocess.run(["pyright", __file__], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
