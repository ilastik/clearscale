"""Knowledge about the units OME-Zarr recognises for space and time axes, to convert between them."""

import unicodedata
from fractions import Fraction
from typing import Dict, Optional, Tuple

_SI_PREFIX_EXPONENTS: Dict[str, int] = {
    "yocto": -24,
    "zepto": -21,
    "atto": -18,
    "femto": -15,
    "pico": -12,
    "nano": -9,
    "micro": -6,
    "milli": -3,
    "centi": -2,
    "deci": -1,
    "": 0,
    "hecto": 2,
    "kilo": 3,
    "mega": 6,
    "giga": 9,
    "tera": 12,
    "peta": 15,
    "exa": 18,
    "zetta": 21,
    "yotta": 24,
}

SPACE = "space"
TIME = "time"

UNITS: Dict[str, Tuple[str, Fraction]] = {}
"""The units recognised by OME-Zarr: {name: (dimension, size in meter or second)}.
Sizes are exact fractions, so that conversion factors between decimal units come out exactly."""
for _prefix, _exponent in _SI_PREFIX_EXPONENTS.items():
    UNITS[f"{_prefix}meter"] = (SPACE, Fraction(10) ** _exponent)
    UNITS[f"{_prefix}second"] = (TIME, Fraction(10) ** _exponent)
UNITS.update(
    {
        "angstrom": (SPACE, Fraction(1, 10**10)),
        "foot": (SPACE, Fraction(3048, 10000)),
        "inch": (SPACE, Fraction(254, 10000)),
        "yard": (SPACE, Fraction(9144, 10000)),
        "mile": (SPACE, Fraction(1609344, 1000)),
        # IAU 2015 value (648000 / pi astronomical units), rounded to the meter
        "parsec": (SPACE, Fraction(30856775814913673)),
        "minute": (TIME, Fraction(60)),
        "hour": (TIME, Fraction(3600)),
        "day": (TIME, Fraction(86400)),
    }
)

_PLURALS = {"foot": "feet", "inch": "inches"}

_NAME_ALIASES: Dict[str, str] = {}
"""Lower-case spellings (plurals, British spelling, a few common words) -> OME-Zarr name"""
for _name in UNITS:
    _plural = _PLURALS.get(_name, f"{_name}s")
    _NAME_ALIASES[_name] = _name
    _NAME_ALIASES[_plural] = _name
    if _name.endswith("meter"):
        _NAME_ALIASES[f"{_name[:-5]}metre"] = _name
        _NAME_ALIASES[f"{_name[:-5]}metres"] = _name
_NAME_ALIASES.update({"micron": "micrometer", "microns": "micrometer", "sec": "second", "secs": "second"})
_NAME_ALIASES.update({"hr": "hour", "hrs": "hour"})

_SI_PREFIX_SYMBOLS: Dict[str, str] = {
    "yocto": "y",
    "zepto": "z",
    "atto": "a",
    "femto": "f",
    "pico": "p",
    "nano": "n",
    # U+00B5 is the micro sign; unicodedata.normalize turns it into the Greek letter mu U+03BC.
    # We get the round-trip via normalize:
    # SYMBOLS["micrometer"] == "\u00b5m"
    # _SYMBOL_ALIASES["\u03bcm"] == "micrometer"
    # _SYMBOL_ALIASES[unicodedata.normalize("NFKC", SYMBOLS["micrometer"])] == "micrometer"
    "micro": "\u00b5",
    "milli": "m",
    "centi": "c",
    "deci": "d",
    "": "",
    "hecto": "h",
    "kilo": "k",
    "mega": "M",
    "giga": "G",
    "tera": "T",
    "peta": "P",
    "exa": "E",
    "zetta": "Z",
    "yotta": "Y",
}

SYMBOLS: Dict[str, str] = {}
"""The standard short symbol of every unit in UNITS: {OME-Zarr name: symbol}, e.g. {"nanometer": "nm"}."""
for _prefix, _symbol in _SI_PREFIX_SYMBOLS.items():
    SYMBOLS[f"{_prefix}meter"] = f"{_symbol}m"
    SYMBOLS[f"{_prefix}second"] = f"{_symbol}s"
SYMBOLS.update(
    {
        "angstrom": "\u00c5",
        "foot": "ft",
        "inch": "in",
        "yard": "yd",
        "mile": "mi",
        "parsec": "pc",
        "minute": "min",
        "hour": "h",
        "day": "d",
    }
)

_SYMBOL_ALIASES: Dict[str, str] = {unicodedata.normalize("NFKC", symbol): name for name, symbol in SYMBOLS.items()}
"""Symbols (case-sensitive, because e.g. "mm" (millimeter) and "Mm" (megameter) differ) -> OME-Zarr name"""
_SYMBOL_ALIASES.update({"um": "micrometer", "us": "microsecond"})  # ASCII spellings of the micro symbols


def canonical_unit_name(unit: str) -> Optional[str]:
    """The OME-Zarr name of `unit`, or None if it is not recognised.
    Recognises the OME-Zarr names, their plurals, British spelling, and common symbols such as "nm", "um", "µm",
    "micron", "Å", "ms", "min"."""
    text = unicodedata.normalize("NFKC", unit).strip()
    if text in _SYMBOL_ALIASES:
        return _SYMBOL_ALIASES[text]
    return _NAME_ALIASES.get(text.casefold())


def unit_conversion_factor(unit: str, other: str) -> float:
    """How many `unit` are one `other` (`unit/other`):
      `unit_conversion_factor("nanometer", "micrometer") == 1000.0`
    It is used in Affine's "pull" direction:
      `value_in_unit = value_in_other * factor`
    If `unit` is source and `other` is target:
      `value_in_source = value_in_target * factor`
    For example sampling micrometer from nanometer (which is what "pull" means):
      `1000nm = 1um * 1000.0 nm/um`
    Raises ValueError if either unit is not recognised, or if they measure different things (space vs time).
    Identical strings convert with factor 1.0, even if they are not recognised."""
    if unit == other:
        return 1.0
    sizes = []
    for spelling in (unit, other):
        name = canonical_unit_name(spelling)
        if name is None:
            raise ValueError(
                f"Unknown unit {spelling!r}. Known are the units OME-Zarr recognises for space and time axes, "
                "and common spellings like 'nm', 'um' or 'micron'."
            )
        sizes.append(UNITS[name])
    # "meter" for unit_type SPACE, "seconds" for TIME
    (unit_type, meter_per_unit), (other_type, meter_per_other) = sizes
    if unit_type != other_type:
        raise ValueError(f"Cannot convert between {unit!r} ({unit_type}) and {other!r} ({other_type}).")
    return float(meter_per_other / meter_per_unit)  # = `meters/other * unit/meters` = `unit/other`


def normalized_unit(unit: str) -> str:
    """The OME-Zarr name of `unit` if it is recognised, otherwise `unit` unchanged (including blank)."""
    return canonical_unit_name(unit) or unit


def is_same_unit(unit: str, other: str) -> bool:
    """Whether `unit` and `other` are the same unit, up to spelling: Identical strings, or both recognised as the
    same OME-Zarr unit ("nm" and "nanometer"). Unrecognised units are only the same if the strings are identical."""
    if unit == other:
        return True
    name = canonical_unit_name(unit)
    return name is not None and name == canonical_unit_name(other)


def abbreviated_unit(unit: str) -> str:
    """The standard short symbol of `unit` if it is recognised ("nanometer" -> "nm", "micron" -> "\u00b5m"),
    otherwise `unit` unchanged (including blank)."""
    name = canonical_unit_name(unit)
    return unit if name is None else SYMBOLS[name]
