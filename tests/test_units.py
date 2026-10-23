import pytest

from clearscale import Unit
from clearscale._services.units import (
    SYMBOLS,
    UNITS,
    abbreviated_unit,
    canonical_unit_name,
    is_same_unit,
    normalized_unit,
    unit_conversion_factor,
)

OME_ZARR_SPACE_UNITS = (
    "angstrom attometer centimeter decimeter exameter femtometer foot gigameter hectometer inch kilometer megameter "
    "meter micrometer mile millimeter nanometer parsec petameter picometer terameter yard yoctometer yottameter "
    "zeptometer zettameter"
).split()
OME_ZARR_TIME_UNITS = (
    "attosecond centisecond day decisecond exasecond femtosecond gigasecond hectosecond hour kilosecond megasecond "
    "microsecond millisecond minute nanosecond petasecond picosecond second terasecond yoctosecond yottasecond "
    "zeptosecond zettasecond"
).split()


def test_table_holds_exactly_the_units_ome_zarr_recognises():
    assert set(UNITS) == set(OME_ZARR_SPACE_UNITS) | set(OME_ZARR_TIME_UNITS)
    assert {name for name, (dimension, _size) in UNITS.items() if dimension == "space"} == set(OME_ZARR_SPACE_UNITS)


@pytest.mark.parametrize("name", OME_ZARR_SPACE_UNITS + OME_ZARR_TIME_UNITS)
def test_every_ome_zarr_name_is_recognised_as_itself(name):
    assert canonical_unit_name(name) == name


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("nm", "nanometer"),
        ("um", "micrometer"),
        ("\u00b5m", "micrometer"),  # micro sign
        ("\u03bcm", "micrometer"),  # Greek letter mu
        ("micron", "micrometer"),
        ("Microns", "micrometer"),
        ("NANOMETERS", "nanometer"),
        ("  nm ", "nanometer"),
        ("metres", "meter"),
        ("micrometre", "micrometer"),
        ("\u00c5", "angstrom"),  # capital A with ring above (visually angstrom)
        ("\u212b", "angstrom"),  # actual angstrom sign
        ("angstroms", "angstrom"),
        ("feet", "foot"),
        ("inches", "inch"),
        ("ms", "millisecond"),
        ("\u00b5s", "microsecond"),
        ("min", "minute"),
        ("sec", "second"),
        ("seconds", "second"),
        ("h", "hour"),
    ],
)
def test_common_spellings_are_recognised(spelling, expected):
    assert canonical_unit_name(spelling) == expected


def test_symbols_are_case_sensitive():
    assert canonical_unit_name("mm") == "millimeter"
    assert canonical_unit_name("Mm") == "megameter"


@pytest.mark.parametrize("spelling", ["", "furlong", "pixel", "a.u.", "NM2"])
def test_unrecognised_units_are_none(spelling):
    assert canonical_unit_name(spelling) is None


@pytest.mark.parametrize(
    "unit, other, expected",
    [
        ("nanometer", "micrometer", 1000.0),
        ("micrometer", "nanometer", 0.001),
        ("nanometer", "meter", 1e9),
        ("meter", "millimeter", 0.001),
        ("millimeter", "kilometer", 1e6),
        ("millimeter", "inch", 25.4),
        ("inch", "foot", 12.0),
        ("foot", "yard", 3.0),
        ("yard", "mile", 1760.0),
        ("nanometer", "angstrom", 0.1),
        ("second", "hour", 3600.0),
        ("hour", "day", 24.0),
        ("second", "minute", 60.0),
        ("microsecond", "millisecond", 1000.0),
        ("nm", "um", 1000.0),
        ("\u00c5", "micron", 10000.0),
    ],
)
def test_unit_conversion_factor_is_exact(unit, other, expected):
    assert unit_conversion_factor(unit, other) == expected


def test_unit_conversion_factor_parsec():
    assert unit_conversion_factor("meter", "parsec") == pytest.approx(3.0856775814913673e16, rel=1e-15)


def test_unit_conversion_factor_inverse_round_trips():
    assert unit_conversion_factor("micrometer", "mile") * unit_conversion_factor("mile", "micrometer") == 1


def test_unit_conversion_factor_identical_strings_need_no_table():
    assert unit_conversion_factor("a.u.", "a.u.") == 1.0


@pytest.mark.parametrize("unit, other", [("furlong", "meter"), ("meter", "furlong"), ("", "meter")])
def test_unit_conversion_factor_rejects_unknown_units(unit, other):
    with pytest.raises(ValueError, match="Unknown unit"):
        unit_conversion_factor(unit, other)


def test_unit_conversion_factor_rejects_space_to_time():
    with pytest.raises(ValueError, match=r"Cannot convert between 'meter' \(space\) and 'second' \(time\)"):
        unit_conversion_factor("meter", "second")


@pytest.mark.parametrize(
    "unit, other, expected",
    [
        ("nm", "nanometer", True),
        ("\u00b5m", "um", True),
        ("micron", "micrometer", True),
        ("nanometer", "nanometer", True),
        ("a.u.", "a.u.", True),
        ("nm", "um", False),
        ("nm", "second", False),
        ("a.u.", "A.U.", False),
        ("furlong", "nm", False),
        ("", "nm", False),
        ("", "", True),
    ],
)
def test_is_same_unit(unit, other, expected):
    assert is_same_unit(unit, other) is expected
    assert is_same_unit(other, unit) is expected


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("nm", "nanometer"),
        ("Microns", "micrometer"),
        ("nanometer", "nanometer"),
        ("furlong", "furlong"),
        ("", ""),
        (" nm ", "nanometer"),
    ],
)
def test_normalized_unit(unit, expected):
    assert normalized_unit(unit) == expected


class TestUnitNormalized:
    def test_spells_recognised_units_as_ome_zarr_names_per_axis(self):
        unit = Unit(t="s", z="micron", y="nm", x="nanometer", c="")

        assert unit.normalized() == Unit(t="second", z="micrometer", y="nanometer", x="nanometer", c="")

    def test_leaves_unrecognised_units_alone(self):
        assert Unit(x="a.u.", y="Pixel").normalized() == Unit(x="a.u.", y="Pixel")

    def test_is_idempotent(self):
        unit = Unit(x="um", y="ms").normalized()

        assert unit.normalized() == unit

    def test_does_not_modify_the_original(self):
        unit = Unit(x="nm")

        unit.normalized()

        assert unit == Unit(x="nm")


def test_every_unit_has_a_unique_symbol_that_is_recognised_as_that_unit():
    assert set(SYMBOLS) == set(UNITS)
    assert len(set(SYMBOLS.values())) == len(SYMBOLS)
    for name, symbol in SYMBOLS.items():
        assert canonical_unit_name(symbol) == name
        assert len(symbol) <= len(name)


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("nanometer", "nm"),
        ("micrometer", "\u00b5m"),
        ("micron", "\u00b5m"),
        ("um", "\u00b5m"),
        ("meter", "m"),
        ("kilometer", "km"),
        ("zettameter", "Zm"),
        ("angstrom", "\u00c5"),
        ("inch", "in"),
        ("second", "s"),
        ("millisecond", "ms"),
        ("microsecond", "\u00b5s"),
        ("minute", "min"),
        ("hour", "h"),
        ("day", "d"),
        ("nm", "nm"),
        ("furlong", "furlong"),
        ("a.u.", "a.u."),
        ("", ""),
        ("\u03bcm", "\u00b5m"),  # Greek letter mu U+03BC -> micro sign U+00B5
    ],
)
def test_abbreviated_unit(unit, expected):
    assert abbreviated_unit(unit) == expected


class TestUnitAbbreviated:
    def test_abbreviates_recognised_units_per_axis(self):
        unit = Unit(t="second", z="micron", y="nanometer", x="nm", c="")

        assert unit.abbreviated() == Unit(t="s", z="\u00b5m", y="nm", x="nm", c="")

    def test_ignores_unrecognised_strings(self):
        assert Unit(x="a.u.", y="Pixel").abbreviated() == Unit(x="a.u.", y="Pixel")

    def test_is_idempotent(self):
        unit = Unit(x="micrometer", y="millisecond")

        assert unit.abbreviated().abbreviated() == unit.abbreviated()

    def test_normalized_undoes_abbreviated(self):
        unit = Unit(t="ms", z="um", y="Micron", x="nanometers", c="")

        assert unit.abbreviated().normalized() == unit.normalized()
