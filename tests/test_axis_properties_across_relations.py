import pytest

from clearscale import Affine, AxisRenaming, PermutationTo, ProjectionTo, Translation
from clearscale._services.axis_links import AxisLinks
from clearscale._services.axis_properties_across_relations import (
    require_matching_units_or_affine_relation,
    reconcile_axis_prop_params,
)
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale._transforms._base import OmeZarrAxes, OmeZarrAxis


def _ome_axes_with_unit(**units):
    return OmeZarrAxes((axis, OmeZarrAxis(name=axis, unit=unit)) for axis, unit in units.items())


class TestUnitValidation:
    @pytest.mark.parametrize(
        "source, target",
        [
            ("um", "um"),
            ("um", "micrometer"),
            ("micrometer", "um"),
            ("nm", "Nanometers"),
            (None, "um"),
            ("um", None),
            (None, None),
            ("", "um"),
            ("um", ""),
            ("custom", "custom"),
        ],
    )
    def test_accepts_matching_or_unspecified_units(self, source, target):
        require_matching_units_or_affine_relation(
            _ome_axes_with_unit(x=source),
            _ome_axes_with_unit(x=target),
            None,
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize(
        "relation",
        [
            Affine.identity("x"),
            Affine.from_factor({"x": 1000}),
            SpatialRelationSequence((Translation(x=1), SpatialRelationSequence((Affine.identity("x"),)))),
        ],
    )
    def test_accepts_differing_units_with_affine(self, relation):
        require_matching_units_or_affine_relation(
            _ome_axes_with_unit(x="um"),
            _ome_axes_with_unit(x="nm"),
            relation,
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize(
        "relation, source, target",
        [
            (AxisRenaming(x="X"), dict(x="um"), dict(X="micrometer")),
            (PermutationTo("yx"), dict(x="um", y="nm"), dict(y="nm", x="um")),
            (ProjectionTo("yx"), dict(x="um"), dict(y="nm", x="um")),
            (ProjectionTo("x"), dict(y="nm", x="um"), dict(x="um")),
            (
                SpatialRelationSequence((ProjectionTo("y"), ProjectionTo("yx"))),
                dict(y="um", x="um"),
                dict(y="um", x="nm"),
            ),
        ],
    )
    def test_follows_axis_dependencies(self, relation, source, target):
        require_matching_units_or_affine_relation(
            _ome_axes_with_unit(**source),
            _ome_axes_with_unit(**target),
            relation,
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize("relation", [None, Translation(x=1), PermutationTo("x"), ProjectionTo("x")])
    def test_rejects_differing_units_without_affine(self, relation):
        with pytest.raises(
            ValueError, match=r"source and target differ.*'x': 'um' vs 'nm'.*State the conversion explicitly.*Affine"
        ):
            require_matching_units_or_affine_relation(
                _ome_axes_with_unit(x="um"),
                _ome_axes_with_unit(x="nm"),
                relation,
                source_label="source",
                target_label="target",
            )

    def test_rejects_differing_unit_across_renaming(self):
        with pytest.raises(ValueError, match="'X': 'um' vs 'nm'"):
            require_matching_units_or_affine_relation(
                _ome_axes_with_unit(x="um"),
                _ome_axes_with_unit(X="nm"),
                AxisRenaming(x="X"),
                source_label="source",
                target_label="target",
            )

    def test_reports_all_conflicting_axes_in_target_order(self):
        with pytest.raises(ValueError, match=r"'y': 'nanometer' vs 'um', 'x': 'nanometer' vs 'um'"):
            require_matching_units_or_affine_relation(
                _ome_axes_with_unit(y="nanometer", x="nanometer"),
                _ome_axes_with_unit(y="um", x="um"),
                None,
                source_label="source",
                target_label="target",
            )

    def test_reordered_axes_do_not_hide_conflicts(self):
        with pytest.raises(ValueError, match="'y': 'nm' vs 'micron'"):
            require_matching_units_or_affine_relation(
                _ome_axes_with_unit(x="micron", y="nm"),
                _ome_axes_with_unit(y="micron", x="micron"),
                PermutationTo("yx"),
                source_label="source",
                target_label="target",
            )


class TestReconcileAxisPropParams:
    @staticmethod
    def reconcile_by_dependencies(target_axes, dependencies, source, provided, unit):
        """reconcile_axis_prop_params for relations that never scale.
        Nothing beyond the dependencies (links.sources) is ever needed here."""
        links = AxisLinks.from_dependencies(tuple(source.keys()), target_axes, dependencies)
        return reconcile_axis_prop_params(target_axes, links, source, provided, unit)

    def test_accepts_unit_when_source_blank(self):
        result = self.reconcile_by_dependencies(("x",), (("x",),), _ome_axes_with_unit(x=None), "infer", {"x": "um"})
        assert result["x"].unit == "um"

    def test_accepts_properties_and_units_for_inserted_axes(self):
        result = self.reconcile_by_dependencies(
            ("c", "x"),
            ((), ("x",)),
            _ome_axes_with_unit(x="um"),
            {"c": OmeZarrAxis(type="channel", discrete=True)},
            {"c": "index"},
        )
        assert result == OmeZarrAxes(
            {
                "c": OmeZarrAxis(name="c", type="channel", discrete=True, unit="index"),
                "x": OmeZarrAxis(name="x", type="space", unit="um"),
            }
        )

    def test_ignores_excess_properties_and_units(self):
        result = self.reconcile_by_dependencies(
            ("x",), (("x",),), _ome_axes_with_unit(x="um"), {"t": OmeZarrAxis(type="custom")}, {"z": "cm"}
        )
        assert result == OmeZarrAxes({"x": OmeZarrAxis(name="x", type="space", unit="um")})

    @pytest.mark.parametrize("provided, expected_type", [("infer", "channel"), (None, None)])
    def test_infers_only_inserted_axes_in_infer_mode(self, provided, expected_type):
        source = OmeZarrAxes(
            {
                "t": OmeZarrAxis(name="t", type="custom"),
                "x": OmeZarrAxis(name="x", type="custom-space"),
                "nonstandard": OmeZarrAxis(name="nonstandard"),
            }
        )
        result = self.reconcile_by_dependencies(
            ("c", "t", "x", "nonstandard"), ((), ("t",), ("x",), ("nonstandard",)), source, provided, None
        )
        assert result == OmeZarrAxes({"c": OmeZarrAxis(name="c", type=expected_type), **source})

    def test_provided_unit_overrides_inherited_unit_in_infer_mode(self):
        result = self.reconcile_by_dependencies(("x",), (("x",),), _ome_axes_with_unit(x="um"), "infer", {"x": "nm"})
        assert result["x"] == OmeZarrAxis(name="x", unit="nm")

    @pytest.mark.parametrize("provided_ome_axes", [None, "infer", {"X": OmeZarrAxis()}])
    def test_inherits_properties_along_dependencies(self, provided_ome_axes):
        source = OmeZarrAxes(
            {
                "z": OmeZarrAxis(name="z", unit="nm"),
                "x": OmeZarrAxis(name="x", type="space", unit="um", discrete=False, long_name="width"),
            }
        )
        result = self.reconcile_by_dependencies(("X",), (("x",),), source, provided_ome_axes, None)
        assert result == OmeZarrAxes(
            {"X": OmeZarrAxis(name="X", type="space", unit="um", discrete=False, long_name="width")}
        )

    @pytest.mark.parametrize("provided_unit", [None, {"x": "nm"}])
    def test_explicit_fields_override_and_blanks_inherit(self, provided_unit):
        source = OmeZarrAxes({"x": OmeZarrAxis(name="x", type="space", unit="um", discrete=False, long_name="width")})
        provided_ome_axes = {"x": OmeZarrAxis(type="custom", discrete=True)}
        result = self.reconcile_by_dependencies(("x",), (("x",),), source, provided_ome_axes, provided_unit)
        assert result["x"] == OmeZarrAxis(
            name="x", type="custom", unit="nm" if provided_unit else "um", discrete=True, long_name="width"
        )

    @pytest.mark.parametrize("unit", ["nanometer", "nm"])
    def test_accepts_equivalent_unit_parameters(self, unit):
        result = self.reconcile_by_dependencies(
            ("x",),
            (("x",),),
            _ome_axes_with_unit(x=None),
            {"x": OmeZarrAxis(type="space", unit="nanometer")},
            {"x": unit},
        )
        assert result["x"] == OmeZarrAxis(name="x", type="space", unit=unit)

    def test_rejects_conflicting_unit_parameters(self):
        with pytest.raises(ValueError, match="Conflicting unit between `unit` and `ome_zarr_axes`"):
            self.reconcile_by_dependencies(
                ("x",), (("x",),), _ome_axes_with_unit(x="um"), {"x": OmeZarrAxis(unit="um")}, {"x": "nm"}
            )
