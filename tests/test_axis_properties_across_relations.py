import pytest

from clearscale import Affine, AxisRenaming, PermutationTo, ProjectionTo, SameSpace, Translation
from clearscale._services.axis_links import AxisLinks
from clearscale._services.axis_properties_across_relations import (
    require_consistent_axis_properties,
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
        require_consistent_axis_properties(
            _ome_axes_with_unit(x=source),
            _ome_axes_with_unit(x=target),
            SameSpace(),
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize(
        "relation",
        [
            Affine.from_factor({"x": 1000}),
            Affine.from_array(((-1000, 0),), "x"),  # flip as well
            Affine.from_unit_conversion({"x": "um"}, {"x": "nm"}),
            SpatialRelationSequence((Translation(x=1), SpatialRelationSequence((Affine.from_factor({"x": 1000}),)))),
            SpatialRelationSequence((Affine.from_factor({"x": 2000}), Affine.from_factor({"x": 0.5}))),
        ],
    )
    def test_accepts_differing_units_with_the_scale_that_converts_them(self, relation):
        require_consistent_axis_properties(
            _ome_axes_with_unit(x="um"),
            _ome_axes_with_unit(x="nm"),
            relation,
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize(
        "relation, message",
        [
            (Affine.from_factor({"x": 100}), r"scaled by 100, but um to nm is 1000"),
            (Affine.from_factor({"x": 0.001}), r"scaled by 0.001, but um to nm is 1000"),  # the wrong way around
            (Affine.from_factor({"x": 0.5}), r"scaled by 0.5, but um to nm is 1000"),
            (
                SpatialRelationSequence((Affine.from_factor({"x": 1000}), Affine.from_factor({"x": 2}))),
                r"scaled by 2000, but um to nm is 1000",
            ),
        ],
    )
    def test_rejects_differing_units_with_a_scale_that_does_not_convert_them(self, relation, message):
        with pytest.raises(ValueError, match=rf"'x': 'um' vs 'nm' \({message}\).*leave one of the units unspecified"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(x="um"),
                _ome_axes_with_unit(x="nm"),
                relation,
                source_label="source",
                target_label="target",
            )

    @pytest.mark.parametrize(
        "source, target, factor",
        [("um", "nm", 1000), ("nm", "um", 0.001), ("mm", "um", 1000), ("micron", "nanometer", 1000), ("s", "ms", 1000)],
    )
    def test_accepts_what_from_unit_conversion_produces(self, source, target, factor):
        relation = Affine.from_unit_conversion({"x": source}, {"x": target})
        require_consistent_axis_properties(
            _ome_axes_with_unit(x=source), _ome_axes_with_unit(x=target), relation, source_label="s", target_label="t"
        )
        with pytest.raises(ValueError, match="scaled by"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(x=source),
                _ome_axes_with_unit(x=target),
                Affine.from_factor({"x": factor * 3}),
                source_label="s",
                target_label="t",
            )

    @pytest.mark.parametrize(
        "source, target, scale",
        [
            ("um", "um", 2.0),  # registration to an atlas can include real scaling
            ("um", "micrometer", 0.5),
            ("custom", "custom", 3.0),
            ("custom", "other", 3.0),  # unknown units cannot be checked
            ("um", "s", 3.0),  # cannot be converted: That is the type's business
            ("um", None, 3.0),  # unspecified
            (None, "nm", 3.0),
        ],
    )
    def test_does_not_check_scales_it_cannot_judge(self, source, target, scale):
        require_consistent_axis_properties(
            _ome_axes_with_unit(x=source),
            _ome_axes_with_unit(x=target),
            Affine.from_factor({"x": scale}),
            source_label="source",
            target_label="target",
        )

    def test_does_not_check_mixed_axes(self):
        rotation = Affine.from_array(((1, 1, 0), (-1, 1, 0)), "yx")
        require_consistent_axis_properties(
            _ome_axes_with_unit(y="um", x="um"),
            _ome_axes_with_unit(y="nm", x="nm"),
            rotation,
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize(
        "relation",
        [
            Affine.identity("x"),
            Affine.from_array(((-1, 0),), "x"),  # flip
            SpatialRelationSequence((Translation(x=1), SpatialRelationSequence((Affine.identity("x"),)))),
            SpatialRelationSequence((Affine.from_factor({"x": 1000}), Affine.from_factor({"x": 0.001}))),
        ],
    )
    def test_rejects_differing_units_with_affine_that_does_not_scale(self, relation):
        with pytest.raises(ValueError, match=r"source and target differ.*'x': 'um' vs 'nm'.*Affine"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(x="um"),
                _ome_axes_with_unit(x="nm"),
                relation,
                source_label="source",
                target_label="target",
            )

    def test_rejects_differing_units_with_affine_that_does_not_scale_per_axis(self):
        # Same as above but 2D. x is scaled by the Affine, so we can't know if the unit mismatch is correct or not.
        # y is identity so its mismatch needs to be flagged.
        with pytest.raises(ValueError, match=r"differ \('y': 'um' vs 'nm'\)"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(y="um", x="um"),
                _ome_axes_with_unit(y="nm", x="nm"),
                Affine.from_factor({"y": 1, "x": 1000}),
                source_label="source",
                target_label="target",
            )

    def test_accepts_differing_units_for_mixed_axes_but_not_for_the_exact_one(self):
        shear = Affine.from_array(((1, 0, 0), (1, 1, 0)), "yx")  # y' = y, x' = y + x
        require_consistent_axis_properties(
            _ome_axes_with_unit(y="um", x="um"),
            _ome_axes_with_unit(y="um", x="nm"),
            shear,
            source_label="source",
            target_label="target",
        )
        with pytest.raises(ValueError, match=r"differ \('y': 'um' vs 'nm'\)"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(y="um", x="um"),
                _ome_axes_with_unit(y="nm", x="nm"),
                shear,
                source_label="source",
                target_label="target",
            )

    def test_follows_exact_links_through_permuting_affine(self):
        swap = Affine.from_array(((0, 1, 0), (1, 0, 0)), "yx")  # y' = x, x' = y
        require_consistent_axis_properties(
            _ome_axes_with_unit(y="um", x="nm"),
            _ome_axes_with_unit(y="nm", x="um"),
            swap,
            source_label="source",
            target_label="target",
        )
        with pytest.raises(ValueError, match=r"differ \('y': 'nm' vs 'um', 'x': 'um' vs 'nm'\)"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(y="um", x="nm"),
                _ome_axes_with_unit(y="um", x="nm"),
                swap,
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
        require_consistent_axis_properties(
            _ome_axes_with_unit(**source),
            _ome_axes_with_unit(**target),
            relation,
            source_label="source",
            target_label="target",
        )

    @pytest.mark.parametrize("relation", [SameSpace(), Translation(x=1), PermutationTo("x"), ProjectionTo("x")])
    def test_rejects_differing_units_without_affine(self, relation):
        with pytest.raises(
            ValueError, match=r"source and target differ.*'x': 'um' vs 'nm'.*State the conversion explicitly.*Affine"
        ):
            require_consistent_axis_properties(
                _ome_axes_with_unit(x="um"),
                _ome_axes_with_unit(x="nm"),
                relation,
                source_label="source",
                target_label="target",
            )

    def test_rejects_differing_unit_across_renaming(self):
        with pytest.raises(ValueError, match="'X': 'um' vs 'nm'"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(x="um"),
                _ome_axes_with_unit(X="nm"),
                AxisRenaming(x="X"),
                source_label="source",
                target_label="target",
            )

    def test_reports_all_conflicting_axes_in_target_order(self):
        with pytest.raises(ValueError, match=r"'y': 'nanometer' vs 'um', 'x': 'nanometer' vs 'um'"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(y="nanometer", x="nanometer"),
                _ome_axes_with_unit(y="um", x="um"),
                SameSpace(),
                source_label="source",
                target_label="target",
            )

    def test_reordered_axes_do_not_hide_conflicts(self):
        with pytest.raises(ValueError, match="'y': 'nm' vs 'micron'"):
            require_consistent_axis_properties(
                _ome_axes_with_unit(x="micron", y="nm"),
                _ome_axes_with_unit(y="micron", x="micron"),
                PermutationTo("yx"),
                source_label="source",
                target_label="target",
            )


class TestTypeValidation:
    @staticmethod
    def _ome_axes_with_type(**types):
        return OmeZarrAxes((axis, OmeZarrAxis(name=axis, type=t)) for axis, t in types.items())

    @pytest.mark.parametrize(
        "source, target",
        [("space", "space"), ("time", "time"), (None, "space"), ("space", None), (None, None), ("custom", "custom")],
    )
    def test_accepts_matching_or_unspecified_types(self, source, target):
        require_consistent_axis_properties(
            self._ome_axes_with_type(x=source),
            self._ome_axes_with_type(x=target),
            SameSpace(),
            source_label="a",
            target_label="b",
        )

    @pytest.mark.parametrize(
        "relation",
        [
            SameSpace(),
            Translation(x=1),
            Affine.identity("x"),
            Affine.from_factor({"x": 1000}),  # scaling does not change the type
            Affine.from_array(((-3, 0),), "x"),
            SpatialRelationSequence((Affine.from_factor({"x": 1000}), Affine.from_factor({"x": 0.001}))),
        ],
    )
    def test_rejects_different_types_for_an_axis_that_is_kept_or_scaled(self, relation):
        with pytest.raises(
            ValueError, match=r"types of source and target differ \('x': 'space' vs 'time'\).*cannot change its type"
        ):
            require_consistent_axis_properties(
                self._ome_axes_with_type(x="space"),
                self._ome_axes_with_type(x="time"),
                relation,
                source_label="source",
                target_label="target",
            )

    def test_rejects_different_types_across_renaming_and_permutation(self):
        with pytest.raises(ValueError, match=r"\('X': 'space' vs 'time'\)"):
            require_consistent_axis_properties(
                self._ome_axes_with_type(x="space"),
                self._ome_axes_with_type(X="time"),
                AxisRenaming(x="X"),
                source_label="source",
                target_label="target",
            )
        with pytest.raises(ValueError, match=r"\('y': 'channel' vs 'space'\)"):
            require_consistent_axis_properties(
                self._ome_axes_with_type(x="space", y="channel"),
                self._ome_axes_with_type(y="space", x="space"),
                PermutationTo("yx"),
                source_label="source",
                target_label="target",
            )

    def test_does_not_check_mixed_axes_or_inserted_ones(self):
        # Mixing space and time is what a shear along time is
        shear = Affine.from_array(((1, 0, 0), (1, 1, 0)), "yx")  # y' = y, x' = y + x
        require_consistent_axis_properties(
            self._ome_axes_with_type(y="space", x="space"),
            self._ome_axes_with_type(y="space", x="time"),
            shear,
            source_label="source",
            target_label="target",
        )
        # Inserted axis does not correspond to any source
        require_consistent_axis_properties(
            self._ome_axes_with_type(x="space"),
            self._ome_axes_with_type(c="channel", x="space"),
            ProjectionTo("cx"),
            source_label="source",
            target_label="target",
        )

    def test_reports_the_first_property_that_conflicts(self):
        axes_a = OmeZarrAxes({"x": OmeZarrAxis(name="x", type="space", unit="um")})
        axes_b = OmeZarrAxes({"x": OmeZarrAxis(name="x", type="time", unit="nm")})
        with pytest.raises(ValueError, match="units of a and b differ"):
            require_consistent_axis_properties(axes_a, axes_b, SameSpace(), source_label="a", target_label="b")


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
