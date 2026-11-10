from dataclasses import replace

import pytest

from clearscale import (
    Affine,
    Multiscale,
    Scale,
    Shape,
    Unit,
    ome_zarr,
    PixelSize,
    Translation,
    ProjectionTo,
    PermutationTo,
    AxisRearrangementTo,
    Factor,
)
from clearscale._services.axis_links import AxisLinks
from clearscale._spatial_relations import SpatialRelationSequence


class TestOmeZarrAxis:
    def test_repr_omits_unset_fields(self):
        axis = ome_zarr.Axis(type="channel")
        assert "type='channel'" in repr(axis)
        assert "unit" not in repr(axis)

    def test_to_ome_zarr_minimal(self):
        axis = ome_zarr.Axis(name="x")
        d = axis.to_ome_zarr(version="0.5")
        assert d == {"name": "x"}

    def test_to_ome_zarr_full_0_6(self):
        axis = ome_zarr.Axis(name="c", type="channel", unit="", discrete=True, long_name="Channel")
        d = axis.to_ome_zarr(version="0.6")
        assert d == {"name": "c", "type": "channel", "longName": "Channel", "discrete": True}

    @pytest.mark.parametrize("version", ["0.4", "0.5"])
    def test_to_ome_zarr_legacy_version_omits_incompatible_modern_fields(self, version):
        axis = ome_zarr.Axis(name="x", type="space", unit="micrometer", discrete=False, long_name="X axis")
        d = axis.to_ome_zarr(version=version)
        assert d == {"name": "x", "type": "space", "unit": "micrometer"}

    def test_from_ome_zarr_roundtrip(self):
        raw = {"name": "z", "type": "space", "unit": "micrometer", "discrete": False, "longName": "Z"}
        axis = ome_zarr.Axis.from_ome_zarr(raw)
        assert axis.name == "z"
        assert axis.type == "space"
        assert axis.unit == "micrometer"
        assert axis.discrete is False
        assert axis.long_name == "Z"
        assert axis.to_ome_zarr(version="0.6") == raw


class TestOmeZarrAxes:
    class AxKey:
        # Pattern used e.g. in bioimageio (`AxisId("x")`)
        def __init__(self, name: str = "key"):
            self._name = name

        def __str__(self):
            return self._name

    def test_fromkeys_produces_matching_names(self):
        axes = ome_zarr.Axes.fromkeys("zyx")
        assert list(axes.keys()) == ["z", "y", "x"]
        assert all(axes[a].name == a for a in axes)

    def test_fromkeys_handles_axis_key_object(self):
        key = self.AxKey()
        axes = ome_zarr.Axes.fromkeys([key])
        assert axes[key].name == str(key)

    def test_name_autofills_from_key(self):
        axes = ome_zarr.Axes({"x": ome_zarr.Axis(type="space")})
        assert axes["x"].name == "x"

    def test_name_autofills_from_axis_key_object(self):
        key = self.AxKey()
        axes = ome_zarr.Axes({key: ome_zarr.Axis(type="space")})
        assert axes[key].name == str(key)

    def test_name_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match its axis key"):
            ome_zarr.Axes({"x": ome_zarr.Axis(name="y")})

    def test_with_axes_infers_by_default(self):
        axes = ome_zarr.Axes({"x": ome_zarr.Axis(type="space")})
        expanded = axes.with_axes("cx")
        assert expanded["c"] == ome_zarr.Axis(name="c", type="channel")
        assert expanded["x"].type == "space"

    def test_with_axes_infer_false_turns_it_off(self):
        axes = ome_zarr.Axes({"x": ome_zarr.Axis(type="space")})
        expanded = axes.with_axes("cx", infer_inserted_types=False)
        assert expanded["c"] == ome_zarr.Axis(name="c")
        assert expanded["x"].type == "space"

    def test_with_axes_infers_type_only_for_insertions(self):
        x_properties = ome_zarr.Axis(name="x", long_name="some description")
        axes = ome_zarr.Axes({"x": x_properties})
        expanded = axes.with_axes("cx", infer_inserted_types=True)
        assert expanded["c"] == ome_zarr.Axis(name="c", type="channel")
        # Modifying existing props is the privilege of with_types_inferred
        assert expanded["x"] is x_properties, "existing axes must be untouched in with_axes regardless of .type"

    def test_with_types_inferred_sets_type(self):
        axes = ome_zarr.Axes.fromkeys("tczyx")
        inferred = axes.with_types_inferred()
        assert [inferred[a].type for a in "tczyx"] == ["time", "channel", "space", "space", "space"]

    def test_with_types_inferred_uses_str_of_axis_key_objects(self):
        key = self.AxKey("x")
        axes = ome_zarr.Axes.fromkeys([key])
        inferred = axes.with_types_inferred()
        assert "x" not in inferred, "sanity check against accidental str(key_obj) insertion"
        assert inferred[key].type == "space"

    def test_with_types_inferred_does_not_override_explicit_type(self):
        axes = ome_zarr.Axes({"c": ome_zarr.Axis(type="space", discrete=False)})
        inferred = axes.with_types_inferred()
        assert inferred["c"].type == "space"
        assert inferred["c"].discrete is False

    def test_with_types_inferred_does_not_raise_if_at_least_one_know_key(self):
        axes = ome_zarr.Axes.fromkeys(["row", "col", "x"])
        inferred = axes.with_types_inferred()
        assert inferred["x"].type == "space"
        assert inferred["row"].type is None
        assert inferred["col"].type is None

    def test_with_types_inferred_idempotent_after_all_typed(self):
        axes = ome_zarr.Axes.fromkeys("yx").with_types_inferred()
        assert axes.with_types_inferred() == axes

    @staticmethod
    def _carried(source, target_axes, dependencies, scales=None, untraced=()):
        """with_axes_carried_over, with links from the direct `dependencies`.
        `untraced`: Indices of targets with untraced coupling. Only relevant in context of sequences: simulates a target
        that mixes one of the sequence's original sources with an axis inserted partway along the sequence."""
        links = AxisLinks.from_dependencies(tuple(source.keys()), target_axes, dependencies, scales)
        if untraced:
            links = replace(
                links,
                links=tuple(replace(l, has_untraced_coupling=i in untraced) for i, l in enumerate(links.links)),
            )
        return source.with_axes_carried_over(target_axes, links)

    def test_with_axes_carried_over_keeps_only_agreeing_values_for_mixed_sources(self):
        source = ome_zarr.Axes(
            {
                "y": ome_zarr.Axis(name="y", type="space", discrete=False, long_name="rows"),
                "x": ome_zarr.Axis(name="x", type="space", discrete=True, long_name="columns"),
            }
        )
        result = self._carried(source, ("mixed", "renamed", "inserted"), (("y", "x"), ("y",), ()))
        assert result == ome_zarr.Axes(
            {
                "mixed": ome_zarr.Axis(name="mixed", type="space"),
                "renamed": ome_zarr.Axis(name="renamed", type="space", discrete=False, long_name="rows"),
                "inserted": ome_zarr.Axis(name="inserted"),
            }
        )

    @pytest.mark.parametrize("first, second", [("um", "um"), ("um", "micrometer"), ("um", "nm"), ("um", None)])
    def test_with_axes_carried_over_never_carries_units_to_mixed_targets(self, first, second):
        # Even if the sources agree, mixing may scale: Nothing says the target is in the same unit.
        source = ome_zarr.Axes(
            {
                "y": ome_zarr.Axis(name="y", type="space", unit=first),
                "x": ome_zarr.Axis(name="x", type="space", unit=second),
            }
        )
        result = self._carried(source, ("Y", "x"), (("y", "x"), ("x",)))
        assert result["Y"] == ome_zarr.Axis(name="Y", type="space")
        assert result["x"] == source["x"]

    @pytest.mark.parametrize(
        "scale, unit", [(1.0, "um"), (-1.0, "um"), (1.0 + 1e-15, "um"), (1000.0, None), (0.5, None)]
    )
    def test_with_axes_carried_over_carries_the_unit_of_exact_targets_only_if_the_scale_is_one(self, scale, unit):
        source = ome_zarr.Axes({"x": ome_zarr.Axis(name="x", type="space", unit="um", long_name="width")})
        result = self._carried(source, ("X",), (("x",),), scales=[scale])
        assert result["X"] == ome_zarr.Axis(name="X", type="space", unit=unit, long_name="width")

    def test_with_axes_carried_over_carries_nothing_to_mixed_targets_with_untraceable_axes(self):
        source = ome_zarr.Axes({"y": ome_zarr.Axis(name="y", type="space", long_name="rows", unit="um")})
        result = self._carried(source, ("Y",), (("y",),), scales=[None], untraced=(0,))
        assert result["Y"] == ome_zarr.Axis(name="Y")

    def test_with_axes_carried_over_carries_exact_targets_even_with_untraced_coupling(self):
        source = ome_zarr.Axes({"y": ome_zarr.Axis(name="y", type="space", long_name="rows", unit="um")})
        result = self._carried(source, ("Y",), (("y",),), untraced=(0,))
        assert result["Y"] == ome_zarr.Axis(name="Y", type="space", long_name="rows", unit="um")

    def test_with_axes_carried_over_rejects_incompatible_links(self):
        source = ome_zarr.Axes({"x": ome_zarr.Axis(name="x")})
        with pytest.raises(AssertionError, match="Received links for"):
            source.with_axes_carried_over(("a",), AxisLinks.identity("x"))


class TestScale:
    def test_default_ome_zarr_axes_has_type_inferred(self):
        s = Scale(shape=Shape(x=10))
        assert s.ome_zarr_axes["x"] == ome_zarr.Axis(name="x", type="space")

    def test_default_ome_zarr_axes_is_blank_for_nonstandard_keys(self):
        s = Scale(shape=Shape(w=10))
        assert s.ome_zarr_axes["w"] == ome_zarr.Axis(name="w")

    def test_ome_zarr_axes_none_does_not_infer_type_but_receives_name(self):
        s = Scale(shape=Shape(x=10), ome_zarr_axes=None)
        assert s.ome_zarr_axes["x"] == ome_zarr.Axis(name="x")

    def test_unit_also_populates_ome_zarr_axes_unit(self):
        s = Scale(shape=Shape(x=10), unit=Unit(x="micrometer"))
        assert s.ome_zarr_axes["x"].unit == "micrometer"

    def test_ome_zarr_axes_also_populates_unit(self):
        s = Scale(shape=Shape(x=10), ome_zarr_axes={"x": ome_zarr.Axis(unit="micrometer")})
        assert s.unit["x"] == "micrometer"

    def test_ome_zarr_axes_tolerates_missing_axes(self):
        s = Scale(
            shape=Shape(x=10, y=10),
            unit=Unit(x="micrometer", y="micrometer"),
            ome_zarr_axes={"x": ome_zarr.Axis(type="space", unit="micrometer")},  # y missing
        )
        assert list(s.ome_zarr_axes.keys()) == ["x", "y"]
        assert s.ome_zarr_axes["x"] == ome_zarr.Axis(name="x", type="space", unit="micrometer")
        assert s.ome_zarr_axes["y"] == ome_zarr.Axis(name="y", type="space", unit="micrometer")

    def test_both_consistent_merge_without_error(self):
        s = Scale(
            shape=Shape(x=10),
            unit=Unit(x="micrometer"),
            ome_zarr_axes={"x": ome_zarr.Axis(type="space")},
        )
        assert s.unit["x"] == "micrometer"
        assert s.ome_zarr_axes["x"].unit == "micrometer"
        assert s.ome_zarr_axes["x"].type == "space"

    def test_conflicting_unit_raises(self):
        with pytest.raises(ValueError, match="Conflicting unit"):
            Scale(
                shape=Shape(x=10),
                unit=Unit(x="micrometer"),
                ome_zarr_axes={"x": ome_zarr.Axis(unit="nanometer")},
            )

    def test_infer_ome_zarr_axes(self):
        s = Scale(shape=Shape(c=3, y=10, x=10))
        assert s.ome_zarr_axes["c"].type == "channel"
        assert s.ome_zarr_axes["y"].type == "space"

    def test_str_other_than_infer_raises(self):
        with pytest.raises(ValueError, match="ome_zarr_axes must be 'infer', None, or"):
            Scale(shape=Shape(row=2, col=2), ome_zarr_axes="wrongstring")  # type: ignore[reportArgumentType]

    def test_with_axes_default_infers_only_new_axis(self):
        s = Scale(shape=Shape(x=10), ome_zarr_axes=None)
        assert s.ome_zarr_axes["x"].type is None, "should not be set with ome_zarr_axes=None"
        expanded = s.with_axes("cx")
        assert expanded.ome_zarr_axes["c"].type == "channel"
        assert expanded.ome_zarr_axes["x"].type is None

    def test_with_axes_infer_false_does_not_infer_new_axis(self):
        s = Scale(shape=Shape(x=10))
        expanded = s.with_axes("cx", infer_inserted_types=False)
        assert expanded.ome_zarr_axes["c"].type is None

    def test_scale_equality_considers_ome_zarr_axes(self):
        s1 = Scale(shape=Shape(x=10), ome_zarr_axes=None)
        s2 = Scale(shape=Shape(x=10), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        assert s1 != s2


class TestScaleFromLists:
    def test_minimal_keys_only_gives_singleton_shape(self):
        s = Scale.from_lists("yx")
        assert s.shape == Shape(y=1, x=1)

    def test_all_params(self):
        s = Scale.from_lists(
            keys="yx",
            shape=[512, 512],
            pixel_size=[0.25, 0.25],
            unit=["micrometer", "micrometer"],
            translation=[1.0, 2.0],
        )
        assert s.shape == Shape(y=512, x=512)
        assert s.pixel_size == PixelSize(y=0.25, x=0.25)
        assert s.unit == Unit(y="micrometer", x="micrometer")
        assert s.translation == Translation(y=1.0, x=2.0)
        assert s.ome_zarr_axes == {
            "y": ome_zarr.Axis(name="y", type="space", unit="micrometer"),
            "x": ome_zarr.Axis(name="x", type="space", unit="micrometer"),
        }

    def test_ome_zarr_axes_as_raw_dicts(self):
        s = Scale.from_lists(
            keys="yx",
            ome_zarr_axes=[{"type": "space", "unit": "micrometer"}, {"type": "space", "unit": "micrometer"}],
        )
        assert s.ome_zarr_axes["y"].type == "space"
        assert s.unit["y"] == "micrometer"

    def test_ome_zarr_axes_as_sole_param(self):
        # Valid OME-Zarr axes meta
        axes_json = [
            {"name": "t", "type": "time", "unit": "sec"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "mm"},
            {"name": "y", "type": "space", "unit": "mm"},
            {"name": "x", "type": "space", "unit": "mm"},
        ]
        s = Scale.from_lists(ome_zarr_axes=axes_json)
        inferred = Scale.from_lists("tczyx", unit=["sec", "", "mm", "mm", "mm"])
        assert s == inferred

    def test_ome_zarr_axes_as_objects(self):
        s = Scale.from_lists(
            keys="yx",
            ome_zarr_axes=[ome_zarr.Axis(type="space"), ome_zarr.Axis(type="space")],
        )
        assert s.ome_zarr_axes["x"].type == "space"

    def test_ome_zarr_axes_mixed_dicts_and_objects(self):
        s = Scale.from_lists(
            keys="cx",
            ome_zarr_axes=[ome_zarr.Axis(type="channel"), {"type": "space"}],
        )
        assert s.ome_zarr_axes["c"].type == "channel"
        assert s.ome_zarr_axes["x"].type == "space"

    def test_per_axis_name_in_dict_must_match_key(self):
        with pytest.raises(ValueError, match="does not match its axis key"):
            Scale.from_lists(keys="yx", ome_zarr_axes=[{"name": "wrong"}, {}])

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one axis key"):
            Scale.from_lists(keys=[])

    def test_duplicate_keys_raise(self):
        with pytest.raises(ValueError, match="unique axis keys"):
            Scale.from_lists(keys="xx", shape=[10, 10])

    @pytest.mark.parametrize("param", ["shape", "pixel_size", "unit", "translation"])
    def test_mismatched_length_raises(self, param):
        kwargs = {param: [1]}  # length 1, but keys has length 2
        with pytest.raises(ValueError, match="has length"):
            Scale.from_lists(keys="yx", **kwargs)  # type: ignore[reportArgumentType]

    def test_mismatched_length_ome_zarr_axes_raises(self):
        with pytest.raises(ValueError, match="length"):
            Scale.from_lists(keys="yx", ome_zarr_axes=[{"type": "space"}])


class TestMultiscale:
    def test_unit_and_ome_zarr_axes_properties_reflect_scales(self):
        ms = Multiscale(
            {
                "s0": Scale(shape=Shape(x=100, y=100), unit=Unit(x="micrometer", y="micrometer")),
                "s1": Scale(shape=Shape(x=50, y=50)),  # blank unit
            }
        )
        assert ms.unit == Unit(x="micrometer", y="micrometer")
        assert ms.ome_zarr_axes["x"].unit == "micrometer"
        assert ms["s0"].ome_zarr_axes is ms["s1"].ome_zarr_axes, "scales should share same instance for ome axes"

    def test_to_ome_zarr_output_matches_unit_property(self):
        ms = Multiscale({"s0": Scale(shape=Shape(y=100, x=100), unit=Unit(x="micrometer"))})
        serialized = ms.to_ome_zarr(version="0.5")
        serialized_unit = serialized["axes"][1]["unit"]
        assert ms.unit["x"] == "micrometer", "sanity check"
        assert serialized_unit == ms.unit["x"]

    def test_complementary_axis_info_merges_without_conflict(self):
        s0 = Scale(shape=Shape(x=100, y=100), unit=Unit(x="micrometer"))
        s1 = Scale(shape=Shape(x=50, y=50), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        ms = Multiscale({"s0": s0, "s1": s1})
        assert ms.ome_zarr_axes["x"].unit == "micrometer"
        assert ms.ome_zarr_axes["x"].type == "space"

    def test_conflicting_unit_across_scales_raises(self):
        s0 = Scale(shape=Shape(x=100, y=100), unit=Unit(x="micrometer"))
        s1 = Scale(shape=Shape(x=50, y=50), unit=Unit(x="nanometer"))
        with pytest.raises(ValueError, match="Conflicting 'unit'"):
            Multiscale({"s0": s0, "s1": s1})

    def test_conflicting_type_across_scales_raises(self):
        s0 = Scale(shape=Shape(x=100, y=100), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        s1 = Scale(shape=Shape(x=50, y=50), ome_zarr_axes={"x": ome_zarr.Axis(type="channel")})
        with pytest.raises(ValueError, match="Conflicting 'type'"):
            Multiscale({"s0": s0, "s1": s1})

    def test_conflicting_other_axis_property_across_scales_raises(self):
        s0 = Scale(shape=Shape(x=100, y=100), ome_zarr_axes={"x": ome_zarr.Axis(long_name="cool")})
        s1 = Scale(shape=Shape(x=50, y=50), ome_zarr_axes={"x": ome_zarr.Axis(long_name="story")})
        with pytest.raises(ValueError, match="Conflicting 'long_name'"):
            Multiscale({"s0": s0, "s1": s1})

    def test_merge_is_order_independent(self):
        s_with_unit = Scale(shape=Shape(x=100, y=100), unit=Unit(x="micrometer"))
        s_with_type = Scale(shape=Shape(x=100, y=100), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        ms_a = Multiscale({"s0": s_with_unit, "s1": s_with_type})
        ms_b = Multiscale({"s0": s_with_type, "s1": s_with_unit})
        assert ms_a.ome_zarr_axes["x"] == ms_b.ome_zarr_axes["x"]


class TestMultiscaleAxisOrderValidation:
    def test_init_with_bad_order_infer_raises(self):
        s0 = Scale(shape=Shape(y=2, x=2, c=2))
        with pytest.raises(ValueError, match="time, then channel, before all others"):
            Multiscale({"s0": s0})

    def test_init_with_bad_order_explicit_axes_raises(self):
        s0 = Scale(
            shape=Shape(y=2, x=2, c=2),
            ome_zarr_axes=dict(
                y=ome_zarr.Axis(type="space"), x=ome_zarr.Axis(type="space"), c=ome_zarr.Axis(type="channel")
            ),
        )
        with pytest.raises(ValueError, match="time, then channel, before all others"):
            Multiscale({"s0": s0})

    def test_init_with_good_order_does_not_raise(self):
        s0 = Scale(shape=dict(zip("tcyx", [1, 3, 10, 10])))
        _ = Multiscale({"s0": s0})

    def test_bad_axis_order_is_rejected_when_reading(self):
        bad_multiscale_dict = {
            "version": "0.5",
            "axes": [
                {"name": "x", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "c", "type": "channel"},
            ],
            "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
        }
        with pytest.raises(ValueError, match="time, then channel, before all others"):
            Multiscale.from_ome_zarr(bad_multiscale_dict, shape_source=lambda _: (10, 10, 3))

    def test_bad_axis_order_is_rejected_in_additional_coordinate_systems(self):
        ms = Multiscale({"s0": Scale(shape=Shape(c=2, y=2, x=2))})
        with pytest.raises(ValueError, match="time, then channel.*coordinate system 'w'"):
            ms.with_coordinate_system("w", reached_by=PermutationTo("yxc"))


class TestMultiscaleTransfer:
    def test_with_coordinate_system_uses_reconciled_properties_for_relation_target(self):
        """Smoke test; full coverage in TestReconcileAxisPropParams"""
        source = Multiscale.from_single(
            Scale(
                Shape(z=4, y=8, x=8),
                ome_zarr_axes={
                    "z": ome_zarr.Axis(type="space", unit="um"),
                    "y": ome_zarr.Axis(type="space", unit="um", long_name="rows"),
                    "x": ome_zarr.Axis(type="space", unit="um"),
                },
            )
        )
        result = source.with_coordinate_system(
            "world",
            reached_by=[ProjectionTo("yx"), PermutationTo("xy"), ProjectionTo("cxy")],
            unit={"c": "index"},
            ome_zarr_axes={"c": ome_zarr.Axis(type="channel", discrete=True), "y": ome_zarr.Axis(long_name="height")},
        )
        assert result.ome_zarr_axes == source.ome_zarr_axes, "shouldn't change the multiscale itself"
        assert result.coordinate_system_ome_zarr_axes("world") == ome_zarr.Axes(
            {
                "c": ome_zarr.Axis(name="c", type="channel", discrete=True, unit="index"),
                "x": ome_zarr.Axis(name="x", type="space", unit="um"),
                "y": ome_zarr.Axis(name="y", type="space", unit="um", long_name="height"),
            }
        )

    def test_as_derived_from_fills_missing_axis_properties_from_parent(self):
        parent = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=100, y=100),
                    ome_zarr_axes={
                        "x": ome_zarr.Axis(type="space", unit="micrometer", discrete=True, long_name="position"),
                        "y": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )
        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=50, y=50),
                    ome_zarr_axes={"x": ome_zarr.Axis(type="space"), "y": ome_zarr.Axis(type="space")},
                )
            }
        )

        result = child.as_derived_from(parent)

        assert result.ome_zarr_axes["x"].type == "space"
        assert result.ome_zarr_axes["x"].unit == "micrometer"
        assert result.ome_zarr_axes["x"].discrete is True
        assert result.ome_zarr_axes["x"].long_name == "position"

    def test_as_derived_from_preserves_child_axis_properties(self):
        parent = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=100, y=100, z=100),
                    ome_zarr_axes={
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                        "y": ome_zarr.Axis(type="space"),
                        "z": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )
        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=50, y=50, z=50),
                    ome_zarr_axes={
                        "x": ome_zarr.Axis(type="channel", unit="nanometer"),
                        "y": ome_zarr.Axis(type="space"),
                        "z": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )

        # along x: nm -> um = 0.001
        result = child.as_derived_from(parent, reached_by=Affine.from_factor({"x": 0.001, "y": 1.0, "z": 1.0}))

        assert result.ome_zarr_axes["x"].type == "channel"
        assert result.ome_zarr_axes["x"].unit == "nanometer"

    def test_as_derived_from_does_not_restore_dropped_axes(self):
        parent = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(c=3, x=100, y=100),
                    ome_zarr_axes={
                        "c": ome_zarr.Axis(type="channel"),
                        "x": ome_zarr.Axis(type="space"),
                        "y": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )
        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=100, y=100),
                    ome_zarr_axes={"x": ome_zarr.Axis(type="space"), "y": ome_zarr.Axis(type="space")},
                )
            }
        )

        result = child.as_derived_from(parent, reached_by=AxisRearrangementTo("cxy"))

        assert set(result.ome_zarr_axes) == {"x", "y"}

    def test_as_derived_from_leaves_inserted_child_axes_unchanged(self):
        parent = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=100, y=100),
                    ome_zarr_axes={"x": ome_zarr.Axis(type="space"), "y": ome_zarr.Axis(type="space")},
                )
            }
        )
        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(c=3, z=100, y=100, x=100),
                    ome_zarr_axes={
                        "c": ome_zarr.Axis(type="channel"),
                        "z": ome_zarr.Axis(type="space"),
                        "y": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(),
                    },
                )
            }
        )

        result = child.as_derived_from(parent, reached_by=AxisRearrangementTo("xy"))

        assert result.ome_zarr_axes["c"].type == "channel"
        assert result.ome_zarr_axes["x"].type == "space"

    def test_as_derived_from_only_merges_matching_axis_keys(self):
        parent = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=100, y=100),
                    ome_zarr_axes={
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                        "y": ome_zarr.Axis(type="space", unit="micrometer"),
                    },
                )
            }
        )
        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(c=1, z=50, x=50),
                    ome_zarr_axes={
                        "c": ome_zarr.Axis(type="channel"),
                        "z": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )

        result = child.as_derived_from(parent, reached_by=AxisRearrangementTo("xy"))

        assert result.ome_zarr_axes["x"].unit == "micrometer"
        assert "y" not in result.ome_zarr_axes

    def test_as_derived_from_accepts_conflicting_axis_properties(self):
        # It *shouldn't* be accepted.
        # Conflicting axis properties indicate that the axis by the same ID is actually a different axis on the child.
        # This should be specified as a drop-reinsert relation as in the test below.
        # But SpatialRelation can't track axis provenance yet (tbd).
        parent = Multiscale({"s0": Scale(shape=Shape(c=3, y=1, x=1))})
        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(c=3, y=1, x=1),
                    ome_zarr_axes={"c": ome_zarr.Axis(type="label")}
                    | {"y": ome_zarr.Axis(type="space"), "x": ome_zarr.Axis(type="space")},
                )
            }
        )

        # with pytest.raises(ValueError, match="Conflicting type for axis 'c'"):  # Should raise but doesn't yet. Document actual behaviour instead.
        result = child.as_derived_from(parent)
        assert result.ome_zarr_axes["c"].type == "label"

    def test_as_derived_from_allows_conflicting_axis_properties_when_axis_is_dropped_and_reinserted(self):
        parent = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(c=3, y=100, x=100),
                    ome_zarr_axes={
                        "c": ome_zarr.Axis(type="channel"),
                        "y": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )

        child = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(c=3, y=100, x=100),
                    ome_zarr_axes={
                        "c": ome_zarr.Axis(type="label"),
                        "y": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )

        # The result's c-axis can now be traced as being a different c-axis than the parent's
        result = child.as_derived_from(parent, reached_by=[ProjectionTo(("y", "x")), ProjectionTo(("c", "y", "x"))])

        assert result.ome_zarr_axes["c"].type == "label"
        assert result.ome_zarr_axes["x"].type == "space"

    @staticmethod
    def _transfer_long_names(matrix, long_names, axes="zyx"):
        """as_derived_from with an Affine, with long_names set on the source only. Returns derived long_names."""
        source = Multiscale.from_single(
            Scale(
                Shape({a: 4 for a in axes}),
                ome_zarr_axes={a: ome_zarr.Axis(type="space", long_name=n) for a, n in zip(axes, long_names)},
            )
        )
        derived = Multiscale.from_single(Scale(Shape({a: 4 for a in axes})))
        result = derived.as_derived_from(source, reached_by=Affine.from_array(matrix, axes))
        return tuple(result.ome_zarr_axes[a].long_name for a in axes)

    def test_as_derived_from_carries_affine_properties_only_where_mixed_axes_agree(self):
        # z' = z + y, y' = y + x, x' = x: z and y are mixed, so nothing carries over if the block disagrees...
        triangular = ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0))
        # ... except to x, which is exactly x', however much it also feeds the mixed y'.
        assert self._transfer_long_names(triangular, ("shared", "shared", "different")) == (None, None, "different")
        # If the block agrees, so does everything it is mixed with
        assert self._transfer_long_names(triangular, ("same", "same", "same")) == ("same", "same", "same")

    def test_as_derived_from_does_not_mix_properties_across_unconnected_affine_blocks(self):
        # y/x are mixed (a 45 degree rotation, unnormalised), z is on its own
        rotation = ((1, 0, 0, 0), (0, 1, 1, 0), (0, -1, 1, 0))
        assert self._transfer_long_names(rotation, ("depth", "plane", "plane")) == ("depth", "plane", "plane")
        assert self._transfer_long_names(rotation, ("depth", "plane", "other")) == ("depth", None, None)

    def test_as_derived_from_carries_properties_one_to_one_along_permuting_affines(self):
        # y' = x, x' = y: Nothing is mixed, so each derived axis inherits from the source axis it maps to
        permuting = ((0, 1, 0), (1, 0, 0))
        assert self._transfer_long_names(permuting, ("rows", "columns"), axes="yx") == ("columns", "rows")

    def test_singular_affine_can_carry_derivation_properties(self):
        source = Multiscale.from_single(
            Scale(Shape(y=4, x=4), ome_zarr_axes={a: ome_zarr.Axis(type="space", long_name="plane") for a in "yx"})
        )
        derived = Multiscale.from_single(Scale(Shape(y=4, x=4)))
        singular = Affine.from_array(((1, 2, 0), (2, 4, 0)), "yx")
        result = derived.as_derived_from(source, reached_by=singular)
        assert [result.ome_zarr_axes[a].long_name for a in "yx"] == ["plane", "plane"]

    def test_singular_affine_with_collapsed_axis_carries_nothing_for_that_axis(self):
        # z is collapsed, so no target depends on it
        source = Multiscale.from_single(
            Scale(
                Shape(z=4, y=4, x=4),
                ome_zarr_axes={a: ome_zarr.Axis(type="space", long_name=a.upper()) for a in "zyx"},
            )
        )
        derived = Multiscale.from_single(Scale(Shape(z=4, y=4, x=4)))
        collapsing = Affine.from_array(((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)), "zyx")
        result = derived.as_derived_from(source, reached_by=collapsing)
        assert [result.ome_zarr_axes[a].long_name for a in "zyx"] == [None, "Y", "X"]


def test_precomputed_axis_types_are_hardcoded():
    info = {
        "num_channels": 2,
        "scales": [
            {"key": "s0", "size": [100, 200, 300], "resolution": [1.0, 2.0, 3.0]},
        ],
    }
    ms = Multiscale.from_precomputed(info)
    assert ms.ome_zarr_axes["c"].type == "channel"
    assert ms.ome_zarr_axes["z"].type == "space"


class TestScaleWithUnitReplaced:
    @staticmethod
    def _scale():
        return Scale(
            shape=Shape(z=4, y=8, x=8),
            pixel_size=PixelSize(z=2.0, y=0.5, x=0.5),
            unit=Unit(z="micrometer", y="micrometer", x="micrometer"),
            translation=Translation(z=1.0, y=2.0, x=3.0),
        )

    def test_replaces_all_units_and_keeps_everything_else(self):
        scale = self._scale()

        result = scale.with_unit_replaced(Unit(z="nanometer", y="nanometer", x="nanometer"))

        assert result.unit == Unit(z="nanometer", y="nanometer", x="nanometer")
        assert result.shape == scale.shape
        assert result.pixel_size == scale.pixel_size
        assert result.translation == scale.translation

    def test_partial_mapping_replaces_only_given_axes(self):
        result = self._scale().with_unit_replaced({"z": "nanometer"})

        assert result.unit == Unit(z="nanometer", y="micrometer", x="micrometer")

    def test_keeps_ome_zarr_axes_in_sync(self):
        scale = self._scale()

        result = scale.with_unit_replaced({"x": "nanometer"})

        assert result.ome_zarr_axes["x"].unit == "nanometer"
        assert result.ome_zarr_axes["x"].type == scale.ome_zarr_axes["x"].type
        assert result.ome_zarr_axes["y"].unit == "micrometer"

    def test_empty_string_makes_unit_unspecified(self):
        result = self._scale().with_unit_replaced({"z": ""})

        assert result.unit["z"] == ""
        assert result.ome_zarr_axes["z"].unit is None

    def test_does_not_modify_the_original(self):
        scale = self._scale()

        scale.with_unit_replaced({"z": "nanometer"})

        assert scale.unit["z"] == "micrometer"

    def test_rejects_axes_the_scale_does_not_have(self):
        with pytest.raises(ValueError, match=r"Cannot replace unit of axes \['t'\]"):
            self._scale().with_unit_replaced({"t": "second"})

    def test_plain_replace_conflicts_with_ome_zarr_axes(self):
        """Why with_unit_replaced exists"""
        with pytest.raises(ValueError, match="Conflicting unit"):
            replace(self._scale(), unit=Unit(z="nanometer", y="nanometer", x="nanometer"))


class TestScaleWithUnitConverted:
    @staticmethod
    def _scale():
        return Scale(
            shape=Shape(z=4, y=8, x=8),
            pixel_size=PixelSize(z=2.0, y=0.5, x=0.5),
            unit=Unit(z="micrometer", y="micrometer", x="micrometer"),
            translation=Translation(z=1.0, y=2.0, x=3.0),
        )

    def test_converts_units_and_updates_other_physical_properties(self):
        scale = self._scale()

        # axiswise and recognises other spellings (mm instead of millimeter)
        result = scale.with_unit_converted(Unit(z="nanometer", y="mm", x="micrometer"))

        assert result.unit == Unit(z="nanometer", y="mm", x="micrometer")
        assert result.shape == scale.shape

        # 1 micron * 1000 nm/micron = 1000 nm (Factor multiplies pixel size)
        expected_factor = Factor(zip("zyx", [1000.0, 0.001, 1.0]))
        assert result.pixel_size == scale.pixel_size * expected_factor
        assert result.pixel_size == PixelSize(z=2000.0, y=0.0005, x=0.5)
        assert result.translation == scale.translation * expected_factor
        assert result.translation == Translation(z=1000.0, y=0.002, x=3.0)

    def test_other_spelling_of_the_same_unit_only_relabels(self):
        converted = self._scale().with_unit_converted({"x": "um"})
        replaced = self._scale().with_unit_replaced({"x": "um"})

        assert converted == replaced

    def test_converts_time_axes(self):
        scale = Scale(
            shape=Shape(t=3, y=8, x=8),
            pixel_size=PixelSize(t=0.5, y=0.5, x=0.5),
            unit=Unit(t="second", y="micrometer", x="micrometer"),
            translation=Translation(t=2.0, y=0.0, x=0.0),
        )

        result = scale.with_unit_converted({"t": "millisecond"})

        assert result.pixel_size["t"] == 500.0
        assert result.translation["t"] == 2000.0
        assert result.unit["t"] == "millisecond"

    def test_converting_back_restores_the_scale(self):
        scale = self._scale()
        nm = Unit(z="nanometer", y="nanometer", x="nanometer")
        um = Unit(z="micrometer", y="micrometer", x="micrometer")
        assert scale.unit == um, "starting with um"

        result = scale.with_unit_converted(nm).with_unit_converted(um)

        assert result is not scale
        assert result == scale

    def test_partial_mapping_converts_only_given_axes(self):
        scale = self._scale()

        result = scale.with_unit_converted({"z": "nanometer"})

        assert result.unit == Unit(z="nanometer", y="micrometer", x="micrometer")
        expected_factor = Factor(z=1000.0)
        assert result.pixel_size == scale.pixel_size * expected_factor
        assert result.translation == scale.translation * expected_factor

    def test_keeps_ome_zarr_axes_in_sync(self):
        scale = self._scale()

        result = scale.with_unit_converted({"x": "nanometer"})

        assert result.ome_zarr_axes["x"].unit == "nanometer"
        assert result.ome_zarr_axes["x"].type == scale.ome_zarr_axes["x"].type
        assert result.ome_zarr_axes["y"].unit == "micrometer"

    def test_does_not_modify_the_original(self):
        scale = self._scale()

        scale.with_unit_converted({"z": "nanometer"})

        assert scale.unit["z"] == "micrometer"

    def test_empty_conversion_is_noop(self):
        scale = Scale(shape=Shape(y=8, x=8), unit=Unit(y="", x=""))

        assert scale.with_unit_converted({"x": ""}) == scale

    def test_unrecognised_unit_conversion_is_noop(self):
        scale = Scale(shape=Shape(y=8, x=8), unit=Unit(y="a.u.", x="a.u."))

        assert scale.with_unit_converted({"x": "a.u."}) == scale

    def test_rejects_conversion_to_unspecified(self):
        with pytest.raises(ValueError, match="Cannot convert.*from 'micrometer' to ''"):
            self._scale().with_unit_converted({"z": ""})

    def test_rejects_conversion_from_unspecified(self):
        scale = Scale(shape=Shape(z=8, x=8), unit=Unit(z="", x=""))

        with pytest.raises(ValueError, match="Cannot convert.*from '' to 'nanometer'"):
            scale.with_unit_converted({"z": "nanometer"})

    def test_rejects_axes_the_scale_does_not_have(self):
        with pytest.raises(ValueError, match=r"Cannot convert unit of axes \['t'\]"):
            self._scale().with_unit_converted({"t": "second"})

    def test_rejects_unrecognised_unit(self):
        with pytest.raises(ValueError, match="Axis 'x': Unknown unit 'furlong'"):
            self._scale().with_unit_converted({"x": "furlong"})

    def test_rejects_space_to_time(self):
        with pytest.raises(ValueError, match=r"Cannot convert between 'micrometer' \(space\) and 'second' \(time\)"):
            self._scale().with_unit_converted({"x": "second"})

    def test_rejects_empty_arg(self):
        with pytest.raises(ValueError, match="Empty Unit"):
            self._scale().with_unit_converted({})

    def test_converted_multiscale_relates_to_the_original_by_the_unit_conversion_docs_example(self):
        """Converting a Scale is the same physical space in other numbers, so the two Multiscales connect through exactly the unit conversion."""
        original_base = Scale(
            shape=Shape(zip("zyx", (23, 256, 256))),
            pixel_size={"z": 220, "y": 10, "x": 10},
            unit={"z": "um", "y": "nm", "x": "nm"},
        )
        original_ms = Multiscale.from_single(original_base, scale_key="s0")

        converted_base = original_base.with_unit_converted({"z": "nm"})
        converted_ms = Multiscale.from_single(converted_base, scale_key="s0")

        conversion_relation = Affine.from_unit_conversion(converted_ms.unit, original_ms.unit)

        derived = converted_ms.as_derived_from(original_ms, reached_by=conversion_relation)

        scale = derived.to_ome_zarr(version="0.6")["coordinateTransformations"][0]["scale"]
        assert scale == [0.001, 1.0, 1.0]


class TestUnitSpellings:
    @staticmethod
    def _axes(unit):
        return {"y": ome_zarr.Axis(type="space", unit=unit), "x": ome_zarr.Axis(type="space", unit=unit)}

    def test_scale_accepts_unit_and_ome_zarr_axes_with_different_spellings(self):
        scale = Scale(shape=Shape(y=4, x=4), unit=Unit(y="nm", x="nm"), ome_zarr_axes=self._axes("nanometer"))

        assert scale.unit == Unit(y="nm", x="nm")

    def test_scale_still_rejects_different_units_given_by_alias(self):
        with pytest.raises(ValueError, match="Conflicting unit"):
            Scale(shape=Shape(y=4, x=4), unit=Unit(y="um", x="um"), ome_zarr_axes=self._axes("nanometer"))

    def test_multiscale_merges_scales_with_different_spellings_keeping_the_first(self):
        first = Scale(shape=Shape(y=8, x=8), unit=Unit(y="nm", x="nm"), ome_zarr_axes="infer")
        second = Scale(shape=Shape(y=4, x=4), unit=Unit(y="nanometer", x="nanometer"), ome_zarr_axes="infer")

        ms = Multiscale({"s0": first, "s1": second})

        assert ms.unit == Unit(y="nm", x="nm")
        assert ms["s1"].unit == Unit(y="nm", x="nm")

    def test_multiscale_rejects_scales_with_different_units_given_by_alias(self):
        first = Scale(shape=Shape(y=8, x=8), unit=Unit(y="nm", x="nm"), ome_zarr_axes="infer")
        second = Scale(shape=Shape(y=4, x=4), unit=Unit(y="um", x="um"), ome_zarr_axes="infer")

        with pytest.raises(ValueError, match="Conflicting 'unit'"):
            Multiscale({"s0": first, "s1": second})


class TestAxisPropertyCarryoverThroughAffines:
    """What a relation proves about axes decides what carries over:
    Unit only follows an axis that is exactly another one at scale +-1.
    Other properties follow at any scale for one-to-one, or the agreement of all axes it is mixed with."""

    @staticmethod
    def _space_ms(axes, **props):
        """Multiscale of space axes (some tests need valid ones), with the given Axis kwargs per axis."""
        return Multiscale.from_single(
            Scale(
                Shape({a: 4 for a in axes}),
                ome_zarr_axes={a: ome_zarr.Axis(**{"type": "space", **props.get(a, {})}) for a in axes},
            )
        )

    @staticmethod
    def _props(axes, *fields):
        return {a: tuple(getattr(axes[a], f) for f in fields) for a in axes}

    def test_with_coordinate_system_does_not_carry_units_through_scaling(self):
        source = self._space_ms("yx", y={"unit": "micrometer"}, x={"unit": "micrometer"})
        scaling = Affine.from_factor({"y": 1.0, "x": 1000.0})

        result = source.with_coordinate_system("nm", reached_by=scaling)

        axes = result.coordinate_system_ome_zarr_axes("nm")
        assert self._props(axes, "type", "unit") == {"y": ("space", "micrometer"), "x": ("space", None)}

    def test_with_coordinate_system_accepts_the_unit_of_a_scaled_axis_explicitly(self):
        source = self._space_ms("yx", y={"unit": "micrometer"}, x={"unit": "micrometer"})
        scaling = Affine.from_factor({"y": 1.0, "x": 1000.0})

        result = source.with_coordinate_system("nm", reached_by=scaling, unit={"x": "nanometer"})

        assert result.coordinate_system_ome_zarr_axes("nm")["x"].unit == "nanometer"

    def test_as_derived_from_does_not_carry_units_back_through_scaling(self):
        parent = self._space_ms("yx", y={"unit": "nanometer"}, x={"unit": "nanometer"})
        child = Multiscale.from_single(Scale(Shape(y=4, x=4), ome_zarr_axes="infer"))
        scaling = Affine.from_factor({"y": 1.0, "x": 1000.0})

        result = child.as_derived_from(parent, reached_by=scaling)

        assert self._props(result.ome_zarr_axes, "unit") == {"y": ("nanometer",), "x": (None,)}

    def test_as_derived_from_carries_units_back_through_flips_and_swaps(self):
        parent = self._space_ms("yx", y={"unit": "um"}, x={"unit": "nm"})
        child = Multiscale.from_single(Scale(Shape(y=4, x=4), ome_zarr_axes="infer"))
        flip_and_swap = Affine.from_array(((0, -1, 0), (1, 0, 0)), "yx")  # y' = -x, x' = y

        result = child.as_derived_from(parent, reached_by=flip_and_swap)

        assert self._props(result.ome_zarr_axes, "unit") == {"y": ("nm",), "x": ("um",)}

    def test_shear_carries_everything_to_the_axis_that_stays_exact_and_only_agreement_to_the_other(self):
        # x' = x + y, y' = y
        source = self._space_ms("yx", y={"unit": "um", "long_name": "rows"}, x={"unit": "um", "long_name": "columns"})
        shear = Affine.from_array(((1, 0, 0), (1, 1, 0)), "yx")

        result = source.with_coordinate_system("sheared", reached_by=shear)

        axes = result.coordinate_system_ome_zarr_axes("sheared")
        assert self._props(axes, "type", "unit", "long_name") == {
            "y": ("space", "um", "rows"),
            "x": ("space", None, None),
        }

    def test_as_derived_from_does_not_label_an_axis_with_what_was_mixed_into_a_dropped_one(self):
        parent = self._space_ms("zx", z={"long_name": "Depth"}, x={"long_name": "Left-Right"})
        mixing = Affine.from_array(((1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0)), "zyx")  # y' = z + y
        child = self._space_ms("zyx")

        plain_drop = child.as_derived_from(parent, reached_by=ProjectionTo("zx"))
        mix_then_drop = child.as_derived_from(parent, reached_by=SpatialRelationSequence((mixing, ProjectionTo("zx"))))

        expected = {"z": ("Depth",), "y": (None,), "x": ("Left-Right",)}
        assert self._props(plain_drop.ome_zarr_axes, "long_name") == expected
        assert self._props(mix_then_drop.ome_zarr_axes, "long_name") == expected

    def test_with_coordinate_system_does_not_carry_properties_to_axes_mixed_with_an_inserted_one(self):
        source = self._space_ms("yx", y={"long_name": "Anterior-Posterior"}, x={"long_name": "Left-Right"})
        mixing = Affine.from_array(((1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0)), "zyx")  # y' = z + y
        relation = SpatialRelationSequence((ProjectionTo("zyx"), mixing))

        result = source.with_coordinate_system("tilted", reached_by=relation)

        axes = result.coordinate_system_ome_zarr_axes("tilted")
        assert self._props(axes, "long_name") == {"z": (None,), "y": (None,), "x": ("Left-Right",)}
