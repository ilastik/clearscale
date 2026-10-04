import pytest

from clearscale import (
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
)


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
    def test_with_coordinate_system_inherits_axis_properties(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(x=100, y=200),
                    ome_zarr_axes={
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                        "y": ome_zarr.Axis(type="space", unit="micrometer"),
                    },
                )
            }
        )

        ms2 = ms.with_coordinate_system("world")

        cs = ms2.coordinate_system_ome_zarr_axes("world")
        assert cs == ms.ome_zarr_axes

    def test_with_coordinate_system_inherits_axis_properties_for_shared_axes_only(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(z=10, y=200, x=100),
                    ome_zarr_axes={
                        "z": ome_zarr.Axis(type="space", unit="micrometer"),
                        "y": ome_zarr.Axis(type="space", unit="micrometer"),
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                    },
                )
            }
        )

        ms2 = ms.with_coordinate_system("world", reached_by=AxisRearrangementTo(("y", "x")))

        cs = ms2.coordinate_system_ome_zarr_axes("world")
        assert set(cs) == {"y", "x"}
        assert cs["x"] == ms.ome_zarr_axes["x"]
        assert cs["y"] == ms.ome_zarr_axes["y"]

    def test_with_coordinate_system_overrides_inherited_axis_properties(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(z=5, y=10, x=100),
                    ome_zarr_axes={
                        "z": ome_zarr.Axis(type="space"),
                        "y": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                    },
                )
            }
        )

        ms2 = ms.with_coordinate_system(
            "world", reached_by=AxisRearrangementTo(("z", "y", "x")), ome_zarr_axes={"x": ome_zarr.Axis(type="custom")}
        )

        axis = ms2.coordinate_system_ome_zarr_axes("world")["x"]
        assert axis.type == "custom"  # overridden
        assert axis.unit == "micrometer"  # inherited

    def test_with_coordinate_system_overrides_inherited_axis_properties_from_unit(self):
        ms = Multiscale(
            {
                "s0": Scale(shape=Shape(y=10, x=100), unit=Unit(x="micrometer")),
            }
        )

        ms2 = ms.with_coordinate_system(
            "world",
            reached_by=AxisRearrangementTo(("y", "x")),
            unit=Unit(x="nanometer"),
        )

        axis = ms2.coordinate_system_ome_zarr_axes("world")["x"]
        assert axis.unit == "nanometer"

    def test_with_coordinate_system_merges_unit_and_axis_properties(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(z=5, y=10, x=100),
                    ome_zarr_axes={
                        "z": ome_zarr.Axis(type="space"),
                        "y": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                    },
                )
            }
        )

        ms2 = ms.with_coordinate_system(
            "world",
            reached_by=AxisRearrangementTo(("z", "y", "x")),
            unit={"x": "nm"},
            ome_zarr_axes={"x": ome_zarr.Axis(type="custom")},
        )

        axis = ms2.coordinate_system_ome_zarr_axes("world")["x"]
        assert axis == ome_zarr.Axis(name="x", type="custom", unit="nm")  # both overridden

    def test_with_coordinate_system_rejects_conflicting_unit_arguments(self):
        ms = Multiscale(
            {
                "s0": Scale(shape=Shape(x=100, y=100), unit=Unit(x="micrometer")),
            }
        )

        with pytest.raises(ValueError, match="Conflicting unit"):
            ms.with_coordinate_system(
                "world",
                reached_by=AxisRearrangementTo(("x",)),
                unit=Unit(x="nanometer"),
                ome_zarr_axes={"x": ome_zarr.Axis(unit="micrometer")},
            )

    def test_with_coordinate_system_default_infers_only_inserted_axes_type(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(nonstandard=5, t=4, x=100, y=100),
                    ome_zarr_axes={
                        "nonstandard": ome_zarr.Axis(),
                        "t": ome_zarr.Axis(type="nonstandard-type"),
                        "x": ome_zarr.Axis(type="space", unit="micrometer"),
                        "y": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )
        assert ms["s0"].ome_zarr_axes["nonstandard"] == ome_zarr.Axis(name="nonstandard")
        assert ms["s0"].ome_zarr_axes["t"] == ome_zarr.Axis(name="t", type="nonstandard-type")
        assert ms["s0"].ome_zarr_axes["x"] == ome_zarr.Axis(name="x", type="space", unit="micrometer")

        ms2 = ms.with_coordinate_system(
            "world",
            reached_by=AxisRearrangementTo(("c", "nonstandard", "t", "y", "x")),
        )

        cs = ms2.coordinate_system_ome_zarr_axes("world")

        assert cs["nonstandard"] == ome_zarr.Axis(name="nonstandard"), "should be unchanged"
        assert cs["t"] == ome_zarr.Axis(name="t", type="nonstandard-type"), "should be unchanged"
        assert cs["x"] == ome_zarr.Axis(name="x", type="space", unit="micrometer"), "should be unchanged"
        assert cs["c"] == ome_zarr.Axis(name="c", type="channel"), "type should be inferred"

    def test_with_coordinate_system_accepts_properties_for_inserted_axes(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(y=10, x=100),
                    ome_zarr_axes={"y": ome_zarr.Axis(type="space"), "x": ome_zarr.Axis(type="space")},
                )
            }
        )

        ms2 = ms.with_coordinate_system(
            "world",
            reached_by=AxisRearrangementTo(("c", "y", "x")),
            ome_zarr_axes={"c": ome_zarr.Axis(type="channel", discrete=True)},
        )

        cs = ms2.coordinate_system_ome_zarr_axes("world")

        assert cs["x"].type == "space"
        assert cs["c"] == ome_zarr.Axis(name="c", type="channel", discrete=True)

    def test_with_coordinate_system_accepts_unit_for_inserted_axes(self):
        ms = Multiscale({"s0": Scale(shape=Shape(x=100, y=100))})

        ms2 = ms.with_coordinate_system("world", reached_by=AxisRearrangementTo(("c", "y", "x")), unit=Unit(c="index"))

        cs = ms2.coordinate_system_ome_zarr_axes("world")

        assert cs["c"].unit == "index"
        assert cs["x"].unit is None

    def test_with_coordinate_system_accepts_matching_unit_and_ome_zarr_axes(self):
        ms = Multiscale({"s0": Scale(shape=Shape(y=10, x=100))})

        ms2 = ms.with_coordinate_system(
            "world",
            reached_by=AxisRearrangementTo(("y", "x")),
            unit=Unit(x="nanometer"),
            ome_zarr_axes={"x": ome_zarr.Axis(unit="nanometer", type="space")},
        )

        axis = ms2.coordinate_system_ome_zarr_axes("world")["x"]

        assert axis.unit == "nanometer"
        assert axis.type == "space"

    def test_with_coordinate_system_ignores_excess_properties_and_unit(self):
        ms = Multiscale({"s0": Scale(shape=Shape(x=100, y=100))})

        ms2 = ms.with_coordinate_system(
            "world",
            reached_by=AxisRearrangementTo(("c", "y", "x")),
            unit=Unit(z="cm"),
            ome_zarr_axes={"t": ome_zarr.Axis(type="fun")},
        )

        cs = ms2.coordinate_system_ome_zarr_axes("world")

        assert "z" not in cs
        assert "t" not in cs

    def test_with_coordinate_system_infers_for_inserted_axes_only(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(z=5, y=10, x=100),
                    ome_zarr_axes={
                        "z": ome_zarr.Axis(type="custom"),
                        "y": ome_zarr.Axis(type="space"),
                        "x": ome_zarr.Axis(type="space"),
                    },
                )
            }
        )

        ms2 = ms.with_coordinate_system("world", reached_by=AxisRearrangementTo(("c", "z", "y", "x")))

        cs = ms2.coordinate_system_ome_zarr_axes("world")

        assert cs["c"].type == "channel"
        assert cs["y"].type == "space"  # inherited
        assert cs["z"].type == "custom"  # existing axis: not inferred (would be "space")

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

        result = child.as_derived_from(parent)

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

        result = child.as_derived_from(parent, by=AxisRearrangementTo(("x", "y")))

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

        result = child.as_derived_from(parent, by=AxisRearrangementTo(("c", "z", "y", "x")))

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

        result = child.as_derived_from(parent, by=AxisRearrangementTo(("c", "z", "x")))

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
        result = child.as_derived_from(parent, by=[ProjectionTo(("y", "x")), ProjectionTo(("c", "y", "x"))])

        assert result.ome_zarr_axes["c"].type == "label"
        assert result.ome_zarr_axes["x"].type == "space"


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
