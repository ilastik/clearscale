import pytest

from clearscale import Multiscale, Scale, Shape, Unit, ome_zarr


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
        d = axis.to_ome_zarr(version="0.6.rc0")
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
        assert axis.to_ome_zarr(version="0.6.rc0") == raw


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

    def test_with_axes_inserts_blank_for_new_axes(self):
        axes = ome_zarr.Axes({"x": ome_zarr.Axis(type="space")})
        expanded = axes.with_axes("cx")
        assert expanded["c"] == ome_zarr.Axis(name="c")
        assert expanded["x"].type == "space"

    def test_with_axes_infers_type_and_discrete_only_for_insertions(self):
        x_properties = ome_zarr.Axis(name="x", long_name="some description")
        axes = ome_zarr.Axes({"x": x_properties})
        expanded = axes.with_axes("cx", infer_inserted_types=True)
        assert expanded["c"] == ome_zarr.Axis(name="c", type="channel", discrete=True)
        # Modifying existing props is the privilege of with_types_inferred
        assert expanded["x"] is x_properties, "existing axes must be untouched in with_axes regardless of .type"

    def test_with_types_inferred_sets_type_and_discrete(self):
        axes = ome_zarr.Axes.fromkeys("tczyx")
        inferred = axes.with_types_inferred()
        assert inferred["t"].type == "time" and inferred["t"].discrete is False
        assert inferred["c"].type == "channel" and inferred["c"].discrete is True
        assert inferred["y"].type == "space" and inferred["y"].discrete is False

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

    def test_with_types_inferred_raises_for_unknown_keys(self):
        # Raise because: Why are you asking for type inference when you are not using the standard keys at all?
        axes = ome_zarr.Axes.fromkeys(["row", "col"])
        with pytest.raises(ValueError, match="none of"):
            axes.with_types_inferred()

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
    def test_default_ome_zarr_axes_is_blank(self):
        s = Scale(shape=Shape(x=10))
        assert s.ome_zarr_axes["x"] == ome_zarr.Axis(name="x")

    def test_unit_also_populates_ome_zarr_axes_unit(self):
        s = Scale(shape=Shape(x=10), unit=Unit(x="micrometer"))
        assert s.ome_zarr_axes["x"].unit == "micrometer"

    def test_ome_zarr_axes_also_populates_unit(self):
        s = Scale(shape=Shape(x=10), ome_zarr_axes={"x": ome_zarr.Axis(unit="micrometer")})
        assert s.unit["x"] == "micrometer"

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

    def test_mismatched_axis_keys_between_unit_and_ome_zarr_axes_raises_cleanly(self):
        # Should raise informative ValueError, not default KeyError
        with pytest.raises(ValueError, match="Incompatible axes/order"):
            Scale(
                shape=Shape(x=10, y=10),
                unit=Unit(x="micrometer", y="micrometer"),
                ome_zarr_axes={"x": ome_zarr.Axis(unit="micrometer")},
            )

    def test_infer_ome_zarr_axes(self):
        s = Scale(shape=Shape(c=3, y=10, x=10), ome_zarr_axes="infer")
        assert s.ome_zarr_axes["c"].type == "channel"
        assert s.ome_zarr_axes["y"].type == "space"

    def test_infer_raises_for_unrecognized_axes(self):
        with pytest.raises(ValueError, match="Cannot infer OME-Zarr axis types"):
            Scale(shape=Shape(row=2, col=2), ome_zarr_axes="infer")

    def test_str_other_than_infer_raises(self):
        with pytest.raises(ValueError, match="ome_zarr_axes must be either"):
            Scale(shape=Shape(row=2, col=2), ome_zarr_axes="wrongstring")  # type: ignore[reportArgumentType]

    def test_with_axes_default_does_not_infer_new_axis(self):
        s = Scale(shape=Shape(x=10), ome_zarr_axes="infer")
        expanded = s.with_axes("cx")
        assert expanded.ome_zarr_axes["c"].type is None

    def test_with_axes_infer_inserted_types_only_infers_new_axis(self):
        s = Scale(shape=Shape(x=10))
        expanded = s.with_axes("cx", infer_inserted_types=True)
        assert expanded.ome_zarr_axes["c"].type == "channel"
        assert expanded.ome_zarr_axes["x"].type is None

    def test_scale_equality_considers_ome_zarr_axes(self):
        s1 = Scale(shape=Shape(x=10))
        s2 = Scale(shape=Shape(x=10), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        assert s1 != s2


class TestMultiscale:
    def test_unit_and_ome_zarr_axes_properties_reflect_scales(self):
        ms = Multiscale(
            {
                "s0": Scale(shape=Shape(x=100), unit=Unit(x="micrometer")),
                "s1": Scale(shape=Shape(x=50), unit=Unit(x="micrometer")),
            }
        )
        assert ms.unit == Unit(x="micrometer")
        assert ms.ome_zarr_axes["x"].unit == "micrometer"
        assert ms["s0"].ome_zarr_axes is ms["s1"].ome_zarr_axes

    def test_to_ome_zarr_output_matches_unit_property(self):
        ms = Multiscale({"s0": Scale(shape=Shape(x=100), unit=Unit(x="micrometer"))})
        serialized = ms.to_ome_zarr(version="0.5")
        serialized_unit = serialized["axes"][0]["unit"]
        assert ms.unit["x"] == "micrometer", "sanity check"
        assert serialized_unit == ms.unit["x"]

    def test_all_scales_share_canonical_unit_and_axes_instances(self):
        ms = Multiscale(
            {
                "s0": Scale(shape=Shape(x=100), unit=Unit(x="micrometer")),
                "s1": Scale(shape=Shape(x=50)),  # blank unit
            }
        )
        assert ms["s0"].unit == ms["s1"].unit == Unit(x="micrometer")
        assert ms["s0"].ome_zarr_axes == ms["s1"].ome_zarr_axes

    def test_complementary_axis_info_merges_without_conflict(self):
        s0 = Scale(shape=Shape(x=100), unit=Unit(x="micrometer"))
        s1 = Scale(shape=Shape(x=50), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        ms = Multiscale({"s0": s0, "s1": s1})
        assert ms.ome_zarr_axes["x"].unit == "micrometer"
        assert ms.ome_zarr_axes["x"].type == "space"

    def test_conflicting_unit_across_scales_raises(self):
        s0 = Scale(shape=Shape(x=100), unit=Unit(x="micrometer"))
        s1 = Scale(shape=Shape(x=50), unit=Unit(x="nanometer"))
        with pytest.raises(ValueError, match="Conflicting 'unit'"):
            Multiscale({"s0": s0, "s1": s1})

    def test_conflicting_type_across_scales_raises(self):
        s0 = Scale(shape=Shape(x=100), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        s1 = Scale(shape=Shape(x=50), ome_zarr_axes={"x": ome_zarr.Axis(type="channel")})
        with pytest.raises(ValueError, match="Conflicting 'type'"):
            Multiscale({"s0": s0, "s1": s1})

    def test_conflicting_other_axis_property_across_scales_raises(self):
        s0 = Scale(shape=Shape(x=100), ome_zarr_axes={"x": ome_zarr.Axis(long_name="cool")})
        s1 = Scale(shape=Shape(x=50), ome_zarr_axes={"x": ome_zarr.Axis(long_name="story")})
        with pytest.raises(ValueError, match="Conflicting 'long_name'"):
            Multiscale({"s0": s0, "s1": s1})

    def test_merge_is_order_independent(self):
        s_with_unit = Scale(shape=Shape(x=100), unit=Unit(x="micrometer"))
        s_with_type = Scale(shape=Shape(x=50), ome_zarr_axes={"x": ome_zarr.Axis(type="space")})
        ms_a = Multiscale({"s0": s_with_unit, "s1": s_with_type})
        ms_b = Multiscale({"s0": s_with_type, "s1": s_with_unit})
        assert ms_a.ome_zarr_axes["x"] == ms_b.ome_zarr_axes["x"]


class TestMultiscaleAxisOrderValidation:
    def test_init_with_bad_order_untyped_does_not_raise(self):
        # space-channel type order would be invalid OME-Zarr,
        # but here there is no type info available to validate
        s0 = Scale(shape=Shape(x=2, c=2))
        _ = Multiscale({"s0": s0})

    def test_init_with_bad_order_infer_raises(self):
        # adding the ome_zarr_axes param explicitly opts in to OME-Zarr rules
        s0 = Scale(shape=Shape(x=2, c=2), ome_zarr_axes="infer")
        with pytest.raises(ValueError, match="axes must be ordered time-channel-others"):
            Multiscale({"s0": s0})

    def test_init_with_bad_order_explicit_axes_raises(self):
        s0 = Scale(
            shape=Shape(x=2, c=2),
            ome_zarr_axes=dict(x=ome_zarr.Axis(type="space"), c=ome_zarr.Axis(type="channel")),
        )
        with pytest.raises(ValueError, match="axes must be ordered time-channel-others"):
            Multiscale({"s0": s0})

    def test_init_with_good_order_does_not_raise(self):
        s0 = Scale(shape=dict(zip("tcyx", [1, 3, 10, 10])), ome_zarr_axes="infer")
        _ = Multiscale({"s0": s0})

    def test_bad_axis_order_round_trips(self, recwarn):
        # Handle existing metadata permissively
        bad_multiscale_dict = {
            "version": "0.5",
            "axes": [
                {"name": "x", "type": "space"},
                {"name": "c", "type": "channel"},
            ],
            "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
        }
        ms = Multiscale.from_ome_zarr(bad_multiscale_dict, shape_source=lambda _: (3, 10))
        assert ms.ome_zarr_axes["c"].type == "channel"

        result = ms.to_ome_zarr(version="0.5")
        assert result == bad_multiscale_dict
        assert len(recwarn) == 0


def test_precomputed_axis_semantics_are_hardcoded():
    info = {
        "num_channels": 2,
        "scales": [
            {"key": "s0", "size": [100, 200, 300], "resolution": [1.0, 2.0, 3.0]},
        ],
    }
    ms = Multiscale.from_precomputed(info)
    assert ms.ome_zarr_axes["c"].type == "channel"
    assert ms.ome_zarr_axes["c"].discrete is True
    assert ms.ome_zarr_axes["z"].type == "space"
    assert ms.ome_zarr_axes["z"].discrete is False
