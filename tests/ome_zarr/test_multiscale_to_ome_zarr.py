import pytest
from clearscale import ome_zarr
from clearscale._axis_values import PixelSize, Shape, Unit, Factor, Translation
from clearscale._multiscale import BlueprintShapes, Multiscale, Scale
from clearscale._services.ome_zarr import SUPPORTED_OME_ZARR_VERSIONS_WRITE
from clearscale._spatial_relations import AxisRearrangementTo


def _multiscale(axes, size=4, pixel_size=None):
    shape = Shape(zip(axes, [size] * len(axes)))
    ps = PixelSize(zip(axes, pixel_size)) if pixel_size else None
    return Multiscale({"s0": Scale(shape=shape, pixel_size=ps, ome_zarr_axes="infer")})


@pytest.mark.parametrize("version", SUPPORTED_OME_ZARR_VERSIONS_WRITE)
def test_serializes_ome_properties(version):
    ms = _multiscale("zyx")
    ms.ome.name = "my-image"
    ms.ome.type = "gaussian"
    ms.ome.metadata = {"method": "skimage.transform.pyramid_gaussian", "version": "0.22", "kwargs": {"sigma": 1.5}}

    written = ms.to_ome_zarr(version=version)
    assert written["name"] == ms.ome.name
    assert written["type"] == ms.ome.type
    assert written["metadata"] == ms.ome.metadata


def _axis_names(result):
    cs = result
    if "coordinateSystems" in result:
        assert len(result["coordinateSystems"]) == 1, "expected exactly 1 coordinate system"
        cs = result["coordinateSystems"][0]
    return [ax["name"] for ax in cs["axes"]]


@pytest.mark.parametrize("version", SUPPORTED_OME_ZARR_VERSIONS_WRITE)
def test_serializes_axes_in_order(version):
    scale = Scale(
        shape=Shape(y=10, x=10),
        ome_zarr_axes={"x": ome_zarr.Axis(type="space"), "y": ome_zarr.Axis(type="space", long_name="row")},
    )
    ms = Multiscale.from_single(scale)
    assert ms.axes == ("y", "x")
    written = ms.to_ome_zarr(version=version)
    assert _axis_names(written) == ["y", "x"]


def _global_scale(result):
    scale_dicts = [t for t in result.get("coordinateTransformations", []) if t["type"] == "scale"]
    return scale_dicts[0]["scale"]


def _global_scale_and_translation(result):
    translation_dicts = [t for t in result.get("coordinateTransformations", []) if t["type"] == "translation"]
    return _global_scale(result), translation_dicts[0]["translation"] if translation_dicts else None


def _dataset_scale(result, key="s0"):
    dataset = next(d for d in result["datasets"] if d["path"] == key)
    (scale_dict,) = [t for t in dataset["coordinateTransformations"] if t["type"] == "scale"]
    return scale_dict["scale"]


class TestCoordinateSystemToLegacy:
    @pytest.mark.parametrize(
        "relation, expected_global_scale_and_translation",
        [
            (Factor(y=2, x=2), ([0.5, 0.5], None)),
            (Translation(y=3, x=4), ([1.0, 1.0], [-3.0, -4.0])),
            ([Factor(y=2, x=2), Translation(y=3, x=4)], ([0.5, 0.5], [-3.0, -4.0])),
            ([Translation(y=3, x=4), Factor(y=2, x=2)], ([0.5, 0.5], [-1.5, -2.0])),
            ([AxisRearrangementTo("zyx"), Translation(z=0, y=3, x=4)], ([1.0, 1.0], [-3.0, -4.0])),
            ([Translation(y=3, x=4), AxisRearrangementTo("cx")], ([1.0, 1.0], [0.0, -4.0])),
        ],
    )
    def test_serializes_compatible_coordinate_system_as_legacy_transforms(
        self, relation, expected_global_scale_and_translation
    ):
        """Axis rearrangements are not strictly compatible with legacy OME-Zarr versions.
        We accommodate them when the rest of the relation *is* compatible (scale[+translation]),
        by rearranging the relation's values directly."""
        ms = _multiscale("yx").with_coordinate_system("world", reached_by=relation)

        result = ms.to_ome_zarr(version="0.4")

        assert "coordinateTransformations" in result
        assert _global_scale_and_translation(result) == expected_global_scale_and_translation

    def test_serializes_coordinate_system_connected_by_reverse_drop(self):
        source = _multiscale("zyx")
        # The source--drop-->derived connection can only be stored in reverse
        # (normally external systems on `derived` are connected as `derived --t--> system`)
        derived_ms = _multiscale("yx").as_derived_from(source, by=[AxisRearrangementTo("yx"), Translation(y=3, x=4)])
        expected_global_scale_and_translation = ([1.0, 1.0], [3.0, 4.0])

        result = derived_ms.to_ome_zarr(version="0.4")

        assert "coordinateTransformations" in result
        assert _global_scale_and_translation(result) == expected_global_scale_and_translation

    def test_does_not_serialize_compatible_coordinate_system_with_identity(self):
        ms = _multiscale("yx").with_coordinate_system("world", reached_by=None)  # None = identity

        result = ms.to_ome_zarr(version="0.4")

        # Writing a nondescript identity on the global level doesn't contribute anything useful
        assert "coordinateTransformations" not in result


class TestLegacyGlobalTScale:
    def test_synthesizes_global_t_scale_when_uniform(self):
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(t=4, z=2, y=2, x=2),
                    pixel_size=PixelSize(t=0.5, z=0.3, y=20.0, x=30.0),
                    ome_zarr_axes="infer",
                ),
                "s1": Scale(
                    shape=Shape(t=4, z=1, y=1, x=1),
                    pixel_size=PixelSize(t=0.5, z=0.6, y=40.0, x=60.0),
                    ome_zarr_axes="infer",
                ),
            },
            _legacy_convention_global_t_scale=0.5,
        )

        result = ms.to_ome_zarr(version="0.4")

        assert "coordinateTransformations" in result
        assert _global_scale(result) == [0.5, 1.0, 1.0, 1.0]
        assert _dataset_scale(result, "s0") == [1.0, 0.3, 20.0, 30.0]
        assert _dataset_scale(result, "s1") == [1.0, 0.6, 40.0, 60.0]

    def test_falls_back_on_nonuniform_t_scale(self):
        # Pathological: Claims to use the global-t-scale convention but actually doesn't (s1[t] != s0[t])
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(t=4, z=2, y=2, x=2),
                    pixel_size=PixelSize(t=0.5, z=0.3, y=20.0, x=30.0),
                    ome_zarr_axes="infer",
                ),
                "s1": Scale(
                    shape=Shape(t=4, z=1, y=1, x=1),
                    pixel_size=PixelSize(t=0.9, z=0.6, y=40.0, x=60.0),
                    ome_zarr_axes="infer",
                ),
            },
            _legacy_convention_global_t_scale=0.5,
        )

        result = ms.to_ome_zarr(version="0.4")

        assert "coordinateTransformations" not in result
        assert _dataset_scale(result, "s0") == [0.5, 0.3, 20.0, 30.0]
        assert _dataset_scale(result, "s1") == [0.9, 0.6, 40.0, 60.0]

    def test_does_not_override_t_scale_convention_when_compatible_coordinate_system_with_identity(self):
        """Writing a nondescript identity on the global level doesn't contribute anything useful.
        More important to maintain convention."""
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(t=4, y=2, x=2), pixel_size=PixelSize(t=0.5, y=20.0, x=30.0), ome_zarr_axes="infer"
                ),
                "s1": Scale(
                    shape=Shape(t=4, y=1, x=1), pixel_size=PixelSize(t=0.5, y=40.0, x=60.0), ome_zarr_axes="infer"
                ),
            },
            _legacy_convention_global_t_scale=0.5,
        )
        ms = ms.with_coordinate_system("world", reached_by=None)  # None = identity

        result = ms.to_ome_zarr(version="0.4")

        assert "coordinateTransformations" in result
        # time pixel size follows convention and is written on global level
        assert _global_scale_and_translation(result) == ([0.5, 1.0, 1.0], None)
        assert _dataset_scale(result, "s0") == [1.0, 20.0, 30.0]
        assert _dataset_scale(result, "s1") == [1.0, 40.0, 60.0]

    def test_t_scale_convention_overrides_compatible_coordinate_system(self):
        """
        The "global t-scale" convention splits what is supposed to be the *dataset transforms* across
        both dataset and global transforms. This leaves no room for an additional transform to an undefined
        external reference space.
        Ideally, we would want to distinguish coordinate systems added by `as_derived_from` from those added by
        `with_coordinate_system`:
        - as_derived_from transfers the convention, but also generates an external system, potentially without
          the user's awareness -> maintain convention. If they loaded nifti-zarr (which uses the convention),
          they expect to write nifti-zarr.
        - with_coordinate_system is an explicit call to add an external reference space -> override convention. If
          they made the effort to add an external system, they expect it to be saved. But, they are almost certainly
          working with OME-Zarr 0.6, which has no "global t-scale" convention.
        Hence, in absense of a good way to distinguish the two: Convention overrides external system when writing legacy metadata.
        """
        ms = Multiscale(
            {
                "s0": Scale(
                    shape=Shape(t=4, y=2, x=2), pixel_size=PixelSize(t=0.5, y=20.0, x=30.0), ome_zarr_axes="infer"
                ),
                "s1": Scale(
                    shape=Shape(t=4, y=1, x=1), pixel_size=PixelSize(t=0.5, y=40.0, x=60.0), ome_zarr_axes="infer"
                ),
            },
            _legacy_convention_global_t_scale=0.5,
        )
        ms = ms.with_coordinate_system("world", reached_by=Factor(t=4.0, y=1.0, x=1.0))

        result = ms.to_ome_zarr(version="0.4")

        assert "coordinateTransformations" in result
        expected_global_transform = ([0.5, 1.0, 1.0], None)  # pixel size, not the reached_by factor
        assert _global_scale_and_translation(result) == expected_global_transform
        assert _dataset_scale(result, "s0") == [1.0, 20.0, 30.0]  # pixel size in global transforms, not here
        assert _dataset_scale(result, "s1") == [1.0, 40.0, 60.0]


class TestMultipleCoordinateSystemsToOmeZarr06:
    """Coverage for handling more than one satellite coordinate system (other tests only ever use one)."""

    def test_multiple_satellite_systems(self):
        ms = _multiscale("yx", pixel_size=[0.5, 0.5])
        ms = ms.with_coordinate_system("world", reached_by=Translation(y=3, x=4))
        ms = ms.with_coordinate_system("mm", reached_by=Factor(y=1000, x=1000), unit=Unit(y="mm", x="mm"))
        ms = ms.with_coordinate_system("proj", reached_by=AxisRearrangementTo("zyx"))

        result = ms.to_ome_zarr(version="0.6")

        names = {s["name"] for s in result["coordinateSystems"]}
        assert names == {ms._intrinsic_ref.name, "world", "mm", "proj"}, "json export must include all"
        assert ms.coordinate_systems == ("world", "mm", "proj"), "public API must exclude intrinsic"

        proj_axes = [a["name"] for a in next(s for s in result["coordinateSystems"] if s["name"] == "proj")["axes"]]
        assert proj_axes == ["z", "y", "x"], "AxisRearrangementTo inserts z"

        by_output = {t["output"]["name"]: t for t in result["coordinateTransformations"]}
        assert by_output["world"] == {
            "type": "translation",
            "translation": [-3.0, -4.0],
            "input": {"name": ms._intrinsic_ref.name},
            "output": {"name": "world"},
        }
        assert by_output["mm"] == {
            "type": "scale",
            "scale": [0.001, 0.001],
            "input": {"name": ms._intrinsic_ref.name},
            "output": {"name": "mm"},
        }
        assert by_output["proj"] == {
            "type": "projectAxis",
            "createdOutputs": [0],
            "input": {"name": ms._intrinsic_ref.name},
            "output": {"name": "proj"},
        }
        mm_units = [a["unit"] for a in next(s for s in result["coordinateSystems"] if s["name"] == "mm")["axes"]]
        assert mm_units == ["mm", "mm"]

        read_back = Multiscale.from_ome_zarr(result, shape_source=lambda p: (4, 4))
        assert set(read_back.coordinate_systems) == {"world", "mm", "proj"}
        assert read_back == ms
        assert read_back.to_ome_zarr(version="0.6") == result

    def test_derive_preserves_multiple_inherited_systems_and_source_lineage(self):
        source = Multiscale.from_single(
            Scale(shape=Shape(z=4, y=4, x=4), pixel_size=PixelSize(z=2, y=1, x=1), ome_zarr_axes="infer"),
            blueprint=BlueprintShapes({"s0": Shape(z=4, y=4, x=4), "s1": Shape(z=2, y=2, x=2)}),
        )
        source = source.with_coordinate_system("world", reached_by=Translation(z=1, y=2, x=3))
        source = source.with_coordinate_system("other", reached_by=Factor(z=1, y=2, x=2))

        derived = source.derive("s1", derived_by=Factor(z=1, y=2, x=2))

        result = derived.to_ome_zarr(version="0.6")
        names = {s["name"] for s in result["coordinateSystems"]}
        # Both inherited satellites, plus a direct lineage link back to the source's own intrinsic system.
        assert names == {derived._intrinsic_ref.name, source._intrinsic_ref.name, "world", "other"}

        read_back = Multiscale.from_ome_zarr(result, shape_source=lambda p: (2, 2, 2))
        assert set(read_back.coordinate_systems) == {source._intrinsic_ref.name, "world", "other"}
        assert read_back == derived
        assert read_back.to_ome_zarr(version="0.6") == result
