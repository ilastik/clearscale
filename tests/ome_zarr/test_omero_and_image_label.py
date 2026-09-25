import warnings

import pytest
from clearscale import Multiscale, OmeZarrGroup, Scale, Shape
from clearscale._services.ome_zarr import MultiscaleProperties
from clearscale.ome_zarr import (
    ImageLabel,
    LabelProperties,
    Omero,
    OmeroChannel,
    OmeroRenderingDefaults,
    OmeroWindow,
)


def _multiscale(shape: Shape) -> Multiscale:
    return Multiscale({"s0": Scale(shape=shape)})


####
# OmeroChannel / OmeroWindow / OmeroRenderingDefaults
####


def test_omero_channel_normalizes_hex_color():
    assert OmeroChannel(color="ff00aa").color == "FF00AA"
    assert OmeroChannel(color="#ff00aa").color == "FF00AA"


@pytest.mark.parametrize("bad_color", ["ff00a", "gg0000", "#ff00aaff", 123])
def test_omero_channel_rejects_invalid_color(bad_color):
    with pytest.raises(ValueError):
        OmeroChannel(color=bad_color)


def test_omero_channel_keeps_unknown_keys_in_extra():
    channel = OmeroChannel.from_ome_zarr({"label": "DAPI", "family": "linear", "coefficient": 1.0})
    assert channel.label == "DAPI"
    assert channel.extra == {"family": "linear", "coefficient": 1.0}
    assert channel.to_ome_zarr() == {"label": "DAPI", "family": "linear", "coefficient": 1.0}


def test_omero_window_requires_start_and_end():
    window = OmeroWindow(start=0, end=1500, min=0, max=65535)
    assert window.to_ome_zarr() == {"start": 0.0, "end": 1500.0, "min": 0.0, "max": 65535.0}

    with pytest.raises(TypeError):
        OmeroWindow(start="a", end=1)


def test_omero_window_from_ome_zarr_ignores_missing_end():
    assert OmeroWindow.from_ome_zarr({"start": 0}) is None
    assert OmeroWindow.from_ome_zarr("not a dict") is None


def test_omero_rendering_defaults_rejects_bad_model():
    with pytest.raises(ValueError):
        OmeroRenderingDefaults(model="rainbow")


def test_omero_rendering_defaults_from_ome_zarr_empty_is_none():
    assert OmeroRenderingDefaults.from_ome_zarr({}) is None
    assert OmeroRenderingDefaults.from_ome_zarr({"model": "color"}).model == "color"


####
# Omero
####


def _sample_omero(n_channels=2) -> Omero:
    return Omero(
        [OmeroChannel(label=f"ch{i}", color="FF0000", window=OmeroWindow(start=0, end=255)) for i in range(n_channels)],
        id=1,
        name="example.tif",
        rdefs=OmeroRenderingDefaults(default_z=5, model="color"),
    )


def test_omero_from_ome_zarr_requires_channels():
    assert Omero.from_ome_zarr({}) is None
    assert Omero.from_ome_zarr({"channels": []}) is None
    assert Omero.from_ome_zarr({"channels": "not a list"}) is None


def test_omero_from_ome_zarr_skips_invalid_channel_entries():
    with pytest.warns(UserWarning, match="Invalid omero channel color"):
        omero = Omero.from_ome_zarr({"channels": [{"label": "ok"}, {"label": "bad", "color": "zzzzzz"}]})
    assert len(omero.channels) == 2  # the whole channel isn't dropped, just the bad color field
    assert omero.channels[1].color is None


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_omero_round_trips_through_group_attrs(version):
    ms = _multiscale(Shape(c=2, y=4, x=4))
    ms.ome.omero = _sample_omero(2)
    group = OmeZarrGroup.from_single(ms)

    attrs = group.to_attrs(version=version)
    ome_attrs = attrs if version == "0.4" else attrs["ome"]
    assert ome_attrs["omero"]["channels"][0] == {
        "label": "ch0",
        "color": "FF0000",
        "window": {"start": 0.0, "end": 255.0},
    }
    assert ome_attrs["omero"]["id"] == 1
    assert ome_attrs["omero"]["rdefs"] == {"defaultZ": 5, "model": "color"}

    read_back = OmeZarrGroup.from_attrs(attrs, shape_source={"s0": (2, 4, 4)})
    assert read_back.multiscales[0].ome.omero == ms.ome.omero


def test_omero_channel_count_mismatch_raises_on_write():
    ms = _multiscale(Shape(c=3, y=4, x=4))
    ms.ome.omero = _sample_omero(2)  # 2 channels, but the image has 3
    group = OmeZarrGroup.from_single(ms)

    with pytest.raises(ValueError, match="omero.channels has 2 entries, but the image has 3"):
        group.to_attrs(version="0.5")


def test_omero_channel_count_mismatch_warns_and_drops_on_read():
    ms = _multiscale(Shape(c=3, y=4, x=4))
    json = {
        "version": "0.5",
        "multiscales": [ms.to_ome_zarr(version="0.5")],
        "omero": _sample_omero(2).to_ome_zarr(),  # only 2 channels for a 3-channel image
    }
    with pytest.warns(UserWarning, match="omero.channels has 2 entries, but the multiscale"):
        group = OmeZarrGroup.from_attrs(json, shape_source={"s0": (3, 4, 4)})
    assert group.multiscales[0].ome.omero is None


def test_omero_without_channel_axis_requires_exactly_one_channel():
    ms = _multiscale(Shape(y=4, x=4))  # no "c" axis
    ms.ome.omero = _sample_omero(1)
    group = OmeZarrGroup.from_single(ms)
    attrs = group.to_attrs(version="0.5")  # should not raise
    assert attrs["ome"]["omero"]["channels"]

    ms.ome.omero = _sample_omero(2)
    with pytest.raises(ValueError, match="but the image has 1 channel"):
        OmeZarrGroup.from_single(ms).to_attrs(version="0.5")


def test_group_with_conflicting_omero_across_multiscales_raises():
    ms1 = _multiscale(Shape(c=1, y=4, x=4))
    ms2 = _multiscale(Shape(c=1, y=2, x=2))
    ms1.ome.omero = _sample_omero(1)
    ms2.ome.omero = Omero([OmeroChannel(label="different")])
    group = OmeZarrGroup(multiscales=(ms1, ms2))

    with pytest.raises(ValueError, match="conflicting 'omero' metadata"):
        group.to_attrs(version="0.6")


####
# ImageLabel / LabelProperties
####


def test_label_properties_merges_color_reserved_key():
    with pytest.raises(ValueError, match="reserved"):
        LabelProperties({1: {"label-value": 1}})
    with pytest.raises(ValueError, match="reserved"):
        LabelProperties({1: {"labelValue": 1}})


def test_label_properties_validates_color():
    with pytest.raises(ValueError):
        LabelProperties({1: {"color": (256, 0, 0, 0)}})
    with pytest.raises(ValueError):
        LabelProperties({1: {"color": (0, 0, 0)}})  # only 3 values

    props = LabelProperties({1: {"color": (255, 0, 0, 128)}})
    assert props[1]["color"] == (255, 0, 0, 128)


def test_image_label_bool_and_source_default():
    assert not ImageLabel()
    assert ImageLabel(source="../../")
    assert ImageLabel(properties={1: {"class": "cell"}})
    assert ImageLabel().source is None, "absence of 'source' must not be filled in as '../../'"


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_image_label_to_ome_zarr_splits_colors_and_properties(version):
    label = ImageLabel(
        source="../../",
        properties={
            1: {"color": (255, 0, 0, 128), "class": "foo", "area (pixels)": 1200},
            3: {},
            4: {"color": (0, 255, 255, 128)},
        },
    )
    d = label.to_ome_zarr(version=version)
    assert d["colors"] == [
        {"label-value": 1, "rgba": [255, 0, 0, 128]},
        {"label-value": 3},
        {"label-value": 4, "rgba": [0, 255, 255, 128]},
    ]
    assert d["properties"] == [{"label-value": 1, "class": "foo", "area (pixels)": 1200}]
    assert d["source"] == {"image": "../../"}
    assert d.get("version") == ("0.4" if version == "0.4" else None)


def test_image_label_from_ome_zarr_merges_colors_and_properties_by_label_value():
    json = {
        "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}, {"label-value": 3}],
        "properties": [{"label-value": 1, "class": "foo"}, {"label-value": 2, "class": "bar"}],
        "source": {"image": "../../raw"},
    }
    label = ImageLabel.from_ome_zarr(json)
    assert set(label.properties.keys()) == {1, 2, 3}
    assert dict(label.properties[1]) == {"color": (255, 0, 0, 128), "class": "foo"}
    assert dict(label.properties[2]) == {"class": "bar"}
    assert dict(label.properties[3]) == {}
    assert label.source.path == "../../raw"


def test_image_label_from_ome_zarr_warns_on_conflicting_values():
    json = {
        "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}],
        "properties": [{"label-value": 1, "color": [0, 0, 0, 0]}],
    }
    with pytest.warns(UserWarning, match="Conflicting 'color'"):
        label = ImageLabel.from_ome_zarr(json)
    assert dict(label.properties[1]) == {"color": (255, 0, 0, 128)}  # 'colors' entry wins


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_image_label_round_trips_through_group_attrs(version):
    ms = _multiscale(Shape(y=4, x=4))
    ms.ome.image_label = ImageLabel(source="../../", properties={1: {"color": (255, 0, 0, 128), "class": "foo"}})
    group = OmeZarrGroup.from_single(ms)

    attrs = group.to_attrs(version=version)
    read_back = OmeZarrGroup.from_attrs(attrs, shape_source={"s0": (4, 4)})
    assert read_back.multiscales[0].ome.image_label == ms.ome.image_label


####
# MultiscaleProperties integration
####


def test_multiscale_properties_bool_includes_omero_and_image_label():
    assert not MultiscaleProperties()
    assert MultiscaleProperties(omero=_sample_omero(1))
    assert MultiscaleProperties(image_label=ImageLabel(source="../../"))


def test_multiscale_to_ome_zarr_never_writes_omero_or_image_label_into_the_entry():
    """omero/image-label are group-level keys; Multiscale.to_ome_zarr() must not leak them into the
    per-entry dict even though they live on the same .ome container as type/name/metadata."""
    ms = _multiscale(Shape(c=1, y=4, x=4))
    ms.ome.omero = _sample_omero(1)
    ms.ome.image_label = ImageLabel(source="../../")
    ms.ome.name = "my-image"

    entry = ms.to_ome_zarr(version="0.5")
    assert "omero" not in entry
    assert "image-label" not in entry
    assert entry["name"] == "my-image"
