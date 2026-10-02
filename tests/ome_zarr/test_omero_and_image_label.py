import copy

import pytest
from clearscale import Multiscale, OmeZarrGroup, Scale, Shape
from clearscale._services.ome_zarr import MultiscaleProperties
from clearscale.ome_zarr import (
    ImageLabel,
    LabelEntry,
    Omero,
    OmeroChannel,
    OmeroWindow,
)


def _multiscale(shape: Shape) -> Multiscale:
    return Multiscale({"s0": Scale(shape=shape)})


def _channel(color="FF0000") -> OmeroChannel:
    return OmeroChannel(color=color, window=OmeroWindow(start=0, end=255, min=0, max=255))


@pytest.mark.parametrize("missing", ["start", "end", "min", "max"])
def test_omero_window_from_ome_zarr_requires_all_four_fields(missing):
    full = {"start": 0, "end": 1, "min": 0, "max": 1}
    del full[missing]
    assert OmeroWindow.from_ome_zarr(full) is None


def test_omero_window_from_ome_zarr_rejects_non_mapping():
    assert OmeroWindow.from_ome_zarr("not a dict") is None
    assert OmeroWindow.from_ome_zarr(None) is None


def test_omero_channel_requires_color_and_window():
    with pytest.raises(TypeError):
        OmeroChannel(color="FF0000")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OmeroChannel(window=OmeroWindow(start=0, end=1, min=0, max=1))  # type: ignore[arg-type]


def test_omero_channel_normalizes_hex_color():
    assert _channel("ff00aa").color == "FF00AA"
    assert _channel("#ff00aa").color == "FF00AA"


@pytest.mark.parametrize("bad_color", ["ff00a", "gg0000", "#ff00aaff", 123])
def test_omero_channel_rejects_invalid_color(bad_color):
    with pytest.raises(ValueError):
        _channel(bad_color)


def test_omero_channel_keeps_unknown_keys_in_extra():
    channel = OmeroChannel.from_ome_zarr(
        {"color": "FF0000", "window": {"start": 0, "end": 1, "min": 0, "max": 1}, "label": "DAPI", "family": "linear"}
    )
    assert channel is not None
    assert channel.extra == {"label": "DAPI", "family": "linear"}
    assert channel.to_ome_zarr()["label"] == "DAPI"
    assert channel.to_ome_zarr()["family"] == "linear"


def test_omero_channel_from_ome_zarr_requires_color_and_window():
    assert OmeroChannel.from_ome_zarr({}) is None
    assert OmeroChannel.from_ome_zarr({"color": "FF0000"}) is None  # no window
    assert OmeroChannel.from_ome_zarr({"window": {"start": 0, "end": 1, "min": 0, "max": 1}}) is None  # no color
    assert OmeroChannel.from_ome_zarr({"color": "zzzzzz", "window": {}}) is None  # invalid color


def test_omero_requires_channels():
    with pytest.raises(ValueError, match="Omero.channels must contain at least one OmeroChannel"):
        Omero([])
    assert Omero.from_ome_zarr({}) is None
    assert Omero.from_ome_zarr({"channels": []}) is None
    assert Omero.from_ome_zarr({"channels": "not a list"}) is None
    assert Omero.from_ome_zarr("not a dict") is None


def test_omero_from_ome_zarr_drops_the_whole_object_on_one_bad_channel():
    """Silently dropping just the bad channel would misalign every later channel's index against the
    data, since channels has no other way to identify which image channel it belongs to."""
    with pytest.warns(UserWarning, match="Invalid entry in 'omero.channels'"):
        omero = Omero.from_ome_zarr(
            {
                "channels": [
                    {"color": "FF0000", "window": {"start": 0, "end": 1, "min": 0, "max": 1}},
                    {"color": "not-a-color"},
                ]
            }
        )
    assert omero is None


def test_omero_keeps_unknown_top_level_keys_in_extra():
    omero = Omero.from_ome_zarr(
        {"channels": [{"color": "FF0000", "window": {"start": 0, "end": 1, "min": 0, "max": 1}}], "id": 1, "name": "x"}
    )
    assert omero is not None
    assert omero.extra == {"id": 1, "name": "x"}
    d = omero.to_ome_zarr()
    assert d["id"] == 1
    assert d["name"] == "x"
    assert len(d["channels"]) == 1


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_omero_round_trips_through_group_attrs(version):
    ms = _multiscale(Shape(c=2, y=4, x=4))
    ms.ome.omero = Omero([_channel("FF0000"), _channel("00FF00")], extra={"id": 1, "name": "example.tif"})
    group = OmeZarrGroup.from_single(ms)

    attrs = group.to_attrs(version=version)
    ome_attrs = attrs if version == "0.4" else attrs["ome"]
    assert ome_attrs["omero"]["channels"][0] == {
        "color": "FF0000",
        "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
    }
    assert ome_attrs["omero"]["id"] == 1

    read_back = OmeZarrGroup.from_attrs(attrs, shape_source={"s0": (2, 4, 4)})
    assert read_back.multiscales[0].ome.omero == ms.ome.omero


def test_omero_extra_deep_copies_so_writer_output_does_not_alias_the_stored_object():
    channel = _channel()
    omero = Omero([channel], extra={"nested": {"a": 1}})
    d = omero.to_ome_zarr()
    d["nested"]["a"] = 999
    assert omero.extra["nested"]["a"] == 1


####
# ImageLabel / LabelEntry
####


def test_label_entry_color_validates_on_assignment():
    entry = LabelEntry()
    entry.color = (255, 0, 0, 128)
    assert entry.color == (255, 0, 0, 128)

    with pytest.raises(ValueError):
        entry.color = (256, 0, 0, 0)
    with pytest.raises(ValueError):
        entry.color = (0, 0, 0)  # only 3 values


def test_label_entry_bool():
    assert not LabelEntry()
    assert LabelEntry(color=(0, 0, 0, 0))
    assert LabelEntry(properties={"class": "cell"})


def test_label_entry_repr_shows_color():
    entry = LabelEntry(color=(255, 0, 0, 128))
    assert "color=(255, 0, 0, 128)" in repr(entry)


def test_image_label_is_frozen_but_labels_dict_is_mutable():
    label = ImageLabel(source="../../", labels={1: LabelEntry(color=(255, 0, 0, 128))})

    with pytest.raises(AttributeError, match="cannot assign to field 'source'"):
        label.source = None  # type: ignore[reportAttributeAccessIssue]

    label.labels[1].properties["class"] = "cell"
    assert label.labels[1].properties == {"class": "cell"}


def test_image_label_bool_and_source_default():
    assert not ImageLabel()
    assert ImageLabel(source="../../")
    assert ImageLabel(labels={1: LabelEntry(properties={"class": "cell"})})
    assert ImageLabel().source is None, "absence of 'source' must not be filled in as '../../'"


def test_image_label_rejects_non_label_entry_values():
    with pytest.raises(TypeError):
        ImageLabel(labels={1: {"color": (0, 0, 0, 0)}})  # type: ignore[arg-type]


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_image_label_to_ome_zarr_splits_colors_and_properties(version):
    label = ImageLabel(
        source="../../",
        labels={
            1: LabelEntry(color=(255, 0, 0, 128), properties={"class": "foo", "area (pixels)": 1200}),
            3: LabelEntry(),  # no color, no properties -> appears in neither list
            4: LabelEntry(color=(0, 255, 255, 128)),
        },
    )
    d = label.to_ome_zarr(version=version)
    assert d["colors"] == [
        {"label-value": 1, "rgba": [255, 0, 0, 128]},
        {"label-value": 4, "rgba": [0, 255, 255, 128]},
    ]
    assert d["properties"] == [{"label-value": 1, "class": "foo", "area (pixels)": 1200}]
    assert d["source"] == {"image": "../../"}
    assert d.get("version") == (version if version in ("0.4", "0.5") else None)


def test_image_label_from_ome_zarr_merges_colors_and_properties_by_label_value():
    json = {
        "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}, {"label-value": 3}],
        "properties": [{"label-value": 1, "class": "foo"}, {"label-value": 2, "class": "bar"}],
        "source": {"image": "../../raw"},
    }
    label = ImageLabel.from_ome_zarr(json)
    assert label is not None
    assert label.labels is not None
    assert set(label.labels.keys()) == {1, 2, 3}
    assert label.labels[1].color == (255, 0, 0, 128)
    assert label.labels[1].properties == {"class": "foo"}
    assert label.labels[2].properties == {"class": "bar"}
    assert not label.labels[3]
    assert label.source is not None
    assert label.source.path == "../../raw"


def test_image_label_from_ome_zarr_last_value_wins_on_duplicate_label_value():
    json = {"properties": [{"label-value": 1, "class": "foo"}, {"label-value": 1, "class": "bar"}]}
    label = ImageLabel.from_ome_zarr(json)
    assert label is not None
    assert label.labels[1].properties == {"class": "bar"}


def test_image_label_extra_color_keys_round_trip_via_properties_prefix():
    """0.6 allows arbitrary extra keys alongside 'rgba' under 'colors'. clearscale folds them into
    .properties with an '@color:' prefix rather than keeping a separate extra-keys concept, since that
    concept has no future (RFC-8 merges colors/properties into one list and drops the distinction)."""
    json = {"colors": [{"label-value": 1, "rgba": [255, 0, 0, 128], "hexColor": "#fff"}]}
    label = ImageLabel.from_ome_zarr(json)
    assert label is not None
    assert label.labels[1].properties == {"@color:hexColor": "#fff"}

    d = label.to_ome_zarr(version="0.6")
    assert d["colors"] == [{"label-value": 1, "rgba": [255, 0, 0, 128], "hexColor": "#fff"}]
    assert "properties" not in d  # the whole entry folded back into 'colors', nothing left for 'properties'


def test_image_label_invalid_rgba_is_dropped_with_warning_rest_of_entry_kept():
    json = {"colors": [{"label-value": 1, "rgba": "not-a-color", "hexColor": "#fff"}]}
    with pytest.warns(UserWarning, match="Invalid image-label color"):
        label = ImageLabel.from_ome_zarr(json)
    assert label is not None
    assert label.labels[1].color is None
    assert label.labels[1].properties == {"@color:hexColor": "#fff"}


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_image_label_round_trips_through_group_attrs(version):
    ms = _multiscale(Shape(y=4, x=4))
    ms.ome.image_label = ImageLabel(
        source="../../", labels={1: LabelEntry(color=(255, 0, 0, 128), properties={"class": "foo"})}
    )
    group = OmeZarrGroup.from_single(ms)

    attrs = group.to_attrs(version=version)
    read_back = OmeZarrGroup.from_attrs(attrs, shape_source={"s0": (4, 4)})
    assert read_back.multiscales[0].ome.image_label == ms.ome.image_label


####
# OmeZarrGroup wiring: reading attaches per-Multiscale, writing reconciles across Multiscales
####


def test_from_attrs_shares_omero_but_copies_image_label_across_multiscales():
    """Omero is frozen with no mutable identity beyond a plain 'extra' dict, so sharing one instance
    across every Multiscale is safe. ImageLabel/LabelEntry are deliberately mutable, so each Multiscale
    must get its own copy - otherwise mutating one Multiscale's image_label would silently mutate every
    other Multiscale parsed from the same group."""
    ms = _multiscale(Shape(c=1, y=4, x=4))
    ms.ome.omero = Omero([_channel()])
    ms.ome.image_label = ImageLabel(labels={1: LabelEntry(color=(255, 0, 0, 128))})
    attrs = OmeZarrGroup.from_single(ms).to_attrs(version="0.6", override_multi_multiscales=True)

    # Simulate a group whose "multiscales" list has two entries describing the same image
    attrs["ome"]["multiscales"] = attrs["ome"]["multiscales"] * 2
    group = OmeZarrGroup.from_attrs(attrs, shape_source={"s0": (1, 4, 4)})

    ms0, ms1 = group.multiscales
    assert ms0.ome.omero is ms1.ome.omero  # frozen: sharing is safe
    assert ms0.ome.image_label is not None
    assert ms0.ome.image_label.labels is not None
    assert ms1.ome.image_label is not None
    assert ms1.ome.image_label.labels is not None

    assert ms0.ome.image_label is not ms1.ome.image_label  # mutable: must not alias
    assert ms0.ome.image_label == ms1.ome.image_label  # but still equal in content

    ms0.ome.image_label.labels[1].properties["touched"] = True
    assert "touched" not in ms1.ome.image_label.labels[1].properties


def test_to_attrs_warns_and_omits_omero_on_conflict_across_multiscales():
    ms1 = _multiscale(Shape(c=1, y=4, x=4))
    ms2 = _multiscale(Shape(c=1, y=2, x=2))
    ms1.ome.omero = Omero([_channel("FF0000")])
    ms2.ome.omero = Omero([_channel("00FF00")])
    group = OmeZarrGroup(multiscales=(ms1, ms2))

    with pytest.warns(UserWarning, match="conflicting `.ome.omero`"):
        attrs = group.to_attrs(version="0.6", override_multi_multiscales=True)
    assert "omero" not in attrs["ome"]


def test_to_attrs_raises_on_image_label_conflict_across_multiscales():
    ms1 = _multiscale(Shape(y=4, x=4))
    ms2 = _multiscale(Shape(y=2, x=2))
    ms1.ome.image_label = ImageLabel(source="../../")
    ms2.ome.image_label = ImageLabel(source="../../elsewhere")
    group = OmeZarrGroup(multiscales=(ms1, ms2))

    with pytest.raises(ValueError, match="Can only write one image-label per group"):
        group.to_attrs(version="0.6", override_multi_multiscales=True)


def test_to_attrs_writes_shared_omero_and_image_label_when_multiscales_agree():
    ms1 = _multiscale(Shape(c=1, y=4, x=4))
    ms2 = _multiscale(Shape(c=1, y=2, x=2))
    ms1.ome.omero = Omero([_channel("FF0000")])
    ms2.ome.omero = Omero([_channel("FF0000")])  # equal, not the same instance
    group = OmeZarrGroup(multiscales=(ms1, ms2))

    attrs = group.to_attrs(version="0.6", override_multi_multiscales=True)
    assert attrs["ome"]["omero"]["channels"][0]["color"] == "FF0000"


def test_to_attrs_multi_multiscale_requires_override():
    ms1 = _multiscale(Shape(y=4, x=4))
    ms2 = _multiscale(Shape(y=2, x=2))
    group = OmeZarrGroup(multiscales=(ms1, ms2))

    with pytest.raises(ValueError, match="collection groups are not supported"):
        group.to_attrs(version="0.6")

    attrs = group.to_attrs(version="0.6", override_multi_multiscales=True)  # does not raise
    assert len(attrs["ome"]["multiscales"]) == 2


####
# MultiscaleProperties integration
####


def test_multiscale_properties_bool_includes_omero_and_image_label():
    assert not MultiscaleProperties()
    assert MultiscaleProperties(omero=Omero([_channel()]))
    assert MultiscaleProperties(image_label=ImageLabel(source="../../"))


def test_multiscale_to_ome_zarr_never_writes_omero_or_image_label_into_the_entry():
    """omero/image-label are group-level keys; Multiscale.to_ome_zarr() must not leak them into the
    per-entry dict even though they live on the same .ome container as type/name/metadata."""
    ms = _multiscale(Shape(c=1, y=4, x=4))
    ms.ome.omero = Omero([_channel()])
    ms.ome.image_label = ImageLabel(source="../../")
    ms.ome.name = "my-image"

    entry = ms.to_ome_zarr(version="0.5")
    assert "omero" not in entry
    assert "image-label" not in entry
    assert entry["name"] == "my-image"


def test_omero_and_image_label_deep_copy_cleanly():
    """Multiscale's carryover methods (with_coordinate_system, as_derived_from, ...) deep-copy the whole
    `.ome` container - this previously broke when OmeroChannel.extra was stored as a MappingProxyType,
    since mappingproxy objects cannot be deep-copied. Both stay plain dicts now."""
    omero = Omero([_channel()], extra={"id": 1})
    image_label = ImageLabel(source="../../", labels={1: LabelEntry(color=(255, 0, 0, 128))})
    copy.deepcopy(omero)
    copy.deepcopy(image_label)
