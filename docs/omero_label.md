# Display and label metadata: `omero` and `image-label`

OME-Zarr stores a few things *about* a multiscale next to the multiscale, not inside it.
Namely:

* `omero`: how to *display* the image (channel colors, contrast windows, names)
* `image-label`: marks the image as a *label image* (a segmentation), with display colors and properties per label value

clearscale exposes both, and a few smaller fields, on `Multiscale.ome`.
This page covers how to read, modify and write them.

> [!WARNING]
> Everything in `Multiscale.ome` is yours to keep correct.
> clearscale never reads image data, so it cannot check these properties against it: not that `omero` has one channel-entry per channel of the data, not that label values actually appear in the data, or that the data is a label image at all.
> It also does not use them for anything: `.ome` is not part of Multiscale equality, and nothing in clearscale updates it when you derive, select or reorder.
> Unlike the rest of `Multiscale`, `.ome` is *mutable*, and so are the dicts inside `Omero` and `ImageLabel`: editing one in place edits it for everyone who holds a reference.
> See [What to watch out for](#what-to-watch-out-for).

The snippets use `zarr-python` 3 for storage, as in the other guides, but clearscale itself only needs a group's attrs.

## Concept

`Multiscale.ome` is a small, mostly mutable bag of OME-Zarr-specific properties:

| `multiscale.ome` field | Where it lives in OME-Zarr                  | Level                   |
|------------------------|---------------------------------------------|-------------------------|
| `.omero`               | `omero` key                                 | group, next to `multiscales` |
| `.image_label`         | `image-label` key                           | group, next to `multiscales` |
| `.name`                | `multiscales[i].name`                       | multiscale entry        |
| `.type`                | `multiscales[i].type`                       | multiscale entry        |
| `.metadata`            | `multiscales[i].metadata`                   | multiscale entry        |

(In OME-Zarr 0.5 and 0.6, all keys are inside the `ome` namespace of the group attrs; in 0.4 they are top-level attrs.)

The standard only loosely specifies what goes into these fields, and little about how a tool must interpret them.
They are for display and bookkeeping.
clearscale never uses them for its own logic: they don't influence axes, shapes, scaling or coordinate systems, and **two Multiscales that differ only in `.ome` compare equal**.

Technically, `omero` and `image-label` describe the *group*, not an individual multiscale.
The probable intention was that a group holding several multiscales would be different downsampling methods of the *same* raw data, so display settings and label information are the same for all of them.
clearscale models them as properties of the Multiscale instead of `OmeZarrGroup`, because it is more convenient for the more common case (one Multiscale per zarr group).
[Watch out for](#what-to-watch-out-for) explains the quirks this causes.

### `omero`

`Omero` holds one `OmeroChannel` per channel.
Each channel may provide:

* `color`: 6 hex digits, like `"FF8800"`
* `window`: an `OmeroWindow` with `start` and `end` (the contrast limits to display), and `min` and `max` (the full range the data can take, e.g. 0 and 255 for `uint8`)
* `label`: the channel's name
* `active`: whether to display the channel
* `inverted`: whether to invert the rendering of darkest to brightest
* `extra`: any other keys of the channel, like `family` or `coefficient`

`Omero.extra` holds any other top-level keys, like `name` or `rdefs`.

This metadata only makes sense when the Multiscale has an axis like `.ome_zarr_axes["c"].type == "channel"`, and the channel axis is not in `.scaled_axes`.

### `image-label`

`ImageLabel` marks a multiscale as a label image.
It has:

* `source`: the path to the image these labels belong to, relative to the label group (usually `"../../"` because the label-Multiscale is in `raw_multiscale.ome.zarr/labels/label_multiscale`)
* `labels`: a `{label_value: LabelEntry}` dict. A `LabelEntry` has an optional RGBA `color` and a dict of arbitrary `properties`

The OME-Zarr JSON stores colors and properties in two separate lists, each entry carrying its `label-value`.
clearscale merges them into one entry per label value, which is easier to work with.
You may want to create an entry for every value present in the segmentation, including background, even if you attach no color or properties, using empty `LabelEntry()`.
clearscale does not inspect the data to fill in missing entries.

## Reading

Read metadata the usual way, through `OmeZarrGroup`.
Here, the attrs are spelled out so you can see where each value comes from:

```python
import clearscale as cs

attrs = {
    "ome": {
        "version": "0.5",
        "multiscales": [
            {
                "name": "tile_7",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5]}],
                    }
                ],
            }
        ],
        "omero": {
            "name": "tile 7",
            "channels": [
                {
                    "label": "DAPI",
                    "color": "0000FF",
                    "active": True,
                    "window": {"start": 100, "end": 2000, "min": 0, "max": 65535},
                },
                {
                    "label": "GFP",
                    "color": "00FF00",
                    "active": True,
                    "window": {"start": 0, "end": 800, "min": 0, "max": 65535},
                },
            ],
        },
    }
}

# With a real group, instead: cs.OmeZarrGroup.from_group(zarr_group)
group = cs.OmeZarrGroup.from_attrs(attrs, shape_source="singletons")
multiscale = group.multiscales[0]

assert multiscale.ome.name == "tile_7"  # from the multiscale entry

omero = multiscale.ome.omero  # from the group-level "omero" key. None if absent or unusable
assert omero is not None
for channel in omero.channels:
    print(channel.label, channel.color, channel.window.start, channel.window.end)
# DAPI 0000FF 100.0 2000.0
# GFP 00FF00 0.0 800.0

assert omero.extra == {"name": "tile 7"}
```

`multiscale.ome.omero` is `None` if the group has no `omero` key, or if clearscale could not use it (see [below](#malformed-omero-metadata)).

Label images are read the same way.
A group that holds a label image is an ordinary `GroupKind.MULTISCALE` group.
The label image is recognized by its `ome.image_label`, nothing else.
`ome.image_label` is `None` for ordinary images.

```python
label_attrs = {
    "ome": {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [{"type": "scale", "scale": [0.5, 0.5]}],
                    }
                ],
            }
        ],
        "image-label": {
            "source": {"image": "../../"},
            "colors": [
                {"label-value": 1, "rgba": [255, 0, 0, 255]},
                {"label-value": 2, "rgba": [0, 0, 255, 255]},
            ],
            "properties": [
                {"label-value": 1, "name": "cell", "area": 12.5},
                {"label-value": 2, "name": "nucleus"},
            ],
        },
    }
}

label_multiscale = cs.OmeZarrGroup.from_attrs(label_attrs, shape_source="singletons").multiscales[0]

image_label = label_multiscale.ome.image_label
assert image_label is not None, "This is a label image"
assert image_label.source == "../../"
for value, entry in image_label.labels.items():
    print(value, entry.color, entry.properties)
# 1 (255, 0, 0, 255) {'name': 'cell', 'area': 12.5}
# 2 (0, 0, 255, 255) {'name': 'nucleus'}
```

### Finding label images

Label images are stored in a `labels` subgroup of the image they belong to:

```
image.zarr/
├── zarr.json         <- multiscales, omero
├── s0/               <- image array
└── labels/
    ├── zarr.json     <- "labels": ["cells", "nuclei"]
    ├── cells/
    │   ├── zarr.json <- multiscales, image-label (source: "../../")
    │   └── s0/       <- label array
    └── nuclei/
        └── ...
```

An `OmeZarrGroup` for an image tells you where `labels` may exist (`group.maybe_subgroups == ("labels",)`).
The `labels` group, if it exists, has `GroupKind.LABELS` and lists the label images in `.children` as `ChildRef`s with paths relative to the `labels` group.
See [OME-Zarr: Path discovery](ome_zarr.md#path-discovery) for the general procedure, and the [writing example](#writing-an-image-with-labels) below for what this looks like in code.

## Setting up and modifying

### Omero

`Omero`, `OmeroChannel`, `OmeroWindow` and the label classes live in `clearscale.ome_zarr`.
Assign a new `Omero` to a Multiscale's `.ome.omero`:

```python
import clearscale as cs

image = cs.Multiscale.from_single(
    cs.Scale(
        shape=cs.Shape(c=2, y=64, x=64),
        pixel_size=cs.PixelSize(c=1, y=0.5, x=0.5),
        unit=cs.Unit(c="", y="micrometer", x="micrometer"),
    )
)

colors = ["0000FF", "00FF00"]
image.ome.omero = cs.ome_zarr.Omero(
    [
        cs.ome_zarr.OmeroChannel(
            color=color,
            window=cs.ome_zarr.OmeroWindow(start=0, end=255, min=0, max=255),
            label=f"channel {index}",
            active=True,
        )
        for index, color in enumerate(colors)
    ],
    extra={"name": "my image"},
)
```

When supplied, `color` must be 6 hex digits (a leading `#` is accepted and removed, lowercase is converted to uppercase), `label` must be a string, and `active` and `inverted` must be booleans.
Window values must be finite numbers, and an `Omero` needs at least one channel.
Note that `OmeroChannel()` is valid, but trying to write a channel without the required color and fully-specified window in OME-Zarr versions 0.4, 0.5 or 0.6 raises an error.

`Omero`, `OmeroChannel` and `OmeroWindow` are frozen dataclasses.
To change one, make a modified copy with `dataclasses.replace` and assign it back:

```python
import dataclasses

omero = image.ome.omero
first, *others = omero.channels
brighter = dataclasses.replace(first, window=dataclasses.replace(first.window, end=128))
recolored = dataclasses.replace(others[0], color="FF00FF")
image.ome.omero = dataclasses.replace(omero, channels=[brighter, recolored])

assert image.ome.omero.channels[0].window.end == 128.0
assert image.ome.omero.channels[1].color == "FF00FF"
```

`extra` is a plain dict, so you *can* edit it in place (`omero.extra["name"] = "new"`), but see [shared instances](#what-to-watch-out-for) first.
When writing, clearscale takes `color`, `window`, `label`, `active` and `inverted` from the channel properties, and `channels` from `Omero.channels`.
It ignores matching extra keys like `extra["label"]`.

To remove omero metadata, set `image.ome.omero = None`.

### Image-label

Assign an `ImageLabel` to `.ome.image_label` to mark a multiscale as a label image:

```python
import clearscale as cs

labels = cs.Multiscale.from_single(
    cs.Scale(
        shape=cs.Shape(y=64, x=64),
        pixel_size=cs.PixelSize(y=0.5, x=0.5),
        unit=cs.Unit(y="micrometer", x="micrometer"),
    )
)

labels.ome.image_label = cs.ome_zarr.ImageLabel(
    source="../../",
    labels={
        0: cs.ome_zarr.LabelEntry(),  # background: declared without color or properties
        1: cs.ome_zarr.LabelEntry(color=(255, 0, 0, 255), properties={"name": "cell"}),
        2: cs.ome_zarr.LabelEntry(properties={"name": "nucleus"}),
    },
)
```

Use `source="../../"` if you store the label image the standard way, at `<image group>/labels/<name>`.
`ImageLabel()` just marks the multiscale as a label image.
For this example, assume the segmentation contains values 0, 1 and 2; all three are declared above.

`ImageLabel.labels` and the `LabelEntry`s are mutable, so you can edit them in place:

```python
image_label = labels.ome.image_label
image_label.labels[3] = cs.ome_zarr.LabelEntry(color=[0, 255, 0, 255])  # colors take 4 ints in 0..255
image_label.labels[2].color = (0, 0, 255, 255)
image_label.labels[2].properties["area"] = 7.0
image_label.labels[1] = cs.ome_zarr.LabelEntry()  # keep the value declared after removing its metadata
```

Label values must be integers, and `LabelEntry.color` must be four integers from 0 to 255 (RGBA).
Anything else raises `ValueError`.

Only `source` can't be reassigned: make a new `ImageLabel(source=..., labels=...)` for that (`copy.deepcopy` the `labels` first if you continue to use the old one, because the `LabelEntry` objects are not copied by default).

### Name, type and metadata

`ome.name` (a display name), `ome.type` (the downscaling method, as text) and `ome.metadata` (a dict describing it) are plain attributes.
They are written into the multiscale entry as they are.
Everything in them must be JSON-compatible, and `.metadata` must stay a dict.

```python
image.ome.name = "my image"
image.ome.type = "gaussian"
image.ome.metadata = {"method": "skimage.transform.pyramid_gaussian", "version": "0.25"}
```

### Writing an image with labels

Writing works exactly like the other metadata: wrap the Multiscale in an `OmeZarrGroup`.
`omero` and `image-label` are written to the right place in the group attrs automatically.
`update_attrs(group, version)` writes them to the group; `to_attrs(version)` returns the attrs if you want to write them yourself, but see [Updating attrs](#updating-attrs-that-already-hold-metadata).

Writing OME-Zarr 0.4, 0.5 or 0.6 requires every channel to have a `color` and a `window` with all four values (`start`, `end`, `min` and `max`).
If any of these are missing, `to_attrs` and `update_attrs` raise `ValueError`.
Only the experimental version `0.9.dev2` allows missing values for these properties.
`multiscale.lowest_lossless_ome_zarr_version` accounts for these `omero` requirements.

This example writes an image group, with its `omero` metadata, a `labels` group, and one label image inside (though with no actual data):

```python
import zarr

image_group = zarr.open_group("image.zarr", mode="w", zarr_format=3)
image_group.create_array("s0", shape=image["s0"].shape.values_tuple(), dtype="uint8")
cs.OmeZarrGroup.from_single(image).update_attrs(image_group, version="0.5")

labels_group = image_group.require_group("labels")
cs.OmeZarrGroup(children=(cs.ChildRef("label_multiscale", "cells"),)).update_attrs(labels_group, version="0.5")

cells_group = labels_group.require_group("cells")
cells_group.create_array("s0", shape=labels["s0"].shape.values_tuple(), dtype="uint16")
cs.OmeZarrGroup.from_single(labels).update_attrs(cells_group, version="0.5")
```

Reading it back:

```python
image_group = zarr.open_group("image.zarr", mode="r")

image_meta = cs.OmeZarrGroup.from_group(image_group)
assert image_meta.kind is cs.GroupKind.MULTISCALE
assert image_meta.multiscales[0].ome.omero is not None
assert image_meta.multiscales[0].ome.image_label is None
assert image_meta.maybe_subgroups == ("labels",)

labels_meta = cs.OmeZarrGroup.from_group(image_group["labels"])
assert labels_meta.kind is cs.GroupKind.LABELS
assert [child.path for child in labels_meta.children] == ["cells"]

cells_meta = cs.OmeZarrGroup.from_group(image_group["labels/cells"])
assert cells_meta.multiscales[0].ome.image_label.source == "../../"
```

(The `labels` group metadata is the one part of this that is not yet tested in clearscale, see [OME-Zarr features](ome_zarr.md#ome-zarr-features).)

Because `.ome` is part of the Multiscale, the usual read, modify, write cycle keeps `omero` and `image-label`:

```python
meta = cs.OmeZarrGroup.from_group(image_group)
meta.multiscales[0].ome.name = "renamed"
attrs = meta.to_attrs(version="0.5")

assert attrs["ome"]["omero"]["name"] == "my image"
assert attrs["ome"]["multiscales"][0]["name"] == "renamed"
```

## What to watch out for

### `omero` and `image-label` belong to the group, not the Multiscale

In a zarr group, they sit next to `multiscales`, so they apply to all multiscales stored in it.
clearscale puts them on `Multiscale.ome` anyway.
Technically, you work with a group-level key through a multiscale-level property.
This has some consequences:

* **Only `OmeZarrGroup` sees them.**
    `Multiscale.from_ome_zarr(...)` gets a single entry of `multiscales` and has no access to the keys next to it, and `Multiscale.to_ome_zarr(...)` does not produce them.
    Always go through `OmeZarrGroup.from_group`/`from_attrs` and `OmeZarrGroup.update_attrs`/`to_attrs`.
* **When a group holds several multiscales, they all get the same `.ome.omero` and `.ome.image_label`** when you read it.
    (OME-Zarr groups hold exactly one multiscale in practice, so this is rarely relevant.)
* **When writing, a group can only hold one of each.**
    If several Multiscales in the group set the same value, it is written once.
    A Multiscale that has none doesn't conflict with the others: if any Multiscale has one, it is written for the group.
    If two Multiscales hold *different* `omero` or *different* `image_label`, `to_attrs` and `update_attrs` raise a `ValueError`.
    Store them in separate groups, or make them agree.

### Copying: what carries `.ome` along, and what doesn't

`Multiscale.copy()` and the methods that return the "same" Multiscale in another form (`with_coordinate_system`, `with_keys`,...) copy `.ome`, including `omero` and `image_label`.
The copies are deep, so editing the copy's `.ome` doesn't change the original.

`Multiscale.derive` and `from_single` create a *new* Multiscale, so **`.ome` is not carried over from the original**: after `derive`, name, type, metadata, omero and image-label are all empty.
`as_derived_from(original)` likewise does not carry over `original.ome`.
This is because a derived image may have different channels, and a downscaled label image may have lost label values altogether, because a scale never sampled them.
It is easy to miss when you only want a different scale, though.
Copy over the parts that still apply:

```python
import copy

derived = image.derive("s0")
assert derived.ome.omero is None

derived.ome.omero = copy.deepcopy(image.ome.omero)  # only if the channels are unchanged!
```

### Instances read from one group are shared

When a group holds several multiscales, all of them point to the *same* `Omero` instance after reading, consistent with `omero` being a property of the group.
It is frozen, but its `extra` dict and the `extra` dicts of its channels are not, so editing one in place changes it for all of them.
`ImageLabel` instances are copied per Multiscale and don't have this problem.
Use `dataclasses.replace` as shown above (it makes copies), or `copy.deepcopy`, when in doubt.

### You are on your own with the content

As the warning at the top says, clearscale does not check `.ome` against anything: Not the number of `omero.channels` against the size of the `c` axis (an image with 3 channels and 1 `OmeroChannel`, or no `c` axis at all, is written without complaint, and viewers will not display it correctly), not label values against the label array, not `image_label` against the data type of the array.
Keep `.ome` in step with the data yourself, especially after selecting or reordering channels.

### Malformed `omero` metadata

When reading `omero` metadata, clearscale is permissive regarding missing properties that would technically be required by the dataset's version.
For example: Up to OME-Zarr 0.6, every channel was required to specify its `color`.
Only the experimental version 0.9.dev2 removes this requirement, but clearscale accepts missing `color` regardless of claimed version.

The lower requirement bound of 0.9.dev2 still applies:
The `omero` block must still be an object with a nonempty `channels` list.
If any channel entry is invalid under 0.9.dev2 (e.g. color not a hex-string, window is missing `start` or `end`), clearscale sets `multiscale.ome.omero` to `None` for the whole block, since skipping that entry would shift subsequent channel indices.
Enable debug logging to diagnose invalid channel lists and entries.

```python
import logging

logging.getLogger("clearscale").setLevel(logging.DEBUG)
logging.basicConfig()
```

`image-label` is more forgiving: entries without a usable `label-value` are skipped one at a time, an invalid `rgba` is ignored, and the rest of the entry is kept if it has other content. "label-value"-only declarations are kept too.

### Updating attrs that already hold metadata

In OME-Zarr 0.5 and 0.6, `to_attrs` returns a single `"ome"` key, and `zarr_group.attrs.update(...)` replaces it as a whole, so removed metadata is gone.
In OME-Zarr 0.4, `to_attrs` returns the flat top-level keys, and `attrs.update` only *adds* to them.
If you read a 0.4 group, set `ome.omero = None` (or `image_label = None`) and write it back with `attrs.update`, **the old `omero` (or `image-label`) key stays in the attrs**, and reading the group gives you the old values again.
The same happens with leftovers of the old version if you change the version of an existing group.

`OmeZarrGroup.update_attrs` avoids this: it writes like `attrs.update(to_attrs(...))`, but first removes OME-Zarr keys that the new metadata doesn't have.
Other attrs are left alone.
Use it whenever the group may already hold OME-Zarr metadata.

```python
legacy_group = zarr.open_group("legacy.zarr", mode="w", zarr_format=2)
legacy_group.create_array("s0", shape=(2, 64, 64), dtype="uint8")
cs.OmeZarrGroup.from_single(image).update_attrs(legacy_group, version="0.4")

legacy = cs.OmeZarrGroup.from_group(legacy_group)
legacy.multiscales[0].ome.omero = None

legacy_group.attrs.update(legacy.to_attrs(version="0.4"))
assert "omero" in legacy_group.attrs, "Stale: a plain `update` cannot remove a key"

legacy.update_attrs(legacy_group, version="0.4")
assert "omero" not in legacy_group.attrs
assert cs.OmeZarrGroup.from_group(legacy_group).multiscales[0].ome.omero is None
```

### Details of the written form

* Colors are normalized: upper case, no `#`. Window values are written as floats (`0.0`, not `0`).
* For OME-Zarr 0.4 and 0.5, `image-label` also contains a `"version"` key, as those specifications ask; for 0.6, it doesn't.
* When reading, a `colors` entry whose only content is an invalid `rgba` is dropped. If it has other keys, it is kept without color.
* A `LabelEntry()` without a color or properties is written as a value-only entry in `properties`, so the label stays declared when reading it back. An `ImageLabel()` with nothing in it still writes `image-label` (with a version entry in 0.4 and 0.5), which marks the multiscale as a label image.
* Colors and properties are merged into one entry per label value when reading, so the *order* of `labels` after reading is "all labels with colors, then all with only properties" and may differ from the file.
* OME-Zarr up to 0.6 allows further keys next to `rgba` in `colors` entries, but RFC-8 proposes to remove this.
    clearscale keeps those in `LabelEntry.properties`, under their key, prefixed with `"@color:"` (e.g. `"@color:foo"`), and writes them back into `colors`.
    You can ignore these keys, and we recommend not to use the prefix.
