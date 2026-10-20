# Image data and physical space

A `Multiscale` can describe more than the shapes of its arrays.
It also describes *where* its pixels are in space, and how that space relates to other spaces: the micrometer grid of the microscope, the physical world, a reference atlas, the image it was cropped or downsampled from.

`clearscale` models this with three concepts:

* **Coordinate systems**: named spaces with axes.
* **Spatial context**: the set of coordinate systems a Multiscale knows about, and how it connects to each of them.
* **Spatial relations**: descriptions of how to resample data to align them with another coordinate system.

You can ignore all of this if you only care about pixel size and shape.
It becomes useful when your output image has to stay correctly positioned relative to something else.

## Concepts

### What is a coordinate system?

A coordinate system is a named list of axes, each with an optional type (space, time, channel) and unit.
A point in a coordinate system is one number per axis.

Every Multiscale has exactly one coordinate system of its own, its "intrinsic" system.
This is the space in which the `Scale` values are correct: `pixel_size` and `translation` tell you how array indices map to coordinates in this space.
The intrinsic system has an automatically generated name that you normally never see or need.

A Multiscale can additionally know about any number of other coordinate systems, listed in `multiscale.coordinate_systems`.
Each of them is connected to the intrinsic system by one coordinate transformation, so coordinates can be converted between them.

### What are they for?

Typical uses:

* **Units and conventions**: the same data expressed in micrometers and in millimeters, or in a different axis order.
* **Stage alignment**: position and acquisition rotation on a microscope sample stage.
* **Raw data vs isometry**: for imaging modalities that acquire raw data with a skew or tilt, an affine warp that undoes the skew to obtain isometric data. 
* **Standardised alignments**: rotation into a standardised anatomical alignment (e.g. medical imaging).
* **Provenance of derived data**: a crop, or a 2D projection that remembers where it came from in the original image.

### What can they express?

For now, clearscale can express scaling, shifting, and changes to the axes (reordering, dropping, inserting), as well as chains of these.
These are the [spatial relations](#the-available-spatial-relations) described below.

Metadata formats differ in how much of this they can store.
OME-Zarr 0.6 stores all coordinate systems and the transformations between them.
Older versions (0.4, 0.5) can only store the Multiscale's own space and, in simple cases, one global scale and translation.
Only version 0.6 preserves the full spatial context when writing.

You can check `Multiscale.lowest_lossless_ome_zarr_version` to find out whether a particular Multiscale knows anything only OME-Zarr 0.6 can express.

## Adding coordinate systems to a Multiscale

Use `Multiscale.with_coordinate_system`.
Give the new system a name and say how it is reached from the Multiscale's own space with `reached_by`:

```python
from clearscale import Affine, Multiscale, PixelSize, Scale, Shape, Translation, Unit

multiscale = Multiscale.from_single(
    Scale(
        shape=Shape(z=40, y=512, x=512),
        pixel_size=PixelSize(z=2.0, y=0.5, x=0.5),
        unit=Unit(z="micrometer", y="micrometer", x="micrometer"),
    )
)

acquisition_shift_from_stage_origin = Translation(z=100.0, y=250.0, x=250.0)
multiscale = multiscale.with_coordinate_system(
    "stage",
    # The stage is reached by *undoing* the acquisition's shift, so the Translation needs to be inverted
    reached_by=acquisition_shift_from_stage_origin.inverted(),
)

multiscale = multiscale.with_coordinate_system(
    "millimeter",
    # One millimeter is 1000 micrometer
    reached_by=Affine.from_factor({"z": 1000, "y": 1000, "x": 1000}),
    unit=Unit(z="mm", y="mm", x="mm"),
)

assert set(multiscale.coordinate_systems) == {"stage", "millimeter"}
```

Things to know:

* `reached_by` may be a single spatial relation or a list of them.
    Lists are applied in order, first to last.
    Omit it if the new system is simply the same space under another name.
* Both new systems above are connected directly to the Multiscale's own space.
    You describe each one relative to the Multiscale, not relative to each other.
* `unit` and/or `ome_zarr_axes` set the axis properties of the new system.
    Properties that you don't set are carried over from the Multiscale's own axes.
* Names must be unique among the Multiscale's coordinate systems.
* Like most methods in clearscale, `with_coordinate_system` returns a new object and leaves the original untouched.
    The result is *spatially identical* to the original: it is the same space, now with extra knowledge about its surroundings.

## Deriving a new Multiscale

The usual image-processing flow is: load a Multiscale, pick one of its scales, process the data, and write the result.
"Deriving" in clearscale means to record this provenance of where the output data came from.

```python
source = OmeZarrGroup.from_group(zarr_group).multiscales[0]   # has coordinate systems, e.g. "stage"

result = source.derive("s1")
```

`derive` creates a new Multiscale from one scale of `source` (here `"s1"`), optionally expanded by a scaling `blueprint`, exactly like `Multiscale.from_single`.

What makes it different from `from_single` is that the result stays in the spatial context of `source`:

All the source's coordinate systems (`"stage"` and so on) are available on the result too, with their connections preserved.

When writing OME-Zarr 0.6, this is stored in the output metadata.

### Deriving by a spatial relation
`derive` is a convenience shortcut for simple cases.

If something else changed between the source and the result — maybe you cropped the data, projected it along one axis, or otherwise transformed it — build the result yourself and connect it to the source using `as_derived_from`:

```python
from clearscale import Multiscale, PixelSize, ProjectionTo, Scale, Shape

projection = (
    Multiscale
    .from_single(
        Scale(shape=Shape(y=512, x=512), pixel_size=PixelSize(y=0.5, x=0.5)),
        scale_key="s1"
    )
    .as_derived_from(volume, by=ProjectionTo("yx"))
)
```

`projection.as_derived_from(volume, by=...)` says "`projection` was derived from `volume`, in this way".

`by` takes the same spatial relations as `with_coordinate_system(reached_by=...)`, because the question is the same: "how do I get from one space to the other?"

The direction to think about is the opposite in this case, as the method and parameter names suggest:

* `with_coordinate_system("other", reached_by=...)`: self --relation-> "other" (*How do you reach the new coordinate system, from me?*)
* `as_derived_from(source, by=...)`: source --relation-> self (*How do you reach me, from `source`?*)

`as_derived_from` uses the answer to carry the source's other coordinate systems across wherever they remain well-defined.

Not every relation can be carried over.
A relation that drops an axis loses information, so connections to the source's other coordinate systems can only be carried over where they remain well-defined.

Without `by`, `as_derived_from` requires both Multiscales to have identical axes.
It also fills in axis properties you left blank (type, unit, ...) from `volume`.

## Spatial relation vs coordinate transformation

A **spatial relation** answers the question: *If I have an image in space A, and I do `<operation>` to it, how is the resulting image's space B related to A?*

The operation you actually performed on the data could be transposing the axes, shifting the origin by cropping, or an affine warp.

This is exactly the inverse of coordinate arithmetic, if this is what you are used to:

* shifting by a `Translation` subtracts that from A-coordinates (shifted B's origin is somewhere inside A-space),
* an `Affine` expresses A-coordinates in terms of B-coordinates (`x_A = linear @ x_B + translation`, a.k.a. "pull" or "backward" direction, as used for sampling B from A).

In addition, also note the opposite directions of thinking in `as_derived_from` vs `with_coordinate_system`:

* When deriving, the spatial relation says, *I made the derived output by doing `<operation>`*.
* When specifying a new coordinate system, the spatial relation says, *To reach this other space, you need to do `<operation>`*

For example, a numerically identical crop shift could be expressed in two different ways:

```python
crop_offset = PixelOffset(z=0, y=100, x=200)
crop_translation = crop_offset * source["s0"].pixel_size

# "I made a new image by cropping `source["s0"]`"
crop_expressed_as_derivation = (
    Multiscale
    .from_single(source["s0"], scale_key="s0")
    .as_derived_from(source, by=crop_translation)
)

# "You can reach original_image by undoing a crop that I know about"
crop_expressed_as_coordinate_system = processed_image.with_coordinate_system(
    "original_image", 
    reached_by=crop_translation.inverted()  # <- same translation, but inverted
)
```

This example is a bit contrived, because it is unlikely that the same shift (in physical units) applies to two different scenarios like this.

This is just to illustrate that the resulting coordinate transformations

* `(crop_expressed_as_derivation --> source) = [0.0, 50.0, 100.0]`
* `(crop_expressed_as_coordinate_system --> "original_image") = [0.0, 50.0, 100.0]`

are arithmetically identical, despite passing `Translation` to `as_derived_from` and `Translation.inverted()` to `with_coordinate_system`.

(Although `crop_expressed_as_derivation --> source` is internally stored as the inverse `(source --> crop_expressed_as_derivation) = [0.0, -50.0, -100.0]`, because this is the direction of the derivation statement.)

## Available spatial relations

All spatial relations can be used in `with_coordinate_system(reached_by=...)` and `as_derived_from(by=...)`.
To chain relations, pass a list; they are applied in order.

| Relation              | What you did                     | Coordinates in the new space                    |
|-----------------------|----------------------------------|-------------------------------------------------|
| `Affine`              | Resampled with an affine matrix  | inverse affine applied to the old coordinate    |
| `Translation`         | Moved the origin                 | old coordinate - translation                    |
| `PermutationTo`       | Reordered axes                   | Same values, new axis order                     |
| `ProjectionTo`        | Dropped and/or inserted axes     | Dropped axes are gone, inserted axes start at 0 |
| `AxisRearrangementTo` | Any combination of the two above | See below                                       |

### `Affine`

* When deriving: "The new image's space relates to the old image's space by this affine".
* When adding coordinate systems: "The old image's coordinates are this affine of the new coordinate system's coordinates".

`Affine(linear=..., translation=...)` maps coordinates of the *new* space to coordinates of the *old* one, in physical units: `x_old = linear @ x_new + translation` (column vectors, axes in the order of the affine).
The columns of `linear` are the new space's axes written in old-space coordinates, and `translation` is the new origin in the old space.
The matrix must be invertible.
The relation is stored in the metadata in the opposite direction, from the old space to the new one.

An `Affine` is only a *true* statement if it carries information that the pixel sizes and translations of the Scales do not already hold.
Resampling is not such information: a downscaled image has a larger pixel size, and that is all there is to say about it.
Do not relate a Multiscale to the one it was downscaled from with a scaling `Affine`. 
The two spaces are both physical, so they are the same space, and the correct scale between them is 1.
Use it for relations the Scales cannot express, such as a rotation or a flip, or for spaces whose coordinates are measured with a different yardstick.

Unit conversion is the common case of the latter: `Affine.from_factor({"x": 1000})` says "a coordinate in the new space times 1000 is the coordinate in the old space", which relates a millimeter space to a micrometer space.
`Affine.from_factor` takes a `Factor` or a plain `{axis: value}` mapping; use `.with_axes(...)` on the result to add axes that are not scaled.

### `Translation`

* When deriving: "The new image's origin is at this coordinate in the old image".
* When adding coordinate systems: "The new coordinate system's origin is at this coordinate relative to the image origin".

`Translation` is in physical units (use `PixelOffset * PixelSize` to convert a pixel offset, such as a crop offset).

Use it for crops, for placing tiles into a shared space, and for moving to the origin of a reference.

### `PermutationTo`

`PermutationTo("xyz")` says "the same axes, now in this order".
Source and target must contain exactly the same axes, otherwise it raises.
It cannot insert or drop.

### `ProjectionTo`

`ProjectionTo("yx")` says "the new image has exactly these axes".
Axes missing from the target are dropped (e.g. a maximum projection along z), axes not present in the source are inserted (e.g. adding a channel axis).
The axes that are present on both sides must remain in their relative order, otherwise it raises.
It cannot change axis order.

Dropping an axis loses information, so a relation like this can't be reversed.

### `AxisRearrangementTo`

`AxisRearrangementTo("xcy")` is the general version: 
the result has exactly these axes in this order, whichever were dropped, inserted or reordered to get there. 
When applied to `"zyx"`, `AxisRearrangementTo("xcy")` drops `z`, inserts `c`, and reorders `y` and `x`. 
If you aren't sure which of the three you need, use this one.

### Combining relations

Relations that act on values (`Affine`, `Translation`) and relations that act on axes can be chained in one list:

```python
multiscale.with_coordinate_system(
    "yx_preview",
    reached_by=[ProjectionTo("yx"), Affine.from_factor({"y": 1000, "x": 1000})],
)
```

Each relation sees the axes that the previous one produced. Here, `ProjectionTo` first reduces `zyx` to `yx`, after which the `Affine` only needs to talk about `y` and `x`.

## Beyond a single Multiscale

Everything above describes the surroundings of *one* Multiscale. To relate several Multiscales to each other (stitching tiles into a mosaic, registering images to a common reference), clearscale has `Scene`, which uses the same concepts.
`Scene` is still in development, its API will change, so please let us know if you need/want it for your work.