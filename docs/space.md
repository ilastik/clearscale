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

For now, clearscale can express affine mappings (such as unit conversions, rotations and flips), shifting, and changes to the axes (reordering, dropping, inserting), as well as chains of these.
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

## Reading relations back

Relations can be read back from a Multiscale's coordinate systems, in the direction in which they are needed:

```python
multiscale = multiscale.with_coordinate_system("stage", reached_by=relation)

# multiscale --relation--> "stage"
relation = multiscale.relation_to_coordinate_system("stage")
```

Coordinate systems are carried over by derivation (see above), so they can still be retrieved after processing.
Say `source` has axes `"zyx"`, and a coordinate system named `"stage"`, reached by a `Translation`, and the result is a maximum projection of it:

```python
result = projection.as_derived_from(source, by=ProjectionTo("yx"))

# "stage" --relation--> result
relation = result.relation_from_coordinate_system("stage")

# e.g. SpatialRelationSequence((ProjectionTo("yx"), Translation(y=250.0, x=250.0)))
```

For invertible connections like `Translation`, both directions work and return inverse relations.

For non-invertible connections like axis drops, only the direction they were recorded in works. In the example, the projection dropped `z` from `source` to `result`, so there is no way from `result` back to `source` or `"stage"`: `result.relation_to_coordinate_system("stage")` raises `CannotConvertToSpatialRelationError`.

* The result is a single relation, a list-like sequence of relations, or `None`, if the other space is the same space under another name.
* It describes the same transformation as the one that was specified, but in canonical form: Relations may be merged or reordered (renamings come last), and values may differ by floating point rounding.
* Units and other axis properties of the other coordinate system are not part of a relation. Use `coordinate_system_ome_zarr_axes(name)` for these.
* `CannotConvertToSpatialRelationError` is also raised for connections that no relation expresses, such as displacement fields read from OME-Zarr metadata written by other tools. Rotations, shears and flips are expressed as `Affine`.
* `NoSuchCoordinateSystemError` is raised for names that are not in `coordinate_systems` (including the Multiscale's own space).

## Spatial relation vs coordinate transformation

A **spatial relation** answers the question: *If I have an image in space A, and I do `<operation>` to it, how is the resulting image's space B related to A?*

The operation you actually performed on the data could be cropping, shifting the origin, transposing the axes.

This is exactly the inverse of coordinate arithmetic, if this is what you are used to:

* shifting by a `Translation` subtracts that from A-coordinates (shifted B's origin is somewhere inside A-space),
* an `Affine` expresses A-coordinates in terms of B-coordinates (`x_A = linear @ x_B + translation`, a.k.a. "pull" or "backward" direction, as used for sampling B from A).

In addition, also note the opposite directions of thinking in `as_derived_from` vs `with_coordinate_system`:

* When deriving, the spatial relation says, *I made the derived output by doing `<operation>`*.
* When specifying a new coordinate system, the spatial relation says, *To reach this other space, you need to do `<operation>`*

For example, a numerically identical crop shift could be expressed in two different ways:

```python
from dataclasses import replace

crop_offset = PixelOffset(z=0, y=100, x=200)
crop_shape = Shape(z=40, y=128, x=128)
crop_translation = crop_offset * source["s0"].pixel_size

# "I made a new image by cropping `source["s0"]`"
crop_expressed_as_derivation = (
    Multiscale
    .from_single(replace(source["s0"], shape=crop_shape), scale_key="s0")
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

### Units

clearscale does not convert units.

Trying to relate a source and target with different units in `with_coordinate_system` or `as_derived_from` generally raises a `ValueError`.

Intentional conversions from one unit to another need to be explicitly specified as a relation.
Without a conversion, a coordinate in one space would be assumed to equal the same number in the other.

State the conversion explicitly with a scaling `Affine` instead (see [Affine](#affine), specifically [Unit conversion](#unit-conversion)).

Things to know:

* An empty unit (like `Unit(x="")`) means "unspecified" and never causes an error. Axes that exist on only one side are not compared.
* Different spellings of the same unit count as the same unit: `"nm"` and `"NANOMETER"`, or `"um"`, `"µm"` and `"micron"`. Unrecognized units only match by string equality (e.g. `"a.u."` does not match `"A.U."`).
* Any `Affine` bypasses the check. clearscale does not verify that it converts correctly.

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
| `AxisRenaming`        | Renamed axes                     | Same values, new axis names                     |

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
Use it for relations the Scales cannot express, such as a rotation or a flip.

Affines with a scaling component are correct only in two main cases:
* In `with_coordinate_system` if the newly added coordinate system has a different pixel size than the Multiscale it is connected to.
* When converting units: `Affine.from_factor({"x": 1000})` can represent e.g. going from `Unit(x="micrometer")` to `Unit(x="millimeter")`.

#### Example: expansion microscopy

An expanded sample is imaged at its expanded size.
To record the sample's original dimensions in the output metadata, add a coordinate system for them:

```python
from clearscale import Affine, Factor

expansion_factor = Factor.uniform("zyx", 4.2)
multiscale = multiscale.with_coordinate_system(
    "unexpanded",
    reached_by=Affine.from_factor(expansion_factor),
)

# Pixel size of the image in unexpanded sample space:
unexpanded_pixel_size = multiscale["s0"].pixel_size / expansion_factor
```

Note that the factor here *divides* the pixel size, because it represents a change of coordinate system rather than an image resampling operation.
Downsampling changes how many pixels cover the same physical distance; a scaling Affine changes the coordinate values assigned to the same pixels, without resampling the image.

#### A matrix that places an image in a world space

Registration tools may report where an image lies in a reference coordinate system, using a transformation such as `x_world = matrix @ x_image + shift`, with coordinates expressed in physical units.

This is the opposite direction from Affine's "pull" convention, `x_source = linear @ x_target + translation`. Here, `x_source` corresponds to the registration tool's `x_world`, and `x_target` to its `x_image`.

When a registration tool provides an image-to-world transformation in physical coordinates, invert it before passing it as reached_by:

```python
from clearscale import Affine

# Registration tool output: "image -> world", in physical coordinates
registration_matrix = [[0, -1, 10], [1, 0, 20]]

# "image -> world" in registration terms is "world -> image" in Affine terms
world_to_image = Affine.from_array(registration_matrix, "yx")

multiscale.with_coordinate_system("world", reached_by=world_to_image.inverted())
```

#### Unit conversion

`Affine.from_unit_conversion` knows all space and time units that OME-Zarr recognises (`"nanometer"`, `"millisecond"`, `"inch"`, ...) and common spellings of them (`"nm"`, `"um"`, `"µm"`, `"micron"`, `"Å"`, `"ms"`, ...).
It returns an `Affine` that converts values from the source unit to the target:

```python
from clearscale import Multiscale, Scale, Shape, Unit, Affine

source_ms = Multiscale.from_single(
    Scale(
        shape=Shape.all_singletons("xy"),
        unit=Unit(x="micrometer", y="micrometer")
    )
)

target_unit = Unit(x="nanometer", y="nanometer")

annotated = source_ms.with_coordinate_system(
    "nm_target",
    unit=target_unit,
    reached_by=Affine.from_unit_conversion(
        source_ms.unit,
        target_unit
    )
)
```

When converting units between different Multiscales, you can use `Scale.with_unit_converted` to create the base for the converted Multiscale:

```python
from clearscale import Multiscale, Scale, Shape, PixelSize, Affine

original_base = Scale(
    shape=Shape(zip("zyx", (23, 256, 256))),
    pixel_size={"z": 2.3, "y": 10, "x": 10},
    unit={"z": "um", "y": "nm", "x": "nm"}
)
original_ms = Multiscale.from_single(original_base, scale_key="s0")

# Scale.with_unit_converted also updates pixel_size and translation, so the resulting Scale is geometrically identical
converted_base = original_base.with_unit_converted({"z": "nm"})
assert converted_base.pixel_size == PixelSize(z=2300.0, y=10.0, x=10.0), "2.3 micron = 2300 nm"

converted_ms = Multiscale.from_single(converted_base, scale_key="s0")
conversion_relation = Affine.from_unit_conversion(original_ms.unit, converted_ms.unit)

derived = converted_ms.as_derived_from(original_ms, by=conversion_relation)
```

clearscale maintains and writes unit strings as you provide them: `"nm"` stays `"nm"`, because tool interoperability can depend on the exact spelling.
For OME-Zarr, unless your workflow strictly requires a custom spelling, you should use the standard forms `"nanometer"`, `"micrometer"`, `"second"`, etc.

You can use `Unit.normalized()` to clean up if you prefer to use abbreviations internally but want to write compliant metadata:

```python
scale = Scale(
    shape=Shape(y=512, x=512),
    unit=Unit(y="um", x="um")
)

cleaned_up = scale.with_unit_replaced(scale.unit.normalized())

assert cleaned_up.unit == Unit(y="micrometer", x="micrometer")
```

For display, the other direction is `Unit.abbreviated()`: it gives every recognized unit its standard short symbol (`"nm"`, `"µm"`, `"min"`).
The micro symbol is the micro sign U+00B5, which clearscale treats as the same unit as `"um"` and `"micron"`.
Replace it if you need pure ASCII.

```python
assert Unit(x="micrometer", t="second").abbreviated() == Unit(x="\u00b5m", t="s"), "for display"
assert Unit(x="\u00b5m", t="s").abbreviated().normalized() == Unit(x="micrometer", t="second"), "roundtrip"
```

#### Matrices in pixel index space

Image resampling may work with pixel indices, not physical coordinates.
`scipy.ndimage.affine_transform(image, matrix, offset)` is a classic example: output pixel `i_out` takes its value from input position `matrix @ i_out + offset`.
That is already the direction of `Affine`, with the output image as the new space and the input image as the old space.
Index space has no units, so to convert it into physical coordinates, `Affine` needs to know both pixel grids.
Pass the `Scale` of each image:

```python
affine = Affine.from_linear_and_offset(
    matrix,
    offset,
    "yx",
    source=input_scale,
    target=output_scale
)
```

`source` is the image the values are taken from, `target` is the image that is produced, and both must have exactly the given axes.
`Affine.from_array` takes the same arguments for a combined `[matrix | offset]` array.
Index `i` refers to the *center* of pixel `i`.

The resulting physical matrix is `source.pixel_size * matrix / target.pixel_size` (axis-wise), and the physical translation is `source.pixel_size * offset + source.translation - physical_matrix @ target.translation`.

Since each Scale's pixel size is in its own unit, this also takes care of unit differences: an identity index-matrix "resampling" a 0.015 micrometer grid from a 15 nm source grid becomes a unit-converting Affine with a scaling factor of 1000.
If the target is just a downscaled version of the source and both Scales are accurate, the result is the identity: Scales already describe the resampling.

If you need a physical `Affine` for a coordinate system that has no `Scale` (i.e. for `with_coordinate_system`), describe the grid your matrix refers to with a placeholder `Scale`; its shape is not used:

```python
grid = Scale(shape=Shape.all_singletons("yx"), pixel_size=PixelSize(y=0.5, x=0.5))
```

#### Matrices from other sources

Without `source` and `target`, `from_array` and `from_linear_and_offset` take the numbers as they are.
If your matrix does not follow the convention above, convert it first:

* **Direction**: If it maps old coordinates to new coordinates, invert it (`.inverted()` after constructing).
* **Row vectors**: `Affine` uses column vectors (`x_old = linear @ x_new`). For a matrix written for row vectors (`x_new @ matrix`), transpose it.
* **Index vs physical coordinates**: If it works on pixel indices, pass `source` and `target`, as above.
* **Axis order**: Rows and columns follow the `axes` you pass. Reorder if your matrix uses another order (for example `xyz` for data that is `zyx`).
* **Pixel reference point**: `Affine` treats index `i` as the pixel center. If your tool places integer indices at pixel corners, the offset needs a half-pixel correction: `offset_center = offset_corner + 0.5 * (matrix @ ones - ones)`.

### `Translation`

* When deriving: "The new image's origin is at this coordinate in the old image".
* When adding coordinate systems: "The new coordinate system's origin is at this coordinate relative to the image origin".

`Translation` is in physical units (use `PixelOffset * PixelSize` to convert a pixel offset, such as a crop offset).

Use it for crops, for placing tiles into a shared space, and for moving to the origin of a reference.

For example, a crop of a Multiscale `source`, with its own pyramid:

```python
from dataclasses import replace
from clearscale import BlueprintShapes, Multiscale, PixelOffset, Shape

s0 = source["s0"]
crop_offset = PixelOffset(z=10, y=100, x=200)  # where the crop starts, in s0 pixels
crop_shape = Shape(z=20, y=128, x=128)

# The data you write is s0_data[10:30, 100:228, 200:328] and its pyramid.
cropped = Multiscale.from_single(
    replace(s0, shape=crop_shape),  # same pixel size and unit, but the cropped shape
    blueprint=BlueprintShapes.downscale_powers_of_2_xyz(base_shape=crop_shape, rounding="ceil"),
).as_derived_from(source, by=crop_offset * s0.pixel_size)  # PixelOffset * PixelSize = Translation
```

The cropped Multiscale's own origin is the crop corner, and the relation says where that corner lies in the source:
With a pixel size of 2.0/0.5/0.5 micrometer, `by` is `Translation(z=20.0, y=50.0, x=100.0)`.

If you set `Scale.translation` manually, make sure to keep this consistent (i.e. mutually exclusive) with any `Translation` relations. `Scale.translation` records shifts that come from resampling (`translating=`), while the offset of a crop belongs in the relation.
Additionally writing the crop offset into the cropped image's `Scale.translation` would count it twice.

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

### `AxisRenaming`

`AxisRenaming({"z": "Z"})` (or `AxisRenaming(z="Z")`) says "the same image, but this axis is called differently now".
The data and its values don't change, only the axis name does, and axis properties (type, unit, ...) stay with the renamed axis.
Axes the renaming doesn't mention keep their name.
All renames apply at once, so `AxisRenaming(y="x", x="y")` swaps the names of the two axes without moving any data, unlike `PermutationTo("xy")`, which swaps the data.

Renaming an axis that doesn't exist, or renaming two axes to the same name, raises.
Relations later in a list must use the new names.

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