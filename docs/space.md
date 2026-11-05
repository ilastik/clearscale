# Image data and physical space

A `Multiscale` can describe more than the shapes of its arrays.
It also describes *where* its pixels are in space, and how that space relates to other spaces: the micrometer grid of the microscope, the physical world, a reference atlas, the image it was cropped or downsampled from.

`clearscale` models this with three concepts:

* **Coordinate systems**: named spaces with axes.
* **Spatial context**: the set of coordinate systems a Multiscale knows about, and how it connects to each of them.
* **Spatial relations**: connections from one space to another.

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
    reached_by=acquisition_shift_from_stage_origin,
)

converted_unit = Unit(z="mm", y="mm", x="mm")
multiscale = multiscale.with_coordinate_system(
    "millimeter",
    # Unit conversion is a scaling affine relation
    reached_by=Affine.from_unit_conversion(multiscale.unit, converted_unit),
    unit=converted_unit,
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

"Deriving" in clearscale saves you the manual labor of carrying over properties from the source metadata to the output metadata.

You might write only a single data array whose shape, pixel size, axis units and so on correspond exactly to the Scale whose data was processed.
In this simple case, use `.derive`:

```python
source = OmeZarrGroup.from_group(zarr_group).multiscales[0]   # has coordinate systems, e.g. "stage"

result = source.derive("s1")
```

`derive` creates a new Multiscale from one scale of `source` (here `"s1"`), optionally expanded by a scaling `blueprint`, exactly like `Multiscale.from_single`.

What makes it different from `from_single` is that the result inherits the spatial context of `source`.
All the source's coordinate systems (`"stage"` and so on) are available on the result too, with their connections preserved.

When writing OME-Zarr 0.6, this is stored in the output metadata.

> [!NOTE]
> "Deriving" does *not* establish a connection between the source Multiscale and the derived Multiscale.
> Multiscale-Multiscale relations are only possible in a [Scene](scene.md).

### Deriving with a spatial relation
`derive` is a convenience shortcut for simple cases where the properties of the extracted Scale do not change.

If something else changed between the source and the result — maybe you cropped the data, projected it along one axis, or otherwise transformed it — build the result yourself to accurately describe the result.
Then use `as_derived_from` to inherit the unchanged properties, including any coordinate systems the parent might define.

In `as_derived_from`, the source acts as a new coordinate system.
Specify how to reach the source from the result just like in `with_coordinate_system`: `result --reached_by--> source`.
The relation you specify is chained before the original relation between the source and its coordinate systems:

```python
from clearscale import Multiscale, PixelSize, ProjectionTo, Scale, Shape

projection = (
    Multiscale
    .from_single(
        Scale(shape=Shape(y=512, x=512), pixel_size=PixelSize(y=0.5, x=0.5)),
        scale_key="s1"
    )
    .as_derived_from(volume, reached_by=ProjectionTo(volume.axes))
)
```

Without `reached_by`, both Multiscales must have identical axes.
Blank axis properties on the result (type, unit, ...) are filled from the corresponding source axes.

Source-only axes are inserted at coordinate 0.
For example, if the result is a slice extracted at z=3.5, use `[ProjectionTo(volume.axes), Translation(z=3.5, y=0, x=0)]` (note Translation must be in physical units, i.e. `PixelOffset * PixelSize`).
The expressive capabilities are limited: For example, it is not possible to distinguish a maximum projection (which represents all source slices) from a slice extracted at z=0.

Axis properties are transferred from source to derived, which is the opposite direction of the relation.
Non-invertible relations (e.g. Affine with a singular matrix) cannot be used in `as_derived_from` as a consequence.

Source coordinate systems are carried over through `derived -> source -> system`.
If a system came from another tool that stored it in the opposite direction `source <- system`, it can only be transferred to `derived` if either the `source <- system` connection or the provided `derived --reached_by--> source` can be inverted.
A source connection stored in the reverse direction can still require an inverse that does not exist.

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
result = projection.as_derived_from(source, reached_by=ProjectionTo(source.axes))

# result --relation--> "stage"
relation = result.relation_to_coordinate_system("stage")

# e.g. SpatialRelationSequence((ProjectionTo("zyx"), Translation(z=100.0, y=250.0, x=250.0)))
```

* The result is a single relation, a list-like sequence of relations, or `None`, if the other space is the same space under another name.
* It describes the same transformation as the one that was specified, but in canonical form: Relations may be merged or reordered (renamings come last), and values may differ by floating point rounding.
* Units and other axis properties of the other coordinate system are not part of a relation. Use `coordinate_system_ome_zarr_axes(name)` for these.
* `CannotConvertToSpatialRelationError` is also raised for connections that no relation expresses, such as displacement fields read from OME-Zarr metadata written by other tools. Rotations, shears and flips are expressed as `Affine`.
* `NoSuchCoordinateSystemError` is raised for names that are not in `coordinate_systems` (including the Multiscale's own space).

Tools other than clearscale may write connections in the opposite direction (`"stage" -> multiscale`).
In this case, `relation_to_coordinate_system` returns the inverse of the stored connection for invertible connections like `Translation`.
If the connection is not invertible, `relation_to_coordinate_system` raises `CannotConvertToSpatialRelationError`.
To retrieve the original as written, or retrieve non-invertible connections, use the opposite counterpart `relation_from_coordinate_system`.

## Spatial relation vs coordinate transformation

A spatial relation maps source coordinates to target coordinates.
`Translation` adds its values; `Affine` applies `x_target = linear @ x_source + translation`.
Lists apply first to last, so Affines compose as `A2 @ A1` for `[R1, R2]`.

Both `as_derived_from` and `with_coordinate_system` describe how to reach another coordinate system from the
current image. A crop whose corner is at `(0, 50, 100)` in source coordinates reaches its source through
`Translation(z=0, y=50, x=100)`, whether recorded as a derivation or as a new coordinate system.

Resampling uses the opposite direction: For example, scipy asks where each output pixel samples the input.
Refer to Affine constructor method docstrings for details on how to convert.

### Units

clearscale does not convert units.

Trying to relate a source and target with different units in `with_coordinate_system` or `as_derived_from` generally raises a `ValueError`.

Intentional conversions from one unit to another need to be explicitly specified as a relation.
Without a conversion, a coordinate in one space would be assumed to equal the same number in the other.
The appropriate relation for a unit conversion is a scaling `Affine` (see [Affine](#affine), specifically [Unit conversion](#unit-conversion)).

Things to know:

* An empty unit (like `Unit(x="")`) means "unspecified" and never causes an error. Axes that exist on only one side are not compared.
* Different spellings of the same unit count as the same unit: `"nm"` and `"NANOMETER"`, or `"um"`, `"µm"` and `"micron"`. Unrecognized units only match by string equality (e.g. `"a.u."` does not match `"A.U."`).
* Any `Affine` bypasses the check. clearscale does not verify that it converts correctly.

## Available spatial relations

All spatial relations can be used in `with_coordinate_system(reached_by=...)` and `as_derived_from(reached_by=...)`.
To chain relations, pass a list; they are applied in order.

| Relation              | Coordinate operation             | Coordinates in the target space                 |
|-----------------------|----------------------------------|-------------------------------------------------|
| `Affine`              | Apply an affine mapping          | linear @ source coordinate + translation        |
| `Translation`         | Add a physical shift             | source coordinate + translation                 |
| `PermutationTo`       | Reordered axes                   | Same values, new axis order                     |
| `ProjectionTo`        | Dropped and/or inserted axes     | Dropped axes are gone, inserted axes start at 0 |
| `AxisRearrangementTo` | Any combination of the two above | See below                                       |
| `AxisRenaming`        | Renamed axes                     | Same values, new axis names                     |

### `Affine`

`Affine(linear=..., translation=...)` maps physical coordinates from source to target:
`x_target = linear @ x_source + translation` (column vectors, axes in the affine's order).
Its values are stored in the same direction as coordinate transformations, also known as "push" or "forward".
The columns of `linear` are the source space's axes, rows are target axes, and `translation` is the source origin in the target space.
Singular matrices are accepted for forward mappings.
Invertibility is required when retrieving the reverse, explicitly converting between push and pull mappings, or carrying properties in `as_derived_from`.

An `Affine` is only a *true* statement if it carries information that the pixel sizes and translations of the Scales do not already hold.
Resampling is not such information: a downscaled image has a larger pixel size, and that is all there is to say about it.
Do not relate a Multiscale to the one it was downscaled from with a scaling `Affine`.
The two spaces are both physical, so they are the same space, and the correct scale between them is 1.
Use it for relations the Scales cannot express, such as a rotation or a flip.

Affines with a scaling component are correct only in two main cases:
* When the target system uses a different coordinate scale, such as the original dimensions of an expanded sample.
* When converting units: `Affine.from_factor({"x": 0.001})` can represent e.g. going from `Unit(x="micrometer")` to `Unit(x="millimeter")`. `Affine.from_unit_conversion(source_unit, target_unit)` helps to get the direction right.

Note that `Affine.from_factor` uses the provided values as-is, but its "push" convention is the opposite direction of clearscale's `Factor` class.
If `Factor(x=2)` is "downscale by 2", then the corresponding downsampling (pull) Affine is `Affine.from_factor(factor.inverted())`.

#### Example: expansion microscopy

An expanded sample is imaged at its expanded size.
To record the sample's original dimensions in the output metadata, add a coordinate system for them:

```python
from clearscale import Affine, Factor

expansion_factor = Factor.uniform("zyx", 4.2)  # Factor here does *not* mean "downscale by 4.2" like usual
multiscale = multiscale.with_coordinate_system(
    "unexpanded",
    reached_by=Affine.from_factor(expansion_factor),
)

# Pixel size of the image in unexpanded sample space:
unexpanded_pixel_size = multiscale["s0"].pixel_size * expansion_factor
```

Note that the factor here *divides* the pixel size, because it represents a change of coordinate system rather than an image resampling operation.
Downsampling changes how many pixels cover the same physical distance; a scaling Affine changes the coordinate values assigned to the same pixels, without resampling the image.

#### A matrix that places an image in a world space

Registration tools may report where an image lies in a reference coordinate system, using a transformation such as `x_world = matrix @ x_image + shift`, with coordinates expressed in physical units.

This is the same as Affine's push convention.
Pass the physical image-to-world mapping directly as `reached_by`:

```python
from clearscale import Affine

# Registration tool output: "image -> world", in physical coordinates
registration_matrix = [[0, -1, 10], [1, 0, 20]]

image_to_world = Affine.from_array(registration_matrix, "yx")

multiscale.with_coordinate_system("world", reached_by=image_to_world)
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
# source is converted_ms and target is original_ms.
# The relation needs to be *from* converted *to* the original as a coordinate system.
conversion_relation = Affine.from_unit_conversion(converted_ms.unit, original_ms.unit)

derived = converted_ms.as_derived_from(original_ms, reached_by=conversion_relation)
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

Image resampling may work with pixel indices, not physical coordinates, and often works with "pull" or "inverse" matrices.
`scipy.ndimage.affine_transform(image, matrix, offset)` is a classic example: output pixel `i_out` takes its value from input position `i_in = matrix @ i_out + offset`.

All Affine helpers assume the provided matrix is in the "push" or "forward" direction.
For a scipy pull matrix, this means the *input* acts as the *target* and the output as the source.
The Affine returned by the constructor will then be accurately converted to physical, but still in "pull" direction, so it needs to be inverted to obtain an accurate "push" Affine relation:

```python
affine = Affine.from_linear_and_offset(
    matrix,
    offset,
    "yx",
    source=input_scale,
    target=output_scale
)
```

The resulting Affine maps physical input coordinates to physical output coordinates.
`Affine.from_array` accepts a combined `[matrix | offset]` array with exactly the same convention.
Without source/target, both constructors take values as is and assume they are physical; invert the resulting Affine if those values are pull values.

For a push index mapping `index_target = matrix @ index_source + offset`, grid conversion gives
`physical_matrix = target.pixel_size * matrix / source.pixel_size` (axis-wise) and
`physical_translation = target.pixel_size * offset + target.translation - physical_matrix @ source.translation`.
Integer indices refer to pixel centers. Both grids must have exactly the given axes in the same order.

Since each Scale's pixel size is in its own unit, this also takes care of unit differences: an identity index-matrix mapping a 15 nm source grid to a 0.015 micrometer target grid becomes a unit-converting Affine with a physical coefficient of 0.001.
If the target is just a downscaled version of the source and both Scales are accurate, the result is a physical identity: Scales already describe the image scaling.

If you need a physical `Affine` for a coordinate system that has no `Scale` (i.e. for `with_coordinate_system`), describe the grid your matrix refers to with a placeholder `Scale`; its shape is not used:

```python
grid = Scale(shape=Shape.all_singletons("yx"), pixel_size=PixelSize(y=0.5, x=0.5))
```

`to_linear_and_offset` returns a push matrix and offset, optionally converted to index space.
For scipy arguments, again, explicitly invert the Affine and swap the input and output grids.
You choose the output grid: its pixel size, and the region it covers (`translation`).
Its shape is what you pass to the resampling function, not to the placeholder `Scale`:

```python
# For example, a relation that deskews an oblique acquisition
affine = multiscale.relation_to_coordinate_system("deskewed")
output_grid = Scale(shape=Shape.all_singletons("zyx"), pixel_size=input_scale.pixel_size, translation=output_origin)
matrix, offset = affine.inverted().to_linear_and_offset(source=output_grid, target=input_scale)
deskewed = scipy.ndimage.affine_transform(data, matrix, offset=offset, output_shape=output_shape)
```

The result is plain nested tuples in the order of `affine.axes`; your array must have its axes in the same order.
`Affine.from_linear_and_offset(*affine.to_linear_and_offset(source=a, target=b), affine.axes, source=a, target=b)` gives the same `Affine` back.
Without `source` and `target`, `to_linear_and_offset` returns the physical values as they are.

#### Matrices from other sources

Without `source` and `target`, `from_array` and `from_linear_and_offset` take the numbers as they are.
If your matrix does not follow the "push" or "forward" physical coordinate mapping convention, convert it first:

* **Direction**: If it maps output coordinates to input coordinates as resampling matrices often do, supply output as source, input as target, and invert it (`.inverted()` after constructing).
* **Row vectors**: `Affine` uses column vectors (`x_target = linear @ x_source`). For a matrix written for row vectors (`x_source @ matrix`), transpose it.
* **Index vs physical coordinates**: If it works on pixel indices, pass `source` and `target`, as above.
* **Axis order**: Rows and columns follow the `axes` you pass. Reorder if your matrix uses another order (for example `xyz` for data that is `zyx`).
* **Pixel reference point**: `Affine` treats index `i` as the pixel center. If your tool places integer indices at pixel corners, the offset needs a half-pixel correction: `offset_center = offset_corner + 0.5 * (matrix @ ones - ones)`.

### `Translation`

`Translation(t)` maps each source coordinate to `source + t` in the target system.
For example, in the context of a crop reaching its source as a coordinate system, t is the crop start in the source image.

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
).as_derived_from(source, reached_by=crop_offset * s0.pixel_size)  # PixelOffset * PixelSize = Translation
```

The cropped Multiscale's own origin is the crop corner, and the relation says where that corner lies in the source:
With a pixel size of 2.0/0.5/0.5 micrometer, `reached_by` is `Translation(z=20.0, y=50.0, x=100.0)`.

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

All relations, whether they act on values (`Affine`, `Translation`) or on axes, can be chained in one list:

```python
multiscale.with_coordinate_system(
    "yx_preview",
    reached_by=[ProjectionTo("yx"), Affine.from_factor({"y": 0.001, "x": 0.001})],
)
```

Each relation sees the axes that the previous one produced. Here, `ProjectionTo` first reduces `zyx` to `yx`, after which the `Affine` only needs to talk about `y` and `x`.

## Beyond a single Multiscale

Everything above describes the surroundings of *one* Multiscale.
To relate several Multiscales to each other (stitching tiles into a mosaic, registering images to a common reference, recording provenance of derived data), see [Scenes](scene.md).
It uses the same concepts.
