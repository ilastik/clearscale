# Scenes

> [!NOTE]
> Scenes are only supported by OME-Zarr 0.6.

A `Multiscale` knows the spaces around *itself*: its [coordinate systems](space.md#what-is-a-coordinate-system).
A `Scene` records how *several Multiscales* relate to each other: tiles that are placed into a mosaic, an image that was registered to an atlas, the outputs of the steps of a processing workflow.

If the spaces you want to relate all belong to a single Multiscale, you don't need a Scene.
Use `Multiscale.with_coordinate_system` instead.

## The idea

A Scene is a graph.
Its nodes are *spaces*: the space of a Multiscale's data (or one of its other coordinate systems), and coordinate systems that belong to the Scene itself.
Its edges are *spatial relations*, exactly the ones described in [Space](space.md#available-spatial-relations).
See there for more details of how to think of relations and the specific types available.

Similar to a Multiscale's scale keys, the Scene also records paths for each Multiscale it relates.
Scene paths are relative to the zarr group whose attrs store the Scene's metadata, just as the Multiscale's scale keys are relative to the zarr group whose attrs store the Multiscale's metadata.

Every Scene method reads relations the same way:

> `a --Relation--> b`: `Relation` maps coordinates from space `a` to space `b`.

`Translation(p)` adds p to a coordinate, and `ProjectionTo("yx")` drops other axes.
If you think of image operations as resampling new pixels out of an existing image, this is the opposite direction.

## Making a Scene

| You have                                         | Use                                                                     |
|--------------------------------------------------|-------------------------------------------------------------------------|
| Two Multiscales, one registered to the other     | [`from_multiscale_pair`](#a-pair-registration-or-provenance)            |
| Many Multiscales that sit in one common space    | [`from_shared_coordinate_system`](#shared-space-tiles)                  |
| A workflow: an original, and a processing output | [`from_multiscale_pair`](#a-pair-registration-or-provenance)            |
| Several results derived from the same original   | [`from_shared_coordinate_system`](#several-derivations-of-one-original) |

In all of them, a `source` or `a` can be a Multiscale (meaning the space of its own data), or `(multiscale, "name")` to select one of the other coordinate systems it knows.
Every Multiscale also needs a unique storage path relative to the group that stores the Scene.
Applying a relation to the axes of the space it starts from must produce the axes of the space it ends in, otherwise the constructor raises.

Scenes relate the specific Multiscale *instances* you pass.
This means if you later make modified variants of the Multiscales you passed to a Scene (e.g. adding another coordinate system through `.with_coordinate_system`), or even exact copies, a Scene you made with the old instances knows nothing of the new instances.
You would then need to make a new Scene with the new Multiscale instances.

### A pair: registration or provenance

```python
import clearscale as cs

scene = cs.Scene.from_multiscale_pair(
    a=raw_ms,
    b=atlas_ms,
    a_to_b=cs.Translation(y=10.0, x=20.0),
    a_path="raw",
    b_path="atlas",
)
```

`a_to_b` maps raw coordinates to atlas coordinates: here, add `(10, 20)`.
It accepts a single relation or a list (applied first to last), or `cs.SameSpace()` if both are the same space.

`a_path` and `b_path` are required: The Multiscales' paths within the Scene.

### Shared space: tiles

```python
import clearscale as cs

positions = [
    (tile1, cs.PixelOffset(y=0, x=0), "tiles/tile1"),
    (tile2, cs.PixelOffset(y=0, x=1000), "tiles/tile2"),
    (tile3, cs.PixelOffset(y=1000, x=0), "tiles/tile3"),
]

scene = cs.Scene.from_shared_coordinate_system(
    coordinate_system="stage",
    connections=[
        cs.Scene.Connection(
            tile,
            to_shared=offset * tile["s0"].pixel_size,
            path=path,
        )
        for tile, offset, path in positions
    ],
)
```

`path` is required for every connection: The Multiscales' paths within the Scene.

This creates a new coordinate system, "stage", that all the tiles are connected to.
Each connection says how the stage is reached from that tile, the same way as for `tile.with_coordinate_system("stage", reached_by=...)`:
`tile --to_shared--> "stage"`.

A tile whose origin sits at p on the stage reaches stage coordinates through `Translation(p)`.

The axes of the shared system are what the relations produce, so all connections must produce the same axes.
Its axis properties (units, types) are taken from the connected Multiscales, as far as each relation allows: See [Axis properties across relations](space.md#axis-properties-across-relations).
For example, a connection through a scaling `Affine` doesn't contribute its unit to the stage, but a connection that keeps the axes does.
Connections may differ in properties that are blank, but contradicting values raise an error.
Set the properties explicitly with `unit=` or `ome_zarr_axes=` for the stage to have any that no connection provides.
Explicitly provided properties are still validated for consistency with the provided relation.

### Several derivations of one original

For example, different segmentations, or different crops of the same original: Each of them is derived from the original, not from each other.
Express them with `from_shared_coordinate_system`, using the original's space as the shared system.
The relation is the same one passed to `as_derived_from`: `result --R--> original`.

```python
import clearscale as cs

scene = cs.Scene.from_shared_coordinate_system(
    coordinate_system="original",
    connections=[
        cs.Scene.Connection(original, to_shared=cs.SameSpace(), path="original"),
        # cells = a full segmentation image
        cs.Scene.Connection(cells, to_shared=cs.SameSpace(), path="cells"),
        # nuclei = a segmentation image cropped from original, like:
        # nuclei.as_derived_from(original, reached_by=Translation(y=32.0, x=48.0))
        cs.Scene.Connection(nuclei, to_shared=cs.Translation(y=32.0, x=48.0), path="nuclei"),
        # thumbnail = a representative slice extracted from the center of the stack (e.g. at 160um)
        cs.Scene.Connection(
            thumbnail_yx,
            to_shared=[cs.ProjectionTo("zyx"), cs.Translation(z=160.0, y=0, x=0)],
            path="thumbnail",
        ),
    ],
)
```

## Retrieving relations

```python
relation = scene.relation_from(volume, to=cropped)
```

`scene.relation_from(source, to=target)` returns the relation `R` with `source --R--> target`, reading the same way as the relations you pass to the constructors.
The relations are returned in canonical form: relations that can be merged are merged.
For the chain above, `volume` to `cropped` is a `ProjectionTo`, then a `Translation`: When there are several relations, they come back as a sequence that you can iterate, index, and pass to anything that accepts relations.

* `source` and `to` can be a name of one of the Scene's coordinate systems, a Multiscale, or `(multiscale, "name")`.
* If both are the same space, the result is `cs.SameSpace()` (the identity-relation).
* If no relation expresses the connection, the result is `None`.
  This includes spaces that are not connected, and spaces reached by dropping axes, which can't be traced backwards: `relation_from(max_projection, to=volume)` is `None` while `relation_from(volume, to=max_projection)` works.
  It also includes connections with no spatial relation equivalent, e.g. a coordinate lookup or a displacement field that was read from a file.
* It raises `ValueError` if a space can't be found, or if it belongs to a Multiscale that is not resolved yet, because its axes are unknown (see below).

## Reading and writing

A Scene refers to its Multiscales by their path relative to the Scene's own zarr group.
When you read one, those Multiscales are placeholders until you provide them:

```python
import clearscale as cs

scene = cs.OmeZarrGroup.from_group(zarr_group).scenes[0]

assert not scene.is_fully_resolved
multiscales = {
    path: cs.OmeZarrGroup.from_group(zarr_group[path]).multiscales[0]
    for path in scene.unresolved_paths
}
scene = scene.with_resolved(multiscales)

assert scene.is_fully_resolved
```

`with_resolved` also verifies that each Multiscale you provide has the coordinate system the Scene expects at its path.

`scene.multiscales` lists the Multiscales a Scene knows, and `scene.multiscale_paths` is a read-only mapping like `{path: Multiscale}`.
For a Scene you made yourself, both contain the paths and Multiscales you passed in.
For a Scene you read, they contain the ones you have resolved so far.

### Writing a Scene

A Scene is stored like any other OME-Zarr object: In the attrs of a zarr group, which you get from an `OmeZarrGroup`.
Wrap the Scene with `OmeZarrGroup.from_single` and call `to_attrs`, exactly as you would for a Multiscale:

```python
import zarr  # assuming you use zarr-python 3.*
import clearscale as cs

raw_ms = (...)  # one Multiscale
atlas_ms = (...)  # another Multiscale

scene = cs.Scene.from_multiscale_pair(
  a=raw_ms, 
  b=atlas_ms, 
  a_to_b=cs.Translation(y=10.0, x=20.0), 
  a_path="raw", 
  b_path="atlas"
)

scene_group = zarr.open_group(
  "registration_scene.zarr", 
  mode="w", 
  zarr_format=3
)
scene_group.attrs.update(
  cs.OmeZarrGroup
  .from_single(scene)
  .to_attrs(version="0.6")
)
```

This writes *only* the Scene's own metadata: the relations, and the Multiscale paths they refer to.
The metadata of the Multiscales themselves, and their arrays, are zarr groups nested inside the scene group. 
You must create them at exactly those paths, relative to the Scene's group.
`scene.multiscale_paths` has what you need, so you can write each Multiscale the usual way, one group per path:

```python
for path, multiscale in scene.multiscale_paths.items():
    multiscale_group = scene_group.require_group(path)
    multiscale_group.attrs.update(
      cs.OmeZarrGroup
      .from_single(multiscale)
      .to_attrs(version="0.6")
    )
    for scale_key in multiscale:
        ...  # write the array for this scale at multiscale_group[scale_key]
```

For the pair above, the result looks like this:

```
registration_scene.zarr/
├── zarr.json         <- "ome": {"version": "0.6", "scene": {...}}
├── raw/
│   ├── zarr.json     <- "ome": {"multiscales": [...]} of raw_ms
│   └── s0/           <- array
└── atlas/
    ├── zarr.json     <- "ome": {"multiscales": [...]} of atlas_ms
    └── s0/           <- array
```

Keep in mind:

* Write the Scene and its Multiscales from the same objects (Multiscale instances).
  The Scene refers to each Multiscale's coordinate systems by name, and a Multiscale you create without naming its coordinate system gets a generated one.
  A Multiscale that you re-create later, even from identical parameters, has a different generated name.
  The Scene would then no longer find its space in the metadata you wrote, and `with_resolved` rejects it when reading.
* Use `scene.multiscale_paths` rather than writing the paths out a second time, so that the layout can't drift from the Scene's metadata.
* A Scene whose Multiscales are not all resolved (see above) can still be written with `to_attrs`, but it can only tell you where the missing Multiscales are supposed to be (`unresolved_paths`), not write them.
* Scenes need OME-Zarr 0.6: `to_attrs(version=...)` raises for earlier versions.
