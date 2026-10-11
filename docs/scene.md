# Scenes

A [`Multiscale`](space.md) knows the spaces around *itself*: its own coordinate systems, and how it was derived.
A `Scene` records how *several Multiscales* relate to each other: tiles that are placed into a mosaic, an image that was registered to an atlas, the outputs of the steps of a processing workflow.

If the spaces you want to relate all belong to a single Multiscale, you don't need a Scene.
Use `Multiscale.with_coordinate_system` instead.

> [!NOTE]
> Scenes can only be written as OME-Zarr 0.6, and the Scene API is still young.
> Please let us know if you need it for your work, or if something is awkward.

## The idea

A Scene is a graph.
Its nodes are *spaces*: the space of a Multiscale's data (or one of its other coordinate systems), and coordinate systems that belong to the Scene itself.
Its edges are *spatial relations*, exactly the ones described in [Space](space.md#available-spatial-relations).

Every Scene method reads relations the same way:

> `a --R--> b`: `R` is what you do with an image in space `a` to obtain space `b`.

`Translation(p)` means "cropped, so that the new origin is at `p`", and `ProjectionTo("yx")` means "dropped the other axes".
This is the opposite of how you might think about coordinates, see [Space](space.md#spatial-relation-vs-coordinate-transformation) for the reasoning.

There is no simple relation for "scaled by": The physical scale of an image is its pixel size, so two spaces with the same units
are not related by scaling. If spaces have different units, relate them with an `Affine`, e.g. `Affine.from_unit_conversion`.

## Making a Scene

| You have                                                | Use                                                             |
|---------------------------------------------------------|-----------------------------------------------------------------|
| Two Multiscales, one registered to the other            | [`from_multiscale_pair`](#a-pair-registration)                  |
| Many Multiscales that sit in one common space           | [`from_shared_coordinate_system`](#shared-space-tiles)          |
| A workflow: an original, processed, and processed again | [`from_multiscale_chain`](#a-chain-workflow-steps)              |
| Several results derived from the same original          | [`from_shared_coordinate_system`](#several-derivations-of-one-original) |

In all of them, `a` or a `source` can be a Multiscale (meaning the space of its own data), or `(multiscale, "name")` to select one of the other coordinate systems it knows.
Applying a relation to the axes of the space it starts from must produce the axes of the space it ends in, otherwise the constructor raises.

### A pair: registration

```python
from clearscale import Scene, Translation

scene = Scene.from_multiscale_pair(
    a=raw,
    b=atlas,
    a_to_b=Translation(y=10.0, x=20.0),
)
```

`a_to_b` says what you do with `raw` to obtain `atlas`: here, crop it with its origin at `(10, 20)`.
It accepts a single relation or a list (applied first to last), or `None` if both are the same space.

### Shared space: tiles

```python
from clearscale import Connection, PixelOffset, Scene

positions = [
    (tile1, PixelOffset(y=0, x=0)),
    (tile2, PixelOffset(y=0, x=1000)),
    (tile3, PixelOffset(y=1000, x=0)),
]

scene = Scene.from_shared_coordinate_system(
    coordinate_system="stage",
    connections=[
        Connection(tile, to_shared=(offset * tile["s0"].pixel_size).inverted())
        for tile, offset in positions
    ],
)
```

This creates a new coordinate system, "stage", that all the tiles are connected to.
Each connection says how the stage is reached from that tile, the same way as for `tile.with_coordinate_system("stage", reached_by=...)`:
`tile --to_shared--> "stage"`.

Because `Translation(p)` places the *origin* of the result at `p`, a tile that sits at position `p` on the stage reaches the stage through `Translation(p).inverted()`.

The axes of the shared system are what the relations produce, so all connections must produce the same axes.
Its axis properties (units, types) are taken from the connected Multiscales.
They may differ in properties that are blank, but contradicting values raise an error, unless you set them explicitly with `unit=` or `ome_zarr_axes=`.

> [!TIP]
> Pass `connections` as a list, not as a dict with Multiscales as keys.
> Multiscales that look alike compare equal, so tiles in a regular grid would collapse into a single key.

### A chain: workflow steps

```python
from clearscale import ChainStep, ProjectionTo, Scene, Translation

scene = Scene.from_multiscale_chain(
    start=volume,
    steps=[
        ChainStep(ProjectionTo("yx"), projection),
        ChainStep(Translation(y=10.0, x=20.0), cropped, coordinate_system="detail"),
    ],
)
```

Each step says what was done to the *previous* Multiscale to obtain its result, like `result.as_derived_from(previous, by=...)`.
Written as a diagram, the Scene above is:

```
volume --identity--> "start"
volume --ProjectionTo--> projection --Translation--> cropped --identity--> "detail"
```

The coordinate systems are optional entry points for readers that offer to look at a Scene "in" a particular space.
`"start"` is the space of `volume`, and can be renamed or left out with `start_coordinate_system=`.
`ChainStep.coordinate_system` adds one for the space of any other step.

Which spaces make good entry points depends on the relations.
Data that comes after a step that dropped axes (e.g. a projection) can't be shown in the space before it, but anything before the step can be shown in the space after it.

Each Multiscale can only appear once in a chain.

### Several derivations of one original

Different segmentations, or different crops of the same original, are not a chain: Each of them is derived from the original, not from each other.
Express them with `from_shared_coordinate_system`, with the original's space as the shared system, and **each relation inverted compared to how you would pass it to `as_derived_from`**.

`result.as_derived_from(original, by=R)` says `original --R--> result`.
A shared system needs the other direction, `result --R⁻¹--> original`:

```python
from clearscale import Connection, Scene, Translation

scene = Scene.from_shared_coordinate_system(
    coordinate_system="original",
    connections=[
        Connection(original, to_shared=None),
        # nuclei = original.as_derived_from(..., by=Translation(y=32.0, x=48.0))
        Connection(nuclei, to_shared=Translation(y=32.0, x=48.0).inverted()),
        # cells = cropped from the original with its origin at (64, 128)
        Connection(cells, to_shared=Translation(y=64.0, x=128.0).inverted()),
    ],
)
```

`Translation` has `.inverted()`, and so does `Affine`.
For relations that act on axes, name the axes the other way round, e.g. `PermutationTo(original_axes)`.
Derivations that *drop* axes (like projections) are a poor fit: Inverting them would insert the axes again at coordinate 0, so the Scene would claim the result sits at that slice of the original.
Use `from_multiscale_pair` or `from_multiscale_chain` for those.

You can check the result with `scene.relation_from("original", to=nuclei)` (see below), which returns the relation you would have given to `as_derived_from`.

## Retrieving relations

```python
relation = scene.relation_from(volume, to=cropped)
```

`scene.relation_from(source, to=target)` returns the relation `R` with `source --R--> target`, read the same way as the relations you pass to the constructors.
The relations are returned in canonical form: relations that can be merged are merged.
For the chain above, `volume` to `cropped` is a `ProjectionTo`, then a `Translation`: When there are several relations, they come back as a sequence that you can iterate, index, and pass to anything that accepts relations.

* `source` and `to` can be a name of one of the Scene's coordinate systems, a Multiscale, or `(multiscale, "name")`.
* If both are the same space, the result is `None`.
* If there is no way to get from one to the other, it raises `ValueError`.
  This includes spaces reached by dropping axes, which can't be traced backwards: `relation_from(projection, to=volume)` fails while `relation_from(volume, to=projection)` works.
  It also raises if a space belongs to a Multiscale that is not resolved yet, because its axes are unknown (see below).
* If the connection isn't a spatial relation at all, e.g. a coordinate lookup or a displacement field that was read from a file, it raises `CannotConvertToSpatialRelationError`.
* `include_children=True` (the default) also follows the connections inside the Multiscales' own coordinate systems.
  With `False`, only the connections of the Scene itself are followed.

## Reading and writing

A Scene refers to its Multiscales by their path relative to the Scene's own zarr group.
When you read one, those Multiscales are placeholders until you provide them:

```python
from clearscale import OmeZarrGroup

scene = OmeZarrGroup.from_group(zarr_group).scenes[0]

assert not scene.is_fully_resolved
multiscales = {
    path: OmeZarrGroup.from_group(zarr_group[path]).multiscales[0]
    for path in scene.unresolved_paths
}
scene = scene.with_resolved(multiscales)

assert scene.is_fully_resolved
```

`with_resolved` also verifies that each Multiscale you provide has the coordinate system the Scene expects at its path.

Writing needs to know the same paths, because Multiscales don't know where they are stored:

```python
paths = {"raw": raw, "atlas": atlas}  # Each Multiscale's path within the Scene's group
scene_metadata = scene.to_ome_zarr(version="0.6", multiscales_by_path=paths)

zarr_group.attrs.update({"ome": {"version": "0.6", "scene": scene_metadata}})
```

Scenes are only supported by OME-Zarr 0.6.
