# clearscale

[![CI](https://github.com/ilastik/clearscale/actions/workflows/ci.yaml/badge.svg?branch=main&event=push)](https://github.com/ilastik/clearscale/actions/workflows/ci.yaml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ilastik/clearscale/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ilastik/clearscale)
[![PyPI](https://img.shields.io/pypi/v/clearscale.svg)](https://pypi.org/project/clearscale/)
[![Python versions](https://img.shields.io/pypi/pyversions/clearscale.svg)](https://pypi.org/project/clearscale/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ilastik/clearscale/blob/main/LICENSE)

clearscale is a dependency-free Python library for reading, transforming, and writing multiscale image metadata.

It provides an abstraction layer between image-processing code and metadata formats such as OME-Zarr.
Work with named axes, shapes, pixel sizes, and spatial relations; clearscale handles the format-specific parsing and serialization.
Your application keeps control of array storage, image processing, and the user interface.

## What it provides

- **Axis-aware metadata:** Immutable, ordered mappings keep axis names attached to their shapes, pixel size, unit and translation values through reordering, insertions and drops.
- **Multiscale planning:** Shape and factor blueprints describe resolution levels and can drive your processing code. Scaling method characterization helps you calculate the corresponding pixel sizes and translations.
- **OME-Zarr support:** Read versions 0.1–0.6 through a common API and write versions 0.4, 0.5, and 0.6, with validation during construction and serialization. See [feature coverage](https://clearscale.readthedocs.io/en/latest/ome_zarr/#ome-zarr-features).
- **Spatial relationships:** Describe alternate coordinate systems, affine transformations, and relations between images, including OME-Zarr 0.6 scenes.
- **A small integration footprint:** Pure Python, no runtime dependencies, and typed APIs designed to work with ordinary Python mappings and your choice of array and storage libraries.

Basic reading support for Neuroglancer Precomputed metadata is also included.

## Installation

Requires Python 3.10 or newer.

```bash
pip install clearscale==0.9.0rc3
```

Conda packages are available from the maintainer's `btbest` channel:

```bash
conda install -c btbest clearscale==0.9.0rc3
```

The 0.9 series is in beta; the public API may change before 1.0.

## Metadata-only quickstart

Suppose your application has written an image array with axes `zyx` at path `s0` within a Zarr group.
Here is how to describe its shape and physical pixel sizes, and prepare OME-Zarr metadata:

```python
import clearscale as cs

shape = cs.Shape(z=16, y=64, x=64)
pixel_size = cs.PixelSize(z=0.5, y=0.25, x=0.25)
unit = cs.Unit(z="micrometer", y="micrometer", x="micrometer")

# A Scale describes one array
scale = cs.Scale(shape=shape, pixel_size=pixel_size, unit=unit)

# A Multiscale describes the image the Scale belongs to
multiscale = cs.Multiscale.from_single(scale, scale_key="s0")

# An OmeZarrGroup prepares the metadata for a Zarr group's attributes
ome_attrs = cs.OmeZarrGroup.from_single(multiscale).to_attrs(version="0.5")
```

Attach `ome_attrs` to the group using your Zarr library.
Typically:
`zarr_group.attrs.update(ome_attrs)`.

The axis order, shape, pixel sizes, and array path should describe the data you actually wrote.
For OME-Zarr 0.5 and 0.6, the underlying store must use Zarr format 3.

This example uses only clearscale.
For complete examples that also create, scale and store arrays, see [documentation](https://clearscale.readthedocs.io/en/latest/#minimal-ome-zarr-example).

## Keeping data and metadata together

For an image pyramid, `BlueprintShapes` and `BlueprintFactors` describe the target shapes or scaling factors for each resolution level.
You can use a blueprint to drive array processing, then apply it to a base `Scale` to describe the resulting multiscale image using `cs.Multiscale.from_single(scale, blueprint=...)`.

If your arrays already exist, and you know how the image was scaled, you can also build metadata from their recorded shapes.
Array shapes alone cannot tell you where the output samples lie.
Hence, clearscale asks for the scaling method's rounding, pixel-sizing, and translation behavior.
The [characterization tools](https://clearscale.readthedocs.io/en/latest/characterization/) help determine these properties for your actual processing function.

## Where clearscale fits

The OME-Zarr ecosystem benefits from the work of projects such as
[bioio](https://github.com/bioio-devs/bioio),
[ngff-zarr](https://github.com/thewtex/ngff-zarr),
[ngio](https://biovisioncenter.github.io/ngio/stable/),
[ome-zarr-py](https://github.com/ome/ome-zarr-py), and
[yaozarrs](https://github.com/tlambert03/yaozarrs).
The [NGFF tools directory](https://ngff.openmicroscopy.org/resources/tools/index.html) is a useful overview of the wider ecosystem.

clearscale focuses on the metadata layer for applications, libraries, and scripts that already have their own storage and processing choices.
This narrow scope keeps it easy to integrate into an existing environment.
It works only with metadata, leaving data access, resampling, chunking and compression to the tools you choose for those jobs.

## Documentation

The [documentation](https://clearscale.readthedocs.io/) includes practical guides and the full API reference:

- [OME-Zarr integration](https://clearscale.readthedocs.io/en/latest/ome_zarr/): read, inspect, and write metadata across versions.
- [Core concepts](https://clearscale.readthedocs.io/en/latest/basics/): axis values, scales, multiscales, and blueprints.
- [Scaling methods](https://clearscale.readthedocs.io/en/latest/characterization/): match metadata to the behavior of your processing code.
- [Coordinate systems](https://clearscale.readthedocs.io/en/latest/space/) and [scenes](https://clearscale.readthedocs.io/en/latest/scene/): describe spatial context and relationships between images.

## Contributing

Questions, bug reports, and contributions are welcome.
If something is awkward to express in your application, please let us know!

See [CONTRIBUTING.md](https://github.com/ilastik/clearscale/blob/main/CONTRIBUTING.md) for development instructions and project scope, or [open an issue](https://github.com/ilastik/clearscale/issues).

## License

Licensed under either the [MIT license](https://github.com/ilastik/clearscale/blob/main/LICENSE-MIT) or the
[Apache License, Version 2.0](https://github.com/ilastik/clearscale/blob/main/LICENSE-APACHE), at your option.
