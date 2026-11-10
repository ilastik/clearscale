# clearscale

clearscale is a small, dependency-free Python package for describing multiscale images and keeping their metadata consistent as you process them.
You read, write, and transform the image arrays with your preferred tools; clearscale handles the metadata, including reading and writing OME-Zarr.

## Install

clearscale supports Python 3.10 and newer and has no runtime dependencies.
Install the 0.9.0 release candidate from PyPI:

```bash
python -m pip install clearscale==0.9.0rc1
```

## Get started

Start with [OME-Zarr](ome_zarr.md) to add metadata support to an existing image-processing tool or script that uses zarr.

- [Basics](basics.md) for the core types: axis values, scaling blueprints, scales, and multiscales.
- [Scaling methods](characterization.md): Choose the right rounding, pixel size, and translation behavior for your scaling method.
- [Coordinate systems](space.md): Describe spatial context and relations between coordinate systems.
- [Scenes](scene.md): Relate several multiscales, for example after registration or stitching.
- [Examples](examples.md): See practical multiscale metadata workflows.

Or refer to the full automatically generated API Reference to look up individual classes and methods.
