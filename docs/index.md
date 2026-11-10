# clearscale documentation

clearscale is a small, dependency-free Python package for describing multiscale images and keeping their metadata consistent as you process them.
You read, write, and transform the image arrays with your preferred tools; clearscale handles the metadata, including reading and writing OME-Zarr.

## Install

clearscale supports Python 3.10 and newer and has no runtime dependencies.
Until the first PyPI release, install from GitHub:

```bash
pip install git+https://github.com/ilastik/clearscale.git
```

After the first release, install with `pip install clearscale`.

## Start here

Start with [OME-Zarr](ome_zarr.md) to add metadata support to an existing image-processing workflow.
Read [Basics](basics.md) for the core types: axis values, scaling blueprints, scales, and multiscales.

- [Scaling methods](characterization.md): Choose the right rounding, pixel size, and translation behavior for your scaling method.
- [Coordinate systems](space.md): Describe spatial context and relations between coordinate systems.
- [Scenes](scene.md): Relate several multiscales, for example after registration or stitching.
- [Examples](examples.md): See practical multiscale metadata workflows.

The API Reference provides generated signatures and docstrings for readers who want to look up individual classes and methods.
The guides above are the recommended starting point.
