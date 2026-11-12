# clearscale

[![CI](https://github.com/ilastik/clearscale/actions/workflows/ci.yaml/badge.svg?branch=main&event=push)](https://github.com/ilastik/clearscale/actions/workflows/ci.yaml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ilastik/clearscale/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ilastik/clearscale)
[![PyPI](https://img.shields.io/pypi/v/clearscale.svg)](https://pypi.org/project/clearscale/)
[![Python versions](https://img.shields.io/pypi/pyversions/clearscale.svg)](https://pypi.org/project/clearscale/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

`clearscale` is a small dependency-free Python package for clear multiscale image metadata manipulation.

`clearscale` acts as an abstraction layer.
It handles metadata parsing, processing and serialization, and allows your code to focus on filesystem handling, data processing, and UX.

## Features

* Zero dependencies, pure Python
* Compatible with Python 3.10+
* Read all OME-Zarr versions (0.1 to 0.6) using one code path - no specific handling per version (unless you want to)
* Write OME-Zarr versions 0.4, 0.5 and 0.6
* Virtually no knowledge about OME-Zarr required
* Read Neuroglancer Precomputed metadata (very basic support)

### Highlights

* Ensure accurate multiscale metadata by [characterizing](docs/characterization.md) your scaling method
* Read and write multiscale transformations into alternate [coordinate systems](docs/space.md) (OME-Zarr 0.6)
* Read and write transformations between multiscales using [scenes](docs/scene.md) (OME-Zarr 0.6)

## Minimal OME-Zarr example

This writes a single numpy array as OME-Zarr.
The absolute minimum requirements:

* axis keys (`"zyx"`)
* writing the data to a zarr *array* inside a zarr *group* (and recording the array sub-path)

```python
# Assuming you have numpy and zarr-python 3.* installed
import numpy as np
import zarr
import clearscale as cs

# You have some array
image = np.random.random((128, 1024, 1024)).astype(np.float32)

# 1. Data: Make a group and write the array into it
# (no clearscale involved)
group = zarr.open_group(
    "demo-output/minimal.ome.zarr", 
    mode="w", 
)
array_path = "s0"
group.create_array(array_path, data=image)

# 2. Metadata: Scale -> Multiscale -> Group -> attrs
group.attrs.update(
    cs.OmeZarrGroup  # Group = one zarr group's OME metadata
    .from_single(
        cs.Multiscale.from_single(  # Multiscale = an image
            scale=cs.Scale.from_lists("zyx"),  # Scale = an array
            scale_key=array_path
        )
    )
    .to_attrs(version="0.5")
)
```

## Install

`clearscale` is considered in "beta" status (0.9) for a short time, with 1.0 release intended soon(TM), when we have some feedback from first users.

Please try it and let us know if you encounter anything unclear!

Install from PyPI:

```bash
pip install clearscale==0.9.0rc3
```

Or using conda:

```yaml
conda install -c btbest clearscale==0.9.0rc3
```

Note that until we publish 1.0, the API is considered subject to change so we can ensure it "solidifies" with as much convenience and robustness as possible.
We hope there won't be too many changes necessary at this point.

## Documentation

Guides (OME-Zarr how-to, core concepts, scaling metadata, ...) and automatically generated API Reference are in the [clearscale documentation](https://clearscale.readthedocs.io/).

The guides are also directly under [`docs/`](docs/index.md) in the repository (on GitHub).
[`docs/ome_zarr.md`](docs/ome_zarr.md) also contains an overview of clearscale's OME-Zarr feature coverage.

## Why clearscale?

There are already fully-featured solutions for reading and writing OME-Zarr.
clearscale does not replace them, but rather complements them.

Check out the [NGFF Tools](https://ngff.openmicroscopy.org/resources/tools/index.html) page for some examples.
Python libraries include e.g.
[bioio](https://github.com/bioio-devs/bioio),
[ngff-zarr](https://github.com/thewtex/ngff-zarr),
[ngio](https://biovisioncenter.github.io/ngio/stable/),
[ome-zarr-py](https://github.com/ome/ome-zarr-py), and
[yaozarrs](https://github.com/tlambert03/yaozarrs).
Some of these handle not only the metadata, but also data processing all in one.

clearscale aims to provide a solution that works for *any* tool or script, no matter the
[choice](https://github.com/zarr-developers/zarr-python)
[of](https://github.com/canpute/simplezarr)
[zarr](https://github.com/google/tensorstore)
[backend](https://github.com/constantinpape/z5),
other packages in the Python environment, legal requirements (e.g. commercial use), or peculiarities of custom workflows.

### Zero dependencies

Being dependency-free means:

* `clearscale` fits in any Python environment. Installing or uninstalling it will not modify any other installed packages.
* `clearscale` does *not* interact with data (numpy/zarr arrays). You implement data processing according to your needs, and use `clearscale` to generate metadata that describes your processing.

This provides full flexibility for data processing, but it comes at the risk of desynchronization between data and metadata.

You can mitigate this by coupling data processing to the metadata:
We generate the metadata first, and then scale the data according to that.
See the example below.
If your arrays already exist, see [Add metadata to existing arrays](docs/examples.md#add-metadata-to-existing-arrays).
Describing data after the fact only works if you know how your scaling method behaves, which you can [characterize](docs/characterization.md).

## Copy-and-paste examples

These examples assume you have `numpy`, `scikit-image` and `zarr-python 3.*` (and of course `clearscale`) installed.

### Downsample a numpy array and save as OME-Zarr multiscale

```python
import os, numpy, skimage, zarr

# Placeholders - plug in any zarr backend and image scaling method
my_data = numpy.random.rand(3, 12, 512, 512)
do_my_scaling = skimage.transform.resize
zarr_group = zarr.open_group(os.path.expanduser("demo-output/plan_first.ome.zarr"), mode="w")

import clearscale as cs

# 1. Define operation plan (the scaling "blueprint")
shape = cs.Shape(zip("tzyx", my_data.shape))
scaling_blueprint = cs.BlueprintShapes.downscale_powers_of_2_xyz(
    base_shape=shape,
    rounding="ceil",
    limit_all=cs.Shape(z=8, y=128, x=128),
)

# 2. Scale data according to the metadata
for scale_key, target_shape in scaling_blueprint.items():
    # Target shape from blueprint
    scaled_data = do_my_scaling(my_data, target_shape.values_tuple())
    # Target array path from blueprint
    zarr_group.create_array(scale_key, data=scaled_data)

# 3. Write metadata: Scale -> Multiscale -> Group -> attrs
pixel_size = cs.PixelSize(t=5.0, z=260.0, y=0.53, x=0.53)
unit       = cs.Unit(t="s", z="micrometer", y="micrometer", x="micrometer")
base       = cs.Scale(shape, pixel_size, unit)

ms = cs.Multiscale.from_single(
    base,
    blueprint=scaling_blueprint,
    pixel_sizing="shape_ratio",  # skimage.transform.resize scales pixel size by the shape ratio...
    translating="half_pixel_space_preservation",  # ...and preserves the image's extent
)
group_meta = cs.OmeZarrGroup.from_single(ms).to_attrs(version="0.6")
zarr_group.attrs.update(group_meta)
```

### Download a single scale of a public OME-Zarr as a valid local OME-Zarr

```python
from pathlib import Path
import zarr  # This example assumes zarr-python 3.* is installed (plus fsspec, requests and aiohttp for https reading)
import clearscale as cs

URL = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
SCALE_KEY = "s6"
LOCAL_PATH = Path(f"demo-output/platy-raw-{SCALE_KEY}.ome.zarr")

# 1. Discover what's on the server
remote_group = zarr.open_group(URL)

remote_meta = cs.OmeZarrGroup.from_group(remote_group)
assert remote_meta.kind is cs.GroupKind.MULTISCALE, "This tells us there is exactly one Multiscale in this group"
assert remote_meta.version == "0.4", "OME-Zarr version is a property of the group, not of individual Multiscales"

remote_multiscale = remote_meta.multiscales[0]
assert SCALE_KEY in remote_multiscale, "Multiscale is dict-like, with the sub-paths to arrays as keys"
assert remote_multiscale[SCALE_KEY].shape == {"z": 357, "y": 405, "x": 430}

# 2. Create the local target and download the data
source_array = remote_group[SCALE_KEY]
local_group = zarr.open_group(str(LOCAL_PATH), mode="w", zarr_format=2)  # OME-Zarr v0.4 must use zarr-format v2
print(f"Downloading {SCALE_KEY} data...")
local_array = local_group.create_array(SCALE_KEY, data=source_array)

# 3. Derive a new Multiscale from the chosen Scale entry in the existing Multiscale
local_multiscale = remote_multiscale.derive(SCALE_KEY)

# 4. Write the new metadata to the downloaded store
local_group_meta = cs.OmeZarrGroup.from_single(local_multiscale)
local_group.attrs.update(local_group_meta.to_attrs(version="0.4"))
```

## Contributing / Feature scope
Any form of feedback, issues, bug reports, feature requests, pull requests and other contributions are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) on GitHub.

## License

Licensed under either the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.
