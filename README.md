# clearscale

`clearscale` is a small pure-Python package for clear multi-scale image metadata manipulation.

Fits in any Python environment. Works with existing code. Handles all OME-Zarr versions.

```python
from clearscale import Shape, PixelSize, Unit, Scale, BlueprintShapes, Multiscale

# 1. Annotate
shape      = Shape(zip("tzyx", mydata.shape))  # mydata: your image numpy/zarr
pixel_size = PixelSize(t=5.0, z=260.0, y=0.53, x=0.53)
unit       = Unit(t="s", z="micron", y="micron", x="micron")

# 2. Define scaling paradigm
scaling_blueprint = BlueprintShapes.downscale_powers_of_2_xyz(
    base_shape=shape,
    rounding="ceil",
    shape_limit=Shape(z=8, y=128, x=128),
)

# 3. Expand metadata
base = Scale(shape, pixel_size, unit)
ms = Multiscale.from_shapes(scaling_blueprint, base=base)

# 4. Export
zarr_group.attrs["ome"]["multiscales"] = [ms.to_ome_zarr(version="0.6.dev3")]
```

## Features

* Zero dependencies, backwards compatible to py3.10+
* Reads Neuroglancer Precomputed and OME-Zarr metadata (all versions)
* Writes OME-Zarr versions 0.4, 0.5 and 0.6.dev3
* Saves you learning about the metadata format(s)
* Helps you write expressive code
* Metadata manipulation lives alongside data manipulation
* Blueprints are as flexible as your image processing needs

## Install

PyPI and conda-forge tbd... for now:

```bash
git clone https://github.com/btbest/clearscale.git
pip install -e clearscale
```

## Examples

### Downsample a numpy array and save it as OME-Zarr

```python
# This example assumes numpy, scikit-image and zarr-python 3.* are installed
import numpy as np
import zarr
from skimage.transform import pyramid_gaussian

from clearscale import BlueprintShapes, PixelSize, Scale, Shape, Unit

# 1. Generate a pyramid
image = np.random.random((128, 1024, 1024)).astype(np.float32)
pyramid = [level.astype(np.float32) for level in pyramid_gaussian(image, downscale=2)]

# 2. Write arrays to zarr on disk and record scaled shapes
group = zarr.open_group("example.ome.zarr", mode="w")
scaled_shapes = []
for i, level in enumerate(pyramid):
    scale_key = f"s{i}"
    group.create_array(scale_key, data=level)
    scaled_shapes.append(
        (
            scale_key,
            Shape(zip("zyx", level.shape))
        )
    )

# 3. Describe the full-resolution image
base = Scale(
    shape=Shape(zip("zyx", image.shape)),
    pixel_size=PixelSize(z=25, y=240, x=240),
    unit=Unit(z="micron", y="nanometer", x="nanometer"),
)

# 4. Use the recorded scale shapes as a blueprint to expand a Multiscale
blueprint = BlueprintShapes(scaled_shapes)
multiscale = blueprint.apply_to_scale(base)

# 5. Save OME-Zarr metadata
group.attrs["multiscales"] = [
    multiscale.to_ome_zarr(version="0.5", axis_types="infer")
]
```

### Download a single scale of a public OME-Zarr as a valid local OME-Zarr

```python
from pathlib import Path
import zarr  # This example assumes zarr-python 3.* is installed
from clearscale import Multiscale


URL = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
SCALE_KEY = "s6"
LOCAL_PATH = Path(f"demo-output/platy-raw-{SCALE_KEY}.ome.zarr")

# 1. Extract the raw metadata
remote_group = zarr.open_group(URL)
ome_multiscale = remote_group.attrs["multiscales"][0]

# 2. Create the local target and download the data
source_array = remote_group[SCALE_KEY]
local_group = zarr.open_group(str(LOCAL_PATH), mode="w", zarr_version=2)
print(f"Downloading {SCALE_KEY} data...")
local_array = local_group.create_array(SCALE_KEY, data=source_array, overwrite=True)

# 3. Extract the correct scale metadata and upgrade it to valid independent metadata
source_multiscale = Multiscale.from_ome_zarr(ome_multiscale, shape_source=remote_group)
extracted_scale = source_multiscale[SCALE_KEY]
target_multiscale = Multiscale({SCALE_KEY: extracted_scale})

# 4. Write the new metadata to the downloaded store
local_group.attrs["multiscales"] = [target_multiscale.to_ome_zarr(version="0.4")]
```

## License

Licensed under either the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.