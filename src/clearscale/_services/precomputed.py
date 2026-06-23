from typing import Dict, List, Literal, Sequence, Tuple, Union

from clearscale._axis_values import AxisKey, PixelSize

SCALES_DICT = Dict[
    Literal["key", "size", "resolution", "voxel_offset"],
    Union[str, List[int], List[float]],
]
INFO_DICT = Dict[
    Literal["scales", "num_channels"],
    Union[List[SCALES_DICT], int],
]


def zero_resolution_axes(resolution: List[float], axes: Sequence[AxisKey]) -> Tuple[AxisKey, ...]:
    return tuple(axis for axis, value in zip(axes, resolution) if value == 0)


def pixel_size_from_resolution(resolution: List[float], axes: Sequence[AxisKey]) -> PixelSize:
    normalized_resolution = [PixelSize._default if value == 0 else value for value in resolution]
    return PixelSize(zip(axes, [1.0] + list(reversed(normalized_resolution))))


def validate_info_dict(info_dict: INFO_DICT) -> None:
    if "scales" not in info_dict:
        raise ValueError("Precomputed info JSON must contain 'scales' field")

    scales_list = info_dict["scales"]
    if not isinstance(scales_list, list) or not scales_list:
        raise ValueError("Precomputed info JSON 'scales' must be a non-empty list")

    required_keys = ("key", "size", "resolution")

    for s in scales_list:
        if any(k not in s for k in required_keys):
            raise ValueError("Precomputed info JSON has invalid scale metadata (missing key, size or resolution).")
