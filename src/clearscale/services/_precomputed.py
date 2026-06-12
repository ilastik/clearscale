from typing import Dict, Literal, Union, List

SCALES_DICT = Dict[
    Literal["key", "size", "resolution", "voxel_offset"],
    Union[str, List[int], List[float]],
]
INFO_DICT = Dict[
    Literal["scales", "num_channels"],
    Union[List[SCALES_DICT], int],
]


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
