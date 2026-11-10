from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import json
import sys
import clearscale as cs

expected_version = sys.argv[1]
assert version("clearscale") == expected_version, "Distribution version does not match the release tag"
assert cs.__version__ == expected_version, "Package version does not match the release tag"
assert (
    Path(cs.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
), "Package was not imported from the virtual environment"
for module in ("characterization", "ome_zarr", "types"):
    import_module(f"clearscale.{module}")
scale = cs.Scale.from_lists("zyx", shape=(2, 3, 4))
group = cs.OmeZarrGroup.from_single(cs.Multiscale.from_single(scale))
attrs = json.loads(json.dumps(group.to_attrs(version="0.6")))
restored = cs.OmeZarrGroup.from_attrs(attrs, shape_source={"s0": (2, 3, 4)})
assert restored.to_attrs(version="0.6") == attrs, "OME-Zarr metadata did not round-trip"
print(f"Wheel smoke test passed for clearscale {expected_version}")
