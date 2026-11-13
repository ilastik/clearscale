"""Run the end-to-end examples of docs/examples.md, so that they keep working.

The examples need more than clearscale (numpy, scipy, zarr, matplotlib), so these tests are skipped without them.
The code is taken from the docs page itself, not mirrored here: What the docs show is what is tested.
"""

import pathlib

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
zarr = pytest.importorskip("zarr", minversion="3.0")

import clearscale as cs

EXAMPLES_PAGE = pathlib.Path(__file__).parent.parent / "docs" / "examples.md"
SCENE_EXAMPLE_HEADING = "## End to end: Two tiles that fit together in a Scene"


def _first_python_block_under(heading: str) -> str:
    section = EXAMPLES_PAGE.read_text().split(heading, 1)[1]
    section = section.split("\n## ", 1)[0]  # Up to the next heading of the same level
    return section.split("```python\n", 1)[1].split("```", 1)[0]


def _run(code: str):
    exec(compile(code, "docs/examples.md (Scene example)", "exec"), {"__name__": "__docs_example__"})


def test_scene_example_runs_and_writes_a_readable_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    _run(_first_python_block_under(SCENE_EXAMPLE_HEADING))  # Its own assertions are the substance

    output = tmp_path / "demo-output"
    assert (output / "two_tiles.png").is_file()
    root = zarr.open_group(str(output / "two_tiles.ome.zarr"), mode="r")
    (scene,) = cs.OmeZarrGroup.from_group(root).scenes
    assert sorted(scene.unresolved_paths) == ["tile1", "tile2"]
    for path in ("tile1", "tile2"):
        assert list(cs.OmeZarrGroup.from_group(root[path]).multiscales[0]) == ["s0", "s1", "s2"]


def test_scene_example_checks_fail_for_a_wrong_registration(tmp_path, monkeypatch):
    """The page says: Change the angle in the Scene only (say to 9°), and the checks fail."""
    monkeypatch.chdir(tmp_path)
    code = _first_python_block_under(SCENE_EXAMPLE_HEADING)
    registration = 'tile2_to_tile1 = cs.Affine.from_linear_and_offset(rotation, shift_um, axes="yx")'
    assert code.count(registration) == 1, "The example changed: Update this test."
    wrong_angle = code.replace(
        registration,
        "wrong = np.deg2rad(9)\n"
        "tile2_to_tile1 = cs.Affine.from_linear_and_offset(\n"
        "    [[np.cos(wrong), -np.sin(wrong)], [np.sin(wrong), np.cos(wrong)]], shift_um, axes='yx'\n"
        ")",
    )

    with pytest.raises(AssertionError):
        _run(wrong_angle)
