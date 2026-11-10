from pathlib import Path
from types import SimpleNamespace
import pytest

pytest.importorskip("mkdocstrings_handlers.python", reason="Documentation tests require the docs extra")
import griffe  # type: ignore[import-not-found]

jinja2 = pytest.importorskip("jinja2")
from mkdocstrings_handlers.python import PythonOptions, do_filter_objects  # type: ignore[import-not-found]


@pytest.fixture(scope="module")
def documented_package():
    return griffe.load("clearscale", search_paths=[Path(__file__).resolve().parents[1] / "src"], allow_inspection=False)


def selected_members(obj):
    templates = Path(__file__).resolve().parents[1] / "scripts/doc_templates/python/material"
    env = jinja2.Environment(
        loader=jinja2.ChoiceLoader(
            [
                jinja2.DictLoader(
                    {
                        "_base/children.html.jinja": "{% for member in obj.all_members|filter_objects(members_list=config.members, inherited_members=config.inherited_members) %}{{ member.name }},{% endfor %}"
                    }
                ),
                jinja2.FileSystemLoader(templates),
            ]
        )
    )
    env.filters["filter_objects"] = do_filter_objects
    rendered = env.get_template("children.html.jinja").render(
        obj=obj, config=PythonOptions.from_data(filters="public", inherited_members=True)
    )
    return set(rendered.strip().split(",")) - {""}


def test_exports_exclude_underscore_names_and_incidental_imports(documented_package):
    root = selected_members(documented_package)
    assert "Shape" in root
    assert "__version__" not in root
    assert "_axis_values" not in root
    assert "Axis" in selected_members(documented_package["ome_zarr"])
    types = selected_members(documented_package["types"])
    assert "ShapeSource" in types
    assert not {"Hashable", "Protocol", "_Ordered"} & types


def test_characterization_intentional_omissions_remain_hidden(documented_package):
    members = selected_members(documented_package["characterization"])
    assert "characterize_shape_scaling_method" in members
    assert "half_pixel_space_preservation" in members
    assert not {"ScalingMethodCharacterization", "ShapeScalingKwargs"} & members


def test_reexports_keep_inherited_methods_but_hide_dunders(documented_package):
    members = selected_members(documented_package["Shape"])
    assert {"with_axes", "keys_tuple"} <= members
    assert not {"__init__", "__iter__", "__eq__"} & members
    multiscale = documented_package["Multiscale"]
    assert "__init__" not in selected_members(multiscale)
    assert multiscale.all_members["__init__"].docstring is not None
    assert "has_shapes" in multiscale.all_members["__init__"].parameters


def test_empty_selection_does_not_fall_back_to_all_members():
    hidden = griffe.Function("__hidden__")
    obj = SimpleNamespace(all_members={"__hidden__": griffe.Alias("__hidden__", hidden, inherited=True)})
    assert selected_members(obj) == set()
