import pytest

from clearscale._services.display import MAX_LISTED, counted, list_truncated


@pytest.mark.parametrize("count", [0, 1, MAX_LISTED - 1, MAX_LISTED])
@pytest.mark.parametrize("indent", ["", "  "])
def test_list_truncated_formats_all_items_up_to_the_limit(count, indent):
    assert list_truncated(range(count), str, "numbers", indent=indent) == [f"{indent}{i}" for i in range(count)]


@pytest.mark.parametrize("extra", [1, 3])
@pytest.mark.parametrize("indent", ["", "  "])
def test_list_truncated_summarizes_the_rest_with_the_noun_and_indent(extra, indent):
    lines = list_truncated(range(MAX_LISTED + extra), str, "numbers", indent=indent)
    assert lines == [f"{indent}{i}" for i in range(MAX_LISTED)] + [f"{indent}... {extra} more numbers"]


def test_counted_pluralizes_the_noun():
    assert [counted(n, "path") for n in (0, 1, 2)] == ["0 paths", "1 path", "2 paths"]
