import pytest

from clearscale._services.display import MAX_LISTED, counted, list_truncated, truncated


@pytest.mark.parametrize("count", [0, 1, MAX_LISTED - 1, MAX_LISTED])
@pytest.mark.parametrize("indent", ["", "  "])
def test_list_truncated_formats_all_items_up_to_the_limit(count, indent):
    assert list_truncated(range(count), str, "numbers", indent=indent) == [f"{indent}{i}" for i in range(count)]


@pytest.mark.parametrize("extra", [1, 3])
@pytest.mark.parametrize("indent", ["", "  "])
def test_list_truncated_summarizes_the_rest_with_the_noun_and_indent(extra, indent):
    lines = list_truncated(range(MAX_LISTED + extra), str, "numbers", indent=indent)
    assert lines == [f"{indent}{i}" for i in range(MAX_LISTED)] + [f"{indent}... {extra} more numbers"]


@pytest.mark.parametrize(
    "text, max_length, expected",
    [
        ("", 5, ""),
        ("abcd", 5, "abcd"),
        ("abcde", 5, "abcde"),
        ("abcdef", 5, "ab..."),
        ("abcdefgh", 6, "abc..."),
        ("abcd", 3, "..."),
    ],
)
def test_truncated(text, max_length, expected):
    assert truncated(text, max_length) == expected


@pytest.mark.parametrize("count, expected", [(0, "0 paths"), (1, "1 path"), (2, "2 paths")])
def test_counted_pluralizes_the_noun(count, expected):
    assert counted(count, "path") == expected
