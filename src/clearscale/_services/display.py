from typing import Callable, List, Sequence, TypeVar

MAX_LISTED = 10
"""Scenes can have thousands of entries. repr and str list only this many of each kind, then summarize the rest."""

_T = TypeVar("_T")


def list_truncated(items: Sequence[_T], fmt: Callable[[_T], str], noun: str, *, indent: str = "") -> List[str]:
    """Format the first `MAX_LISTED` items, then summarize how many more there are."""
    lines = [indent + fmt(item) for item in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        lines.append(f"{indent}... {len(items) - MAX_LISTED} more {noun}")
    return lines
