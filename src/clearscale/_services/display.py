from typing import Callable, List, Sequence, TypeVar

MAX_LISTED = 10
"""Scenes can have thousands of entries. repr and str list only this many of each kind, then summarize the rest."""

_T = TypeVar("_T")


def counted(n: int, noun: str) -> str:
    """'1 scale', '2 scales': a count as prose (not `scales=2`, which would suggest an attribute that may not exist)."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def list_truncated(items: Sequence[_T], fmt: Callable[[_T], str], noun: str, *, indent: str = "") -> List[str]:
    """Format the first `MAX_LISTED` items, then summarize how many more there are."""
    lines = [indent + fmt(item) for item in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        lines.append(f"{indent}... {len(items) - MAX_LISTED} more {noun}")
    return lines
