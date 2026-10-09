"""Domain types shared by every detector."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    is_checked: bool
    score: float  # detector confidence or fill ratio; internal, not in the response
