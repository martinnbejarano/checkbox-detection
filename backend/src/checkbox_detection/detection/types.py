"""Domain types passed between the pipeline stages and to `api`. Imports nothing else."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    """Output of stage 1 (locate), input of stages 2 (classify) and 3 (label)."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass(frozen=True, slots=True)
class Box:
    """One checkbox as the API serializes it."""

    rect: Rect
    is_checked: bool
    checked_score: float  # ink ratio (fill_ratio) or P(filled) (cnn); internal, not in response
    label: str | None = None
