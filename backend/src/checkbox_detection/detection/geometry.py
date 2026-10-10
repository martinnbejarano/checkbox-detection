"""Pure box geometry helpers. Must not import anything outside `detection.types`."""

from checkbox_detection.detection.types import Box


def sort_reading_order(boxes: list[Box]) -> list[Box]:
    # Exact top-then-left sort: boxes on one visual row with a few px of skew can swap order.
    # Row grouping by vertical overlap comes with real-form tuning (#18).
    return sorted(boxes, key=lambda b: (b.rect.y1, b.rect.x1))
