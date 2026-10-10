from checkbox_detection.detection.geometry import nms, sort_reading_order
from checkbox_detection.detection.types import Box, Rect


def test_nms_keeps_the_outer_of_nested_duplicates() -> None:
    outer = Rect(100, 100, 150, 150, 1.0)
    inner = Rect(104, 104, 146, 146, 1.0)  # inner border of the same box, IoU ~0.7
    neighbour = Rect(140, 100, 190, 150, 1.0)  # touches the outer box, IoU ~0.1
    far = Rect(500, 500, 520, 520, 1.0)

    assert set(nms([inner, far, outer, neighbour], max_iou=0.3)) == {outer, neighbour, far}


def test_reading_order_groups_skewed_rows() -> None:
    def box(x: int, y: int) -> Box:
        return Box(Rect(x, y, x + 40, y + 40, 1.0), False, 0.0)

    # Row 1 slopes down to the left by a few px, so a plain (y, x) sort would put c first.
    a, b, c = box(100, 206), box(300, 203), box(500, 200)
    d, e = box(100, 300), box(300, 298)  # row 2

    assert sort_reading_order([e, c, d, b, a]) == [a, b, c, d, e]
