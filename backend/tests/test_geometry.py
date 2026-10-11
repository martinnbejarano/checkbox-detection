from checkbox_detection.detection.geometry import nms, sort_reading_order, tile_windows
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


def test_tile_windows_cover_the_page_with_overlap() -> None:
    windows = tile_windows(2550, 3300, tile=1280, overlap=200)

    # Evenly spread, the last one flush with the page edge; every overlap is at least 200 px.
    xs = sorted({(x1, x2) for x1, _, x2, _ in windows})
    ys = sorted({(y1, y2) for _, y1, _, y2 in windows})
    assert xs == [(0, 1280), (635, 1915), (1270, 2550)]
    assert ys == [(0, 1280), (1010, 2290), (2020, 3300)]
    assert len(windows) == 9


def test_tile_windows_on_a_page_smaller_than_a_tile_is_the_page() -> None:
    assert tile_windows(900, 700, tile=1280, overlap=200) == [(0, 0, 900, 700)]
