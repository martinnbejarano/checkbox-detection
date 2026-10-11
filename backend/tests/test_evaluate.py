import pytest
from evaluate import match, same_label

from checkbox_detection.detection.types import Rect


def test_match_takes_the_highest_iou_pairs_first_one_to_one() -> None:
    truth = [
        Rect(10, 0, 110, 100, 1.0),  # a
        Rect(30, 0, 130, 100, 1.0),  # b, overlaps a
        Rect(500, 500, 540, 540, 1.0),  # c, nothing near it
    ]
    found = [
        Rect(25, 0, 125, 100, 1.0),  # p: IoU 0.74 with a, 0.90 with b
        Rect(0, 0, 90, 100, 1.0),  # q: IoU 0.73 with a, 0.46 with b
    ]

    # Taking truths in order would give a→p and b→q; b→p is the better pair, so a gets q.
    assert match(truth, found, min_iou=0.3) == [(0, 1), (1, 0)]


def test_match_keeps_iou_at_the_threshold_and_drops_below_it() -> None:
    truth = [Rect(0, 0, 100, 100, 1.0), Rect(300, 0, 400, 100, 1.0)]
    found = [
        Rect(0, 0, 100, 50, 1.0),  # IoU exactly 0.5 with truth 0
        Rect(300, 0, 400, 49, 1.0),  # IoU 0.49 with truth 1
    ]

    assert match(truth, found, min_iou=0.5) == [(0, 0)]


@pytest.mark.parametrize(
    ("found", "truth", "same"),
    [
        ("Fee Simple", "Fee Simple", True),
        ("FEE simple", "Fee Simple", True),
        ("  Fee   Simple ", "Fee Simple", True),
        ("Fee Simple.", "Fee Simple", False),
        (None, "Fee Simple", False),
    ],
    ids=["equal", "case", "whitespace", "other_text", "no_text"],
)
def test_same_label_ignores_case_and_whitespace_only(
    found: str | None, truth: str, same: bool
) -> None:
    assert same_label(found, truth) is same
