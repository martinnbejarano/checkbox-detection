"""Golden tests: end-to-end accuracy of each (locator, classifier) combo on each sample stays at
or above the floor recorded after tuning (spec §8). The samples are evaluation only; nothing was
tuned on them. The `-rotN` samples are tilted or turned N degrees and made upright by stage 0.

A floor is the accuracy measured after tuning minus about one box, so a real regression fails
while a one-box wobble does not. Re-record a floor only with the `make eval` tables in the PR.
"""

import json
from pathlib import Path

import pytest
from conftest import MODELS_DIR
from fastapi.testclient import TestClient

from checkbox_detection.config import Settings
from checkbox_detection.detection.geometry import match
from checkbox_detection.detection.types import Rect
from checkbox_detection.main import create_app

REPO = Path(__file__).resolve().parents[2]
GROUND_TRUTH = REPO / "backend" / "tests" / "data" / "ground_truth"

# (locator, classifier) -> end-to-end accuracy floor (%) per sample, recorded after tuning.
FLOORS: dict[tuple[str, str], dict[str, float]] = {
    ("opencv", "fill_ratio"): {
        "sample-000": 99.1,
        "sample-001": 90.6,
        "sample-002": 87.5,
        "sample-003": 97.4,
        "sample-000-rot3": 98.3,
        "sample-001-rot2": 76.7,
        "sample-002-rot4": 87.5,
        "sample-003-rot5": 97.4,
        "sample-000-rot90": 99.1,
        "sample-001-rot90": 90.6,
        "sample-002-rot90": 87.5,
        "sample-003-rot90": 97.4,
    },
    ("opencv", "cnn"): {
        "sample-000": 99.1,
        "sample-001": 90.6,
        "sample-002": 87.5,
        "sample-003": 97.4,
        "sample-000-rot3": 98.3,
        "sample-001-rot2": 76.7,
        "sample-002-rot4": 87.5,
        "sample-003-rot5": 97.4,
        "sample-000-rot90": 99.1,
        "sample-001-rot90": 90.6,
        "sample-002-rot90": 87.5,
        "sample-003-rot90": 97.4,
    },
    ("yolox", "fill_ratio"): {
        "sample-000": 99.1,
        "sample-001": 95.3,
        "sample-002": 97.9,
        "sample-003": 96.2,
        "sample-000-rot3": 96.6,
        "sample-001-rot2": 95.3,
        "sample-002-rot4": 97.9,
        "sample-003-rot5": 92.4,
        "sample-000-rot90": 99.1,
        "sample-001-rot90": 95.3,
        "sample-002-rot90": 97.9,
        "sample-003-rot90": 97.4,
    },
    ("yolox", "cnn"): {
        "sample-000": 99.1,
        "sample-001": 95.3,
        "sample-002": 97.9,
        "sample-003": 96.2,
        "sample-000-rot3": 97.4,
        "sample-001-rot2": 95.3,
        "sample-002-rot4": 97.9,
        "sample-003-rot5": 98.7,
        "sample-000-rot90": 99.1,
        "sample-001-rot90": 95.3,
        "sample-002-rot90": 97.9,
        "sample-003-rot90": 98.7,
    },
}


def end_to_end(client: TestClient, sample: str) -> float:
    """Share of ground-truth boxes found at IoU >= 0.5 with the right state, in %."""
    truth = json.loads((GROUND_TRUTH / f"{sample}.json").read_text())
    image = (REPO / truth["image"]).read_bytes()
    response = client.post("/detect", files={"file": ("page", image)})
    found = response.json()["boxes"]
    pairs = match(
        [Rect(*t["bbox"], 1.0) for t in truth["boxes"]],
        [Rect(*b["bbox"], b["score"]) for b in found],
        0.5,
    )
    right = sum(truth["boxes"][t]["is_checked"] == found[f]["is_checked"] for t, f in pairs)
    return 100 * right / len(truth["boxes"])


CASES = [
    (*combo, sample, floor) for combo, floors in FLOORS.items() for sample, floor in floors.items()
]


@pytest.mark.parametrize(
    ("locator", "classifier", "sample", "floor"),
    CASES,
    ids=[f"{loc}+{cls}-{sample}" for loc, cls, sample, _ in CASES],
)
def test_end_to_end_accuracy_stays_above_floor(
    locator: str, classifier: str, sample: str, floor: float
) -> None:
    settings = Settings(locator=locator, classifier=classifier, models_dir=MODELS_DIR)
    with TestClient(create_app(settings)) as client:
        accuracy = end_to_end(client, sample)

    assert accuracy >= floor
