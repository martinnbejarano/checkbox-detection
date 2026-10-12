"""Shared by the training scripts: the built data (from data.py), the model paths, page loading."""

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from checkbox_detection.detection.geometry import match
from checkbox_detection.detection.locate import Locator
from checkbox_detection.detection.types import Rect

HERE = Path(__file__).resolve().parent
BUILD = HERE / "data" / "build"
PRETRAINED = HERE / "data" / "pretrained"
RUNS = HERE / "runs"
MODELS = HERE.parent / "models"  # the service's models dir: fine-tuned ONNX files go here


def split(name: str) -> dict[str, Any]:
    """The COCO json of a split, with the image folder."""
    coco: dict[str, Any] = json.loads(
        (BUILD / "annotations" / f"instances_{name}.json").read_text()
    )
    coco["dir"] = BUILD / name
    return coco


def boxes_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {i["id"]: [] for i in coco["images"]}
    for a in coco["annotations"]:
        out[a["image_id"]].append(a)
    return out


def load_rgb(coco: dict[str, Any], image: dict[str, Any]) -> np.ndarray:
    """A page as the service sees it: RGB uint8 (every built page is a 3-channel JPEG)."""
    return cv2.cvtColor(cv2.imread(str(coco["dir"] / image["file_name"])), cv2.COLOR_BGR2RGB)


def truth(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Boxes in the ground-truth shape scripts/evaluate.py scores (bbox as x1, y1, x2, y2)."""
    return [
        {
            "bbox": [round(x), round(y), round(x + w), round(y + h)],
            "is_checked": a["checked"],
        }
        for a in boxes
        for x, y, w, h in [a["bbox"]]
    ]


Page = tuple[np.ndarray, list[dict[str, Any]]]


def val_pages(limit: int | None) -> list[Page]:
    """(RGB page, truth) for validation pages: all of them, or a fixed random sample."""
    coco = split("val")
    pages = list(zip(coco["images"], boxes_by_image(coco).values(), strict=True))
    if limit:
        pages = random.Random(21).sample(pages, min(limit, len(pages)))
    return [(load_rgb(coco, im), truth(boxes)) for im, boxes in pages]


def detection_counts(locator: Locator, pages: list[Page]) -> tuple[int, int, int]:
    """(truth boxes, found boxes, matched at IoU 0.5) over the pages."""
    t = f = m = 0
    for image, gt in pages:
        found = locator.locate(image)
        t, f = t + len(gt), f + len(found)
        m += len(match([Rect(*g["bbox"], 1.0) for g in gt], found, 0.5))
    return t, f, m


def f2(t: int, f: int, m: int) -> float:
    """F-beta with beta 2: recall counts four times as much as precision (spec: favour recall)."""
    p, r = m / f if f else 0.0, m / t if t else 0.0
    return 5 * p * r / (4 * p + r) if p + r else 0.0
