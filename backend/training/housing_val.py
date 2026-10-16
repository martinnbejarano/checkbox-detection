"""Build a validation set of housing-form pages, in the ground-truth JSON format of
`backend/tests/data/`, from the fork of Roboflow `annotation-mmvj7/new_merge_project`.

The 15 pages of `backend/tests/data/external/` (the test set) come from the same project and are
left out, with any page that is a pixel copy of one of them. The selection rules are the test
set's (its README): full-resolution pages whose boxes are printed squares, with at least one
checked and one unchecked box, deduplicated, the first pages of each layout group in file order.
The test set took the first pages of each group, so this takes the ones after them.

Each source region spans the box and its label text; the printed square sits at its left end.
It is found as the square ring of the page's box side (the median side the service's own
locator returns) with the most ink on the ring and the least just outside it. A region without
one (a box under a heavy pen stroke) gets a square of that side, placed where the page's found
squares sit in their regions.

    uv run python housing_val.py   # writes data/housing_val/
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from checkbox_detection.detection.binarize import binarize, to_clean_gray
from checkbox_detection.detection.locate.yolox import YoloxLocator

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "new_merge_project"
OUT = HERE / "data" / "housing_val"
TEST = HERE.parent / "tests" / "data" / "external"
MODELS = HERE.parent / "models"
REMAP = {"checked": True, "unchecked": False}
PER_GROUP = 12  # pages kept per (file-name prefix, page size) group
# Pages marked by circling a word (out of scope) and pages whose boxes are faint dotted glyphs.
SKIP_PREFIXES = ("CSD", "Annualrecert")
SEARCH = (0.3, 1.0)  # the square's centre lies within this many region heights of its left edge
RING = 2  # ring thickness in px, inside the square's edge; the outside band is as thick
MIN_RING_INK = 0.5  # share of the ring that is ink for a square to count as found
OUTSIDE_WEIGHT = 0.3  # ink just outside the square counts against it this much (a pen stroke)
LEFT_BIAS = 0.01  # score lost per px to the right, against a letter of the label
MAX_ESTIMATED = 0.25  # pages with more estimated boxes than this share are left out
# A found square this many region heights off the page's usual spot in the region is a letter.
MAX_OFFSET = 0.4
# Two pages of the same size are one page when fewer than this share of the pixels differ by
# more than 64 on a 512 px wide copy: another fill of the same blank form differs by its ink.
DUPLICATE_SHARE = 0.001


def stem(file_name: str) -> str:
    return re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", file_name)


def prefix(name: str) -> str:
    return re.sub(r"[-_]?\d.*$", "", name)


def thumb(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 512 / gray.shape[1]
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.int16)


def same_page(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.mean(np.abs(a - b) > 64)) < DUPLICATE_SHARE


class RingScores:
    """For every pixel as a box's centre and each odd (width, height): the share of ink on the
    box's border ring and on the band just outside it, from box-filter sums of the binary page."""

    def __init__(self, binary: np.ndarray, shapes: list[tuple[int, int]]) -> None:
        ink = (binary > 0).astype(np.float32)

        def window(w: int, h: int) -> np.ndarray:
            return cv2.boxFilter(ink, -1, (w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)

        self.ring, self.outside = {}, {}
        for w, h in shapes:
            t = 1 if min(w, h) < 16 else RING  # a small box has a 1 px border
            full, inner, outer = (
                window(w, h),
                window(w - 2 * t, h - 2 * t),
                window(w + 2 * t, h + 2 * t),
            )
            self.ring[w, h] = (full - inner) / (w * h - (w - 2 * t) * (h - 2 * t))
            self.outside[w, h] = (outer - full) / ((w + 2 * t) * (h + 2 * t) - w * h)

    def best_in(self, region: list[float]) -> tuple[float, list[int]] | None:
        """(ring ink share, [x1, y1, x2, y2]) of the best box whose centre is in the region's
        left end, or None when no box there has MIN_RING_INK on its ring."""
        x, y, _, h = region
        best: tuple[float, float, list[int]] | None = None
        for (bw, bh), ring in self.ring.items():
            rx, ry = bw // 2, bh // 2
            x_lo, x_hi = max(rx, round(x - SEARCH[0] * h)), round(x + SEARCH[1] * h)
            y_lo, y_hi = max(ry, round(y - 0.3 * h)), round(y + 1.3 * h)
            sub = ring[y_lo:y_hi, x_lo:x_hi]
            if sub.size == 0 or sub.max() < MIN_RING_INK:
                continue
            score = sub - OUTSIDE_WEIGHT * self.outside[bw, bh][y_lo:y_hi, x_lo:x_hi]
            score = score - LEFT_BIAS * np.arange(sub.shape[1])[None]
            score[sub < MIN_RING_INK] = -np.inf
            cy, cx = np.unravel_index(int(score.argmax()), score.shape)
            value = float(score[cy, cx])
            if best is None or value > best[0]:
                px, py = int(cx) + x_lo, int(cy) + y_lo
                best = (value, float(sub[cy, cx]), [px - rx, py - ry, px + rx + 1, py + ry + 1])
        return (best[1], best[2]) if best else None


def page_boxes(
    image: np.ndarray, regions: list[tuple[list[float], bool]], locator: YoloxLocator
) -> tuple[list[dict[str, Any]], int]:
    """Ground-truth boxes for a page, and how many were estimated rather than found."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = to_clean_gray(rgb)
    rects = locator.locate(gray)
    if len(rects) < 3:
        return [], len(regions)
    bw = int(np.median([r.x2 - r.x1 for r in rects])) | 1
    bh = int(np.median([r.y2 - r.y1 for r in rects])) | 1
    shapes = [(bw + dw, bh + dh) for dw in (-2, 0, 2) for dh in (-2, 0, 2)]
    scores = RingScores(binarize(gray), shapes)
    found = [scores.best_in(r) for r, _ in regions]
    hits = [(r, b) for (r, _), f in zip(regions, found, strict=True) if f for b in [f[1]]]
    if len(hits) < 3:
        return [], len(regions)
    # Where a found square sits in its region, to place the ones not found. A found square far
    # from that spot is a letter of the label (the square under a pen stroke scored lower), so
    # it is placed the same way.
    dx = float(np.median([b[0] - r[0] for r, b in hits]))
    dy = float(np.median([(b[1] + b[3]) / 2 - (r[1] + r[3] / 2) for r, b in hits]))
    boxes = []
    estimated = 0
    for (region, checked), f in zip(regions, found, strict=True):
        x, y, _, h = region
        box = f[1] if f else None
        if box and (
            abs(box[0] - x - dx) > MAX_OFFSET * h
            or abs((box[1] + box[3]) / 2 - (y + h / 2) - dy) > MAX_OFFSET * h
        ):
            box = None
        if box is None:
            estimated += 1
            x1, y1 = round(x + dx), round(y + h / 2 + dy - bh / 2)
            box = [x1, y1, x1 + bw, y1 + bh]
        boxes.append({"bbox": box, "is_checked": checked, "label": None, "tags": []})
    return boxes, estimated


def main() -> None:
    test_thumbs = [thumb(cv2.imread(str(p))) for p in sorted(TEST.glob("*.jpg"))]
    candidates: dict[tuple[str, tuple[int, int]], list[tuple[str, Path, list]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for split in ("train", "valid", "test"):
        coco = json.loads((RAW / split / "_annotations.coco.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        for info in coco["images"]:
            name, size = stem(info["file_name"]), (info["width"], info["height"])
            anns = by_image[info["id"]]
            if size == (640, 640) or prefix(name) in SKIP_PREFIXES:
                continue
            if not anns or any(names[a["category_id"]] not in REMAP for a in anns):
                continue
            states = {REMAP[names[a["category_id"]]] for a in anns}
            if states != {True, False} or (name, *size) in seen:
                continue
            seen.add((name, *size))
            regions = [(a["bbox"], REMAP[names[a["category_id"]]]) for a in anns]
            candidates[prefix(name), size].append((name, RAW / split / info["file_name"], regions))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    locator = YoloxLocator(MODELS, 4)
    index = 0
    report: list[str] = []
    for group in sorted(candidates):
        kept = 0
        for name, path, regions in sorted(candidates[group], key=natural_key):
            if kept == PER_GROUP:
                break
            image = cv2.imread(str(path))
            if image is None or image.shape[1::-1] != group[1]:
                continue
            t = thumb(image)
            if any(same_page(t, tt) for tt in test_thumbs):
                report.append(f"skip {name}: a test page")
                continue
            boxes, estimated = page_boxes(image, regions, locator)
            if not boxes or estimated > MAX_ESTIMATED * len(boxes):
                report.append(f"skip {name}: squares not found ({estimated} of {len(regions)})")
                continue
            out_name = f"housing-{index:03d}"
            cv2.imwrite(str(OUT / f"{out_name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rel = (OUT / f"{out_name}.jpg").relative_to(HERE.parent.parent)
            truth: dict[str, Any] = {"image": str(rel), "boxes": boxes}
            (OUT / f"{out_name}.json").write_text(json.dumps(truth, indent=None))
            report.append(
                f"{out_name} <- {name} {group[1][0]}x{group[1][1]}: {len(boxes)} boxes "
                f"({sum(b['is_checked'] for b in boxes)} checked), {estimated} estimated"
            )
            index += 1
            kept += 1
    (OUT / "README.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"{index} pages -> {OUT}")


def natural_key(item: tuple[str, Path, list]) -> list[Any]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", item[0])]


if __name__ == "__main__":
    main()
