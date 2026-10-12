"""Choose OpenCVParams, BinarizeParams and FillParams on the validation split.

Coordinate descent: one field at a time is set to each of its candidates while the others stay
fixed, the best value is kept, and the pass is repeated until nothing changes. A field moves only
when it gains at least MIN_GAIN on validation; smaller gains are noise of this split and do not
carry to other forms. Prints one table row per try.

- opencv + binarize: detection F2 at IoU 0.5 (recall weighs 4x precision; spec: favour recall).
- fill_ratio: state accuracy on the true boxes whose state is trusted; ties go to the higher
  threshold, so an ambiguous ratio resolves to unchecked.

LabelParams are not searched: a search against the PDF text layer of synthetic pages chose values
that read the samples' labels worse, so its result was not applied and the search was removed.

    uv run python search.py opencv|fill
"""

import argparse
from collections.abc import Callable
from dataclasses import fields, replace
from functools import partial
from typing import Any

import numpy as np
from common import detection_counts, f2, val_pages

from checkbox_detection.detection import binarize as binarize_module
from checkbox_detection.detection.classify import fill_ratio
from checkbox_detection.detection.locate import opencv
from checkbox_detection.detection.types import Rect

# A field moves only when it gains at least this much (0.5 points), in every stage. Why:
# .agents/research/tuning-results.md, "Choices informed by evaluation data".
MIN_GAIN = 0.005
VAL_PAGES = 300  # opencv: a fixed random sample of validation pages

OPENCV_GRID: dict[str, tuple[Any, ...]] = {
    "min_aspect": (0.6, 0.7, 0.8),
    "max_aspect": (1.25, 1.4, 1.6),
    "poly_epsilon_frac": (0.03, 0.05, 0.08),
    "min_rectangularity": (0.7, 0.8, 0.85),
    "confident_rectangularity": (0.82, 0.86, 0.9),
    "max_confident_border_frac": (0.08, 0.11, 0.15),
    "min_hole_frac": (0.15, 0.25, 0.35),
    "line_length_factor": (1.2, 1.5, 2.0),
    "median_band": ((0.6, 1.5), (0.75, 1.33), (0.85, 1.2)),
    "min_confident": (3, 5, 8),
}
BINARIZE_GRID: dict[str, tuple[Any, ...]] = {
    "block_frac": (0.01, 0.02, 0.03),
    "threshold_offset": (5, 10, 15, 20),
    "min_overlay_saturation": (50, 70, 90),
}
FILL_GRID: dict[str, tuple[Any, ...]] = {
    "min_margin_frac": (0.1, 0.15, 0.2),
    "border_gap_frac": (0.0, 0.05, 0.1),
    "min_inside_share": (0.5, 0.6, 0.7, 0.8),
    "edge_length_frac": (0.7, 0.8, 0.9),
}
FILL_THRESHOLDS = np.round(np.arange(0.02, 0.21, 0.01), 2)


def descend(
    name: str, start: Any, grid: dict[str, tuple[Any, ...]], score: Callable[[Any], float]
) -> Any:
    best, best_score = start, score(start)
    print(
        f"\n### {name}\n\n| field | value | score |\n|---|---|--:|\n| start | | {best_score:.4f} |"
    )
    changed = True
    while changed:
        changed = False
        for field, values in grid.items():
            for value in values:
                if value == getattr(best, field):
                    continue
                trial = replace(best, **{field: value})
                s = score(trial)
                print(f"| {field} | {value} | {s:.4f} |", flush=True)
                if s >= best_score + MIN_GAIN:
                    best, best_score, changed = trial, s, True
    defaults = type(start)()
    moved = {
        f.name: getattr(best, f.name)
        for f in fields(best)
        if getattr(best, f.name) != getattr(defaults, f.name)
    }
    print(f"\n{name}: best {best_score:.4f}, changed from defaults: {moved}")
    return best


def use_binarize(params: binarize_module.BinarizeParams) -> None:
    """Every stage that binarizes calls these names; point them at `params`."""
    for module in (opencv, fill_ratio):
        module.binarize = partial(binarize_module.binarize, params=params)  # type: ignore[attr-defined]
        module.to_clean_gray = partial(binarize_module.to_clean_gray, params=params)  # type: ignore[attr-defined]


def search_opencv() -> None:
    pages = val_pages(VAL_PAGES)
    locator = opencv.OpenCVLocator()

    def detection(params: opencv.OpenCVParams) -> float:
        locator.params = params
        return f2(*detection_counts(locator, pages))

    best = descend("OpenCVParams", opencv.OpenCVParams(), OPENCV_GRID, detection)
    locator.params = best

    def with_binarize(params: binarize_module.BinarizeParams) -> float:
        use_binarize(params)
        return detection(best)

    descend("BinarizeParams", binarize_module.BinarizeParams(), BINARIZE_GRID, with_binarize)


def search_fill() -> None:
    pages = [(img, [g for g in gt if g["is_checked"] is not None]) for img, gt in val_pages(None)]
    pages = [(img, gt) for img, gt in pages if gt]
    labels = np.array([g["is_checked"] for _, gt in pages for g in gt])

    def ratios(params: fill_ratio.FillParams) -> np.ndarray:
        out = []
        for image, gt in pages:
            binary = fill_ratio.binarize(fill_ratio.to_clean_gray(image))
            out += [fill_ratio.mark_ratio(binary, Rect(*g["bbox"], 1.0), params) for g in gt]
        return np.array(out)

    def accuracy(params: fill_ratio.FillParams) -> float:
        r = ratios(params)
        return max(np.mean((r > t) == labels) for t in FILL_THRESHOLDS)

    best = descend(
        "FillParams (threshold swept per try)", fill_ratio.FillParams(), FILL_GRID, accuracy
    )
    r = ratios(best)
    print("\n| fill_threshold | accuracy | checked recall | unchecked recall |\n|--:|--:|--:|--:|")
    for t in FILL_THRESHOLDS:
        acc, checked, unchecked = (
            np.mean((r > t) == labels),
            np.mean(r[labels] > t),
            np.mean(r[~labels] <= t),
        )
        print(f"| {t} | {acc:.4f} | {checked:.4f} | {unchecked:.4f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["opencv", "fill"])
    {"opencv": search_opencv, "fill": search_fill}[parser.parse_args().stage]()
