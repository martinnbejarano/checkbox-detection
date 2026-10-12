"""Accuracy and latency of every locator x classifier combo against the ground truth (spec §7).

Prints Markdown tables per combo for "samples" and "external (unseen)", then by tag and
by scale (samples only) and, with --labels, label accuracy (samples only: the external set has
no labels). Combos whose model files are missing are skipped with a note.

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --scales 0.5,1,1.5 --labels --locator opencv
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from checkbox_detection.config import Settings
from checkbox_detection.detection import CLASSIFIERS, LOCATORS, Pipeline, build_pipeline
from checkbox_detection.detection.geometry import iou
from checkbox_detection.detection.types import Rect
from checkbox_detection.imaging import load_pages

REPO = Path(__file__).resolve().parents[2]  # ground-truth image paths are relative to it
DATA = REPO / "backend" / "tests" / "data"
SAMPLES, EXTERNAL = "samples", "external (unseen)"
SETS = {SAMPLES: DATA / "ground_truth", EXTERNAL: DATA / "external"}
MATCH_IOU, LOOSE_IOU = 0.5, 0.3


def match(truth: list[Rect], found: list[Rect], min_iou: float) -> list[tuple[int, int]]:
    """Greedy one-to-one (truth index, found index) pairs, highest IoU first, IoU >= min_iou."""
    candidates = sorted(
        ((iou(t, f), ti, fi) for ti, t in enumerate(truth) for fi, f in enumerate(found)),
        reverse=True,
    )
    pairs: list[tuple[int, int]] = []
    for score, ti, fi in candidates:
        if score < min_iou:
            break
        if all(ti != t and fi != f for t, f in pairs):
            pairs.append((ti, fi))
    return sorted(pairs)


def same_label(found: str | None, truth: str) -> bool:
    """Equal after ignoring case and runs of whitespace."""

    def norm(text: str) -> str:
        return " ".join(text.split()).casefold()

    return found is not None and norm(found) == norm(truth)


@dataclass
class Score:
    """Counts for one page, or summed over pages."""

    truth: int = 0
    found: int = 0
    matched: int = 0  # at MATCH_IOU
    matched_loose: int = 0  # at LOOSE_IOU
    false_checked: int = 0  # matched, truth unchecked, found checked
    false_unchecked: int = 0
    labeled: int = 0  # matched boxes whose truth has a label
    label_ok: int = 0
    tag_truth: Counter[str] = field(default_factory=Counter)
    tag_ok: Counter[str] = field(default_factory=Counter)  # matched with the right state
    ms: list[float] = field(default_factory=list)  # detection time per page
    label_ms: list[float] = field(default_factory=list)

    def __iadd__(self, other: "Score") -> "Score":
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)
        return self

    @property
    def state_ok(self) -> int:
        return self.matched - self.false_checked - self.false_unchecked


def summed(scores: Iterable[Score]) -> Score:
    total = Score()
    for s in scores:
        total += s
    return total


def score_page(
    pipeline: Pipeline, image: np.ndarray, truth: list[dict[str, Any]], labels: bool
) -> Score:
    start = perf_counter()
    boxes = pipeline.run(image)
    score = Score(ms=[(perf_counter() - start) * 1000])
    if labels:
        start = perf_counter()
        boxes = pipeline.label(image, boxes)
        score.label_ms.append((perf_counter() - start) * 1000)

    rects = [Rect(*t["bbox"], score=1.0) for t in truth]
    score.truth, score.found = len(truth), len(boxes)
    score.matched_loose = len(match(rects, [b.rect for b in boxes], LOOSE_IOU))
    for t in truth:
        score.tag_truth.update(t.get("tags", []))
    for ti, fi in match(rects, [b.rect for b in boxes], MATCH_IOU):
        t, b = truth[ti], boxes[fi]
        score.matched += 1
        score.false_checked += b.is_checked and not t["is_checked"]
        score.false_unchecked += t["is_checked"] and not b.is_checked
        if b.is_checked == t["is_checked"]:
            score.tag_ok.update(t.get("tags", []))
        if labels and t.get("label") is not None:
            score.labeled += 1
            score.label_ok += same_label(b.label, t["label"])
    return score


def load(truth_file: Path, scale: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The page decoded as the service decodes it, and its truth, both resized by `scale`."""
    truth = json.loads(truth_file.read_text())
    image = load_pages((REPO / truth["image"]).read_bytes(), Settings())[0].image
    boxes = truth["boxes"]
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
        boxes = [{**b, "bbox": [round(v * scale) for v in b["bbox"]]} for b in boxes]
    return image, boxes


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}" if whole else "-"


def ms(values: list[float], q: int) -> str:
    return f"{np.percentile(values, q):.0f}" if values else "-"


def detail_table(scores: dict[str, Score]) -> list[str]:
    """One row per page plus a total row."""
    total = summed(scores.values())
    lines = [
        "| page | boxes | P | R | F1 | R@0.3 | state acc | false ✓ | false ☐ | **end-to-end** "
        "| ms p50 | ms p95 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name, s in [*scores.items(), ("**total**", total)]:
        f1 = pct(2 * s.matched, s.truth + s.found)
        lines.append(
            f"| {name} | {s.truth} | {pct(s.matched, s.found)} | {pct(s.matched, s.truth)} "
            f"| {f1} | {pct(s.matched_loose, s.truth)} | {pct(s.state_ok, s.matched)} "
            f"| {s.false_checked} | {s.false_unchecked} | **{pct(s.state_ok, s.truth)}** "
            f"| {ms(s.ms, 50)} | {ms(s.ms, 95)} |"
        )
    return lines


def main(locators: list[str], classifiers: list[str], scales: list[float], labels: bool) -> None:
    settings = Settings()
    cv2.setNumThreads(settings.cv_threads)  # as the service runs, so latency is comparable
    models_dir = REPO / "backend" / settings.models_dir
    totals: dict[tuple[str, str, float], Score] = {}  # (combo, set, scale) -> summed score
    out: list[str] = []

    for locator in locators:
        for classifier in classifiers:
            combo = f"{locator} + {classifier}"
            try:
                pipeline = build_pipeline(locator, classifier, models_dir, settings.onnx_threads)
            except FileNotFoundError as e:
                out += [f"> skipped {combo}: {e}", ""]
                continue
            for set_name, folder in SETS.items():
                for scale in scales if set_name == SAMPLES else [1.0]:
                    scores = {
                        f.stem: score_page(
                            pipeline, *load(f, scale), labels and set_name == SAMPLES and scale == 1
                        )
                        for f in sorted(folder.glob("*.json"))
                    }
                    totals[combo, set_name, scale] = summed(scores.values())
                    if scale == 1:
                        out += [f"### {set_name}: {combo}", "", *detail_table(scores), ""]

    combos = list(dict.fromkeys(c for c, _, _ in totals))
    samples = {c: totals[c, SAMPLES, 1.0] for c in combos if (c, SAMPLES, 1.0) in totals}
    tags = sorted({t for s in samples.values() for t in s.tag_truth})
    out += [
        "### End-to-end accuracy by tag (samples)",
        "",
        "| combo | " + " | ".join(tags) + " |",
        "|---|" + "--:|" * len(tags),
        *(
            f"| {c} | "
            + " | ".join(f"{pct(s.tag_ok[t], s.tag_truth[t])} (n={s.tag_truth[t]})" for t in tags)
            + " |"
            for c, s in samples.items()
        ),
        "",
        "### End-to-end accuracy by scale (samples)",
        "",
        "| combo | " + " | ".join(f"{x:g}x" for x in scales) + " |",
        "|---|" + "--:|" * len(scales),
        *(
            f"| {c} | "
            + " | ".join(
                pct(totals[c, SAMPLES, x].state_ok, totals[c, SAMPLES, x].truth) for x in scales
            )
            + " |"
            for c in combos
        ),
        "",
    ]
    if labels:
        out += [
            "### Label accuracy (samples, matched boxes with a ground-truth label)",
            "",
            "| combo | labels | accuracy | OCR ms/page p50 | OCR ms/page p95 |",
            "|---|--:|--:|--:|--:|",
            *(
                f"| {c} | {s.labeled} | {pct(s.label_ok, s.labeled)} "
                f"| {ms(s.label_ms, 50)} | {ms(s.label_ms, 95)} |"
                for c, s in samples.items()
            ),
            "",
        ]
    print("\n".join(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locator", choices=sorted(LOCATORS), help="only this locator")
    parser.add_argument("--classifier", choices=sorted(CLASSIFIERS), help="only this classifier")
    parser.add_argument(
        "--scales", default="1", help="comma-separated resize factors, e.g. 0.5,1,1.5"
    )
    parser.add_argument("--labels", action="store_true", help="also run and score the label stage")
    args = parser.parse_args()
    scales = sorted({1.0, *(float(x) for x in args.scales.split(","))})
    main(
        [args.locator] if args.locator else list(LOCATORS),
        [args.classifier] if args.classifier else list(CLASSIFIERS),
        scales,
        args.labels,
    )
