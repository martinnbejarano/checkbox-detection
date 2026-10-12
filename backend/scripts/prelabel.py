"""Run a pipeline on an image and write what it finds as a ground-truth JSON draft to correct.

Tags are left empty; the label is filled only with --labels.

    uv run python scripts/prelabel.py ../samples/sample-000.png draft.json
    uv run python scripts/prelabel.py ../samples/sample-000.png draft.json \\
        --locator yolox --classifier cnn --labels
"""

import argparse
import json
from pathlib import Path

from checkbox_detection.config import Settings
from checkbox_detection.detection import build_pipeline
from checkbox_detection.imaging import load_pages

REPO = Path(__file__).resolve().parents[2]  # ground-truth image paths are relative to it


def prelabel(image: Path, out: Path, locator: str, classifier: str, labels: bool) -> None:
    settings = Settings()
    pipeline = build_pipeline(
        locator, classifier, REPO / "backend" / settings.models_dir, settings.onnx_threads
    )
    page = load_pages(image.read_bytes(), settings)[0].image
    boxes, _ = pipeline.detect(page, labels)
    draft = {
        "image": image.resolve().relative_to(REPO).as_posix(),
        "boxes": [
            {
                "bbox": [b.rect.x1, b.rect.y1, b.rect.x2, b.rect.y2],
                "is_checked": b.is_checked,
                "label": b.label,
                "tags": [],
            }
            for b in boxes
        ],
    }
    # One box per line, like the committed ground truth, so drafts diff cleanly against it.
    rows = ",\n".join(f"    {json.dumps(b)}" for b in draft["boxes"])
    out.write_text(f'{{\n  "image": {json.dumps(draft["image"])},\n  "boxes": [\n{rows}\n  ]\n}}\n')


if __name__ == "__main__":
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--locator", default=settings.locator)
    parser.add_argument("--classifier", default=settings.classifier)
    parser.add_argument("--labels", action="store_true", help="also read each box's label")
    args = parser.parse_args()
    prelabel(args.image, args.out, args.locator, args.classifier, args.labels)
