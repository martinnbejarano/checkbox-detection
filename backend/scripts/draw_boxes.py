"""Draw boxes on an image for visual review: the ground truth, or what a pipeline finds.

Green is checked, red is unchecked; with ground truth the label is written above each box.

    uv run python scripts/draw_boxes.py tests/data/ground_truth/sample-000.json out.png
    uv run python scripts/draw_boxes.py tests/data/ground_truth/sample-000.json out.png \\
        --locator opencv --classifier fill_ratio
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from checkbox_detection.config import Settings
from checkbox_detection.detection import build_pipeline

REPO = Path(__file__).resolve().parents[2]  # ground-truth image paths are relative to it


def draw_boxes(ground_truth: Path, out: Path, locator: str | None, classifier: str | None) -> None:
    truth = json.loads(ground_truth.read_text())
    image = Image.open(REPO / truth["image"]).convert("RGB")
    if locator or classifier:
        settings = Settings()  # a missing flag takes the service default
        pipeline = build_pipeline(
            locator or settings.locator,
            classifier or settings.classifier,
            REPO / "backend" / settings.models_dir,
            settings.onnx_threads,
        )
        boxes = [
            {"bbox": [b.rect.x1, b.rect.y1, b.rect.x2, b.rect.y2], "is_checked": b.is_checked}
            for b in pipeline.run(np.asarray(image))
        ]
    else:
        boxes = truth["boxes"]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, image.width // 120))
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        color = (0, 160, 0) if box["is_checked"] else (220, 0, 0)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        if box.get("label"):
            draw.text((x1, y1 - 2), box["label"], fill=color, font=font, anchor="lb")
    image.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--locator", help="draw this pipeline's boxes instead of the ground truth")
    parser.add_argument(
        "--classifier", help="draw this pipeline's boxes instead of the ground truth"
    )
    args = parser.parse_args()
    draw_boxes(args.ground_truth, args.out, args.locator, args.classifier)
