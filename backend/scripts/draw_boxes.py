"""Draw ground-truth boxes on their image for visual review.

Green is checked, red is unchecked; the label is written above each box.

    uv run python scripts/draw_boxes.py tests/data/ground_truth/sample-000.json out.png
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]  # ground-truth image paths are relative to it


def draw_boxes(ground_truth: Path, out: Path) -> None:
    truth = json.loads(ground_truth.read_text())
    image = Image.open(REPO / truth["image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, image.width // 120))
    for box in truth["boxes"]:
        x1, y1, x2, y2 = box["bbox"]
        color = (0, 160, 0) if box["is_checked"] else (220, 0, 0)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        if box["label"]:
            draw.text((x1, y1 - 2), box["label"], fill=color, font=font, anchor="lb")
    image.save(out)


if __name__ == "__main__":
    draw_boxes(Path(sys.argv[1]), Path(sys.argv[2]))
