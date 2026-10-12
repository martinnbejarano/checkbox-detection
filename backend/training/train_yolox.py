"""Fine-tune the `yolox` locator from the published `best_ckpt.pth`, export it to ONNX with the
same input and output, and pick `YoloxParams` on validation.

The model keeps its three classes (Text, ChoiceButton, Signature) so the ONNX output is unchanged;
every training box is a ChoiceButton, whatever its state. Training images are made the way
`YoloxLocator` makes its input: a tile whose side is a fraction of the page width, clean gray
(colour overlays removed) as 3 channels, letterboxed to 640 on grey 114.

    uv run python train_yolox.py train
    uv run python train_yolox.py tune
"""

import argparse
import copy
import itertools
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from common import (
    MODELS,
    PRETRAINED,
    RUNS,
    boxes_by_image,
    detection_counts,
    f2,
    load_rgb,
    split,
    val_pages,
)
from torch import nn
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import ModelEMA

from checkbox_detection.detection.binarize import to_clean_gray
from checkbox_detection.detection.locate.yolox import (
    CHOICE_BUTTON,
    INPUT,
    YoloxLocator,
    letterbox,
)
from checkbox_detection.detection.onnx import load_session

# YOLOX casts with tensor.type(x.type()), and PyTorch has no "torch.mps.FloatTensor" type string.
_tensor_type = torch.Tensor.type


def _mps_aware_type(self: torch.Tensor, dtype: Any = None, *args: Any, **kwargs: Any) -> Any:
    if dtype == "torch.mps.FloatTensor":
        return self.to("mps", torch.float32)
    return _tensor_type(self, dtype, *args, **kwargs)


torch.Tensor.type = _mps_aware_type  # type: ignore[method-assign]

PRETRAINED_ONNX = MODELS / "form-field-v1-nano.onnx"
OUT_ONNX = MODELS / "form-field-v1-nano-tuned.onnx"
CKPT_URL = "https://huggingface.co/nutrientdocs/form-field-v1-nano/resolve/7cbc1ecd55d59f6315a1415dddc59c3c8b115f05/best_ckpt.pth"

# Tile side as a share of the page width: around the service's 0.6, wide enough that the model
# also learns the box sizes a different tile_frac would show it.
TILE_FRACS = (0.4, 1.0)
MAX_LABELS = 300  # targets per tile; a 1004 page holds about 120 boxes
VAL_THREADS = 4  # onnxruntime threads for validation only (the service uses 1)
TILE_FRAC_GRID = (0.5, 0.6, 0.7, 0.8, 1.0)  # `tune`: YoloxParams candidates
SCORE_THRESHOLD_GRID = (0.1, 0.2, 0.3)
VAL_PAGES = 200  # validation pages scored per epoch to pick the checkpoint, and in `tune`


def nano() -> YOLOX:
    """YOLOX-Nano as in the published exp.py: depth 0.33, width 0.25, depthwise, SiLU, 3 classes."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(0.33, 0.25, in_channels=in_channels, act="silu", depthwise=True)
    head = YOLOXHead(3, 0.25, in_channels=in_channels, act="silu", depthwise=True)
    model = YOLOX(backbone, head)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    return model


class Tiles(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random tiles of the training pages, as the locator cuts them, with their boxes."""

    def __init__(self, length: int) -> None:
        self.coco = split("train")
        self.pages = list(zip(self.coco["images"], boxes_by_image(self.coco).values(), strict=True))
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(index * 7919 + torch.initial_seed())
        image, boxes = rng.choice(self.pages)
        gray = to_clean_gray(load_rgb(self.coco, image))
        h, w = gray.shape
        side = min(round(rng.uniform(*TILE_FRACS) * w), max(h, w))
        # Centre the tile on a random box half of the time, so dense regions are seen often.
        if boxes and rng.random() < 0.5:
            bx, by, bw, bh = rng.choice(boxes)["bbox"]
            x0 = round(bx + bw / 2 - rng.uniform(0.1, 0.9) * side)
            y0 = round(by + bh / 2 - rng.uniform(0.1, 0.9) * side)
        else:
            x0, y0 = rng.randint(0, max(0, w - side)), rng.randint(0, max(0, h - side))
        x0, y0 = min(max(0, x0), max(0, w - side)), min(max(0, y0), max(0, h - side))
        tile = gray[y0 : y0 + side, x0 : x0 + side]
        canvas, scale = letterbox(tile)
        size = (round(tile.shape[1] * scale), round(tile.shape[0] * scale))
        canvas = scan_augment(canvas, size, rng)

        targets = np.zeros((MAX_LABELS, 5), np.float32)
        n = 0
        for b in boxes:
            bx, by, bw, bh = b["bbox"]
            x1, y1 = max(bx, x0), max(by, y0)
            x2, y2 = min(bx + bw, x0 + tile.shape[1]), min(by + bh, y0 + tile.shape[0])
            # A box cut by the tile edge is labelled only when most of it is inside: the service
            # overlaps tiles so every box is whole in one of them.
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 0.6 * bw * bh or n == MAX_LABELS:
                continue
            cx, cy = ((x1 + x2) / 2 - x0) * scale, ((y1 + y2) / 2 - y0) * scale
            targets[n] = (CHOICE_BUTTON, cx, cy, (x2 - x1) * scale, (y2 - y1) * scale)
            n += 1
        x = torch.from_numpy(np.repeat(canvas[None], 3, 0).astype(np.float32))
        return x, torch.from_numpy(targets)


def scan_augment(canvas: np.ndarray, size: tuple[int, int], rng: random.Random) -> np.ndarray:
    """Contrast, blur and noise inside the image part of the letterbox (the pad stays 114)."""
    w, h = size
    part = canvas[:h, :w].astype(np.float32)
    part = (part - 255) * rng.uniform(0.6, 1.1) + 255 + rng.uniform(-20, 10)
    if rng.random() < 0.3:
        part = cv2.GaussianBlur(part, (0, 0), rng.uniform(0.3, 1.0))
    part += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0, 6), part.shape)
    canvas[:h, :w] = np.clip(part, 0, 255).astype(np.uint8)
    return canvas


def export(model: YOLOX, path: Path) -> None:
    """ONNX as published: input `images` 1x3x640x640, output `output` 1x8400x8, decode baked in."""
    model = model.cpu().eval()
    model.head.decode_in_inference = True
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, INPUT, INPUT),),
        str(path),
        input_names=["images"],
        output_names=["output"],
        opset_version=11,
        dynamo=False,
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(21)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = nano()
    ckpt = PRETRAINED / "best_ckpt.pth"
    if not ckpt.is_file():
        torch.hub.download_url_to_file(CKPT_URL, str(ckpt))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    model.to(device)
    ema = ModelEMA(model, 0.9998)

    loader = torch.utils.data.DataLoader(
        Tiles(args.tiles_per_epoch),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    pages = val_pages(VAL_PAGES)
    locator = YoloxLocator(MODELS, VAL_THREADS)
    RUNS.mkdir(exist_ok=True)

    def score(net: YOLOX) -> tuple[float, tuple[int, int, int]]:
        with tempfile.TemporaryDirectory() as tmp:
            export(net, Path(tmp) / "m.onnx")
            locator.session = load_session(Path(tmp) / "m.onnx", VAL_THREADS)
            counts = detection_counts(locator, pages)
        return f2(*counts), counts

    locator.session = load_session(PRETRAINED_ONNX, VAL_THREADS)
    base = detection_counts(locator, pages)
    print(f"pretrained: F2 {f2(*base):.4f} (truth, found, matched) {base}")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        start = perf_counter()
        for i, (x, targets) in enumerate(loader):
            out = model(x.to(device), targets.to(device))
            opt.zero_grad()
            out["total_loss"].backward()
            opt.step()
            sched.step()
            ema.update(model)
            if i % 50 == 0:
                loss = out["total_loss"].item()
                print(f"  epoch {epoch + 1} step {i}/{len(loader)} loss {loss:.3f}", flush=True)
        value, counts = score(copy.deepcopy(ema.ema))  # export moves it to the CPU
        took = perf_counter() - start
        print(f"epoch {epoch + 1}: {took:.0f}s, val F2 {value:.4f} {counts}", flush=True)
        if value > best:
            best = value
            torch.save({"model": ema.ema.state_dict()}, RUNS / "yolox_tuned.pth")
    model.load_state_dict(torch.load(RUNS / "yolox_tuned.pth", weights_only=True)["model"])
    export(model, OUT_ONNX)
    print(f"best val F2 {best:.4f}; wrote {OUT_ONNX}")


def tune(args: argparse.Namespace) -> None:
    """Validation detection metrics over a YoloxParams grid, for the pretrained and tuned models."""
    pages = val_pages(args.val_pages)
    for onnx_file in (PRETRAINED_ONNX, OUT_ONNX):
        locator = YoloxLocator(MODELS, VAL_THREADS)
        locator.session = load_session(onnx_file, VAL_THREADS)
        print(f"\n{onnx_file.name}: val detection (P / R / F2), {len(pages)} pages")
        print("| tile_frac | score_threshold | P | R | F2 |")
        print("|--:|--:|--:|--:|--:|")
        for tile, thr in itertools.product(TILE_FRAC_GRID, SCORE_THRESHOLD_GRID):
            locator.params = replace(locator.params, tile_frac=tile, score_threshold=thr)
            t, f, m = detection_counts(locator, pages)
            print(
                f"| {tile} | {thr} | {m / max(f, 1):.3f} | {m / max(t, 1):.3f} "
                f"| {f2(t, f, m):.3f} |",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "tune"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tiles-per-epoch", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--val-pages", type=int, default=VAL_PAGES)
    args = parser.parse_args()
    {"train": train, "tune": tune}[args.command](args)
