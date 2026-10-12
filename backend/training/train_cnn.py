"""Fine-tune the `cnn` state classifier from its published `state_cnn.pt`, export it to ONNX
with the same input and output, and pick `StateCnnParams` on validation.

Crops are made with the service's own `StateCnnClassifier._crop`, so training sees exactly what the
service feeds the model. Only boxes whose source state is trusted are used (data.py).

    uv run python train_cnn.py
"""

import argparse
import random
from dataclasses import replace

import cv2
import numpy as np
import onnxruntime as ort
import torch
from common import MODELS, PRETRAINED, RUNS, boxes_by_image, load_rgb, split
from torch import nn

from checkbox_detection.detection.binarize import to_clean_gray
from checkbox_detection.detection.classify.state_cnn import (
    HEIGHT,
    MEAN,
    STD,
    WIDTH,
    StateCnnClassifier,
)
from checkbox_detection.detection.types import Rect

PRETRAINED_ONNX = MODELS / "form-field-v1-state.onnx"
OUT_ONNX = MODELS / "form-field-v1-state-tuned.onnx"
STATE_URL = "https://huggingface.co/nutrientdocs/form-field-v1-state/resolve/62b1049a066d63eacf0f3542dc155b11c7284883/state_cnn.pt"

# A locator's box is a few px off; train crops move each edge by up to this share of the side.
EDGE_JITTER = 0.1
CROPS_PER_BOX = 3  # jittered copies per training box
PAD_FRACS = (0.0, 0.05, 0.1, 0.15)  # StateCnnParams.pad_frac candidates; training draws from them
ERASED_INK = (
    0.03  # interior ink share: below it in clean gray but above it in plain gray = erased mark
)
SPECK_SHARE = 0.5  # empty training crops that get 1-3 specks of dust or toner
THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)  # StateCnnParams.threshold candidates


def state_cnn() -> nn.Sequential:
    """The published 96k-param network, rebuilt from the layer names and shapes of `state_cnn.pt`
    and the ONNX graph (depthwise-separable blocks, global average pool, linear head)."""

    def block(cin: int, cout: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )

    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        block(32, 64, 2),
        block(64, 128, 2),
        block(128, 128, 1),
        block(128, 192, 2),
        block(192, 192, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),  # index 10: no weights; the ONNX graph has none (inference)
        nn.Linear(192, 2),
    )


def crops(
    name: str, per_box: int, rng: random.Random, pad: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(crops uint8 [n, 40, 160], labels [n]) for every trusted-state box of a split.

    With several crops per box each one moves the box edges a little; the pad is `pad`, or one of
    PAD_FRACS at random per crop.

    A checked box whose mark the clean gray erases (a colour tick, as `to_clean_gray` removes
    colour overlays) is left out: the model would see an empty box labelled checked. About 21% of
    the checked training boxes were like that, and the model learned to call empty boxes with a
    thick border checked.
    """
    cropper = StateCnnClassifier(MODELS, 1)
    coco = split(name)
    xs, ys = [], []
    for image, boxes in zip(coco["images"], boxes_by_image(coco).values(), strict=True):
        boxes = [b for b in boxes if b["checked"] is not None]
        if not boxes:
            continue
        rgb = load_rgb(coco, image)
        gray = to_clean_gray(rgb)
        plain = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for b in boxes:
            x, y, w, h = b["bbox"]
            if b["checked"] and interior_ink(gray, b["bbox"]) < ERASED_INK < interior_ink(
                plain, b["bbox"]
            ):
                continue
            for _ in range(per_box):
                j = EDGE_JITTER if per_box > 1 else 0.0
                rect = Rect(
                    round(x + rng.uniform(-j, j) * w),
                    round(y + rng.uniform(-j, j) * h),
                    round(x + w + rng.uniform(-j, j) * w),
                    round(y + h + rng.uniform(-j, j) * h),
                    1.0,
                )
                pad_frac = rng.choice(PAD_FRACS) if pad is None else pad
                cropper.params = replace(cropper.params, pad_frac=pad_frac)
                xs.append(cropper._crop(gray, rect))
                ys.append(int(b["checked"]))
    return np.stack(xs), np.array(ys)


def interior_ink(gray: np.ndarray, bbox: list[float]) -> float:
    """Share of dark pixels inside the box, a fifth of the side in from each edge."""
    x, y, w, h = bbox
    m = round(0.2 * min(w, h))
    inside = gray[round(y) + m : round(y + h) - m, round(x) + m : round(x + w) - m]
    return float(np.mean(inside < 128)) if inside.size else 0.0


def to_input(batch: np.ndarray) -> torch.Tensor:
    return (torch.from_numpy(batch).float()[:, None] / 255 - MEAN) / STD


def augment(batch: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scan-like changes per crop: contrast, brightness, noise, and specks in empty boxes.

    Without specks the empty training boxes are all clean, and the model learns that any ink
    inside means checked: a 2 px dot scored P(filled) 0.92. A speck is not an intentional mark
    (labeling policy), so it stays unchecked.
    """
    batch = batch.copy()
    for i in np.nonzero((labels == 0) & (rng.random(len(batch)) < SPECK_SHARE))[0]:
        for _ in range(rng.integers(1, 5)):
            # The box sits in the middle 40x40 of the 40x160 crop; a speck is round or square,
            # up to 7 px across (about 15% of the box side).
            x, y, r = int(rng.integers(64, 97)), int(rng.integers(4, 37)), int(rng.integers(0, 4))
            ink = int(rng.integers(0, 120))
            if rng.random() < 0.5:
                cv2.circle(batch[i], (x, y), r, ink, -1)
            else:
                cv2.rectangle(batch[i], (x - r, y - r), (x + r, y + r), ink, -1)
    out = batch.astype(np.float32)
    out = (out - 255) * rng.uniform(0.6, 1.2, (len(out), 1, 1)) + 255  # fainter or darker ink
    noise = rng.uniform(0, 6, (len(out), 1, 1))  # some crops stay clean, as on a born-digital page
    out += rng.uniform(-25, 10, (len(out), 1, 1)) + rng.normal(0, 1, out.shape) * noise
    return np.clip(out, 0, 255).astype(np.uint8)


def p_filled(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = torch.cat(
            [model(to_input(x[i : i + 1024]).to(device)).cpu() for i in range(0, len(x), 1024)]
        )
    return torch.softmax(logits, 1)[:, 1].numpy()


def balanced_accuracy(p: np.ndarray, y: np.ndarray, threshold: float) -> float:
    pred = p > threshold
    return float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1)]))


def train(args: argparse.Namespace) -> None:
    rng = random.Random(21)
    torch.manual_seed(21)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x_train, y_train = crops("train", CROPS_PER_BOX, rng)
    x_val, y_val = crops("val", 1, rng)
    print(f"crops: train {len(x_train)} ({y_train.mean():.2f} checked), val {len(x_val)}")

    model = state_cnn()
    weights = PRETRAINED / "state_cnn.pt"
    if not weights.is_file():
        torch.hub.download_url_to_file(STATE_URL, str(weights))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    check_matches_onnx(model, x_val[:256])
    model.to(device)
    acc = balanced_accuracy(p_filled(model, x_val, device), y_val, 0.5)
    print(f"pretrained: val balanced acc {acc:.4f}")

    # Classes are weighted so the rarer one counts as much.
    weight = torch.tensor(
        len(y_train) / (2 * np.bincount(y_train, minlength=2)), dtype=torch.float32
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(x_train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    nrng = np.random.default_rng(21)
    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        order = nrng.permutation(len(x_train))
        for i in range(len(x_train) // args.batch):
            idx = order[i * args.batch : (i + 1) * args.batch]
            x = to_input(augment(x_train[idx], y_train[idx], nrng)).to(device)
            loss = loss_fn(model(x), torch.from_numpy(y_train[idx]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = balanced_accuracy(p_filled(model, x_val, device), y_val, 0.5)
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}, val balanced acc {acc:.4f}")
        if acc > best:
            best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    RUNS.mkdir(exist_ok=True)
    torch.save(best_state, RUNS / "state_cnn_tuned.pt")
    export(model.cpu())
    print(f"best val balanced acc {best:.4f}; wrote {OUT_ONNX}")


def check_matches_onnx(model: nn.Module, x: np.ndarray) -> None:
    """The rebuilt network with the published weights gives the published ONNX's logits."""
    model.eval()
    with torch.no_grad():
        ours = model(to_input(x)).numpy()
    session = ort.InferenceSession(str(PRETRAINED_ONNX), providers=["CPUExecutionProvider"])
    theirs = session.run(None, {"crop": to_input(x).numpy()})[0]
    assert np.allclose(ours, theirs, atol=1e-3), np.abs(ours - theirs).max()


def export(model: nn.Module) -> None:
    model.eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, HEIGHT, WIDTH),),
        str(OUT_ONNX),
        input_names=["crop"],
        output_names=["logits"],
        dynamic_axes={"crop": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )


def tune(_: argparse.Namespace) -> None:
    """Validation accuracy of each (pad_frac, threshold) for the pretrained and the tuned model."""
    rng = random.Random(21)
    for onnx_file in (PRETRAINED_ONNX, OUT_ONNX):
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        print(f"\n{onnx_file.name}: val accuracy (checked recall / unchecked recall)")
        print("| pad_frac | " + " | ".join(f"{t:.2f}" for t in THRESHOLDS) + " |")
        print("|---|" + "--:|" * len(THRESHOLDS))
        for pad in PAD_FRACS:
            x, y = crops("val", 1, rng, pad)  # at the true boxes
            logits = session.run(None, {"crop": to_input(x).numpy()})[0]
            p = 1 / (1 + np.exp(logits[:, 0] - logits[:, 1]))
            cells = [
                f"{np.mean((p > t) == y):.4f} "
                f"({np.mean(p[y == 1] > t):.3f}/{np.mean(p[y == 0] <= t):.3f})"
                for t in THRESHOLDS
            ]
            print(f"| {pad} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train", "tune"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    {"train": train, "tune": tune}[args.command](args)
