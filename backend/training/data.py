"""Build the train / validation data for tuning.

Two kinds of pages, written as one COCO dataset per split under data/build/:

- Real scans from Roboflow (CC BY 4.0), classes remapped to checked / unchecked.
- Synthetic pages from the blank 1004 / 1004MC / 1004C PDFs: every checkbox is a vector square, so
  the bbox is exact; boxes are marked at random and the page is degraded like a scan.

The split is by source form: a validation page never shares a layout with a training page.
Never reads samples/, backend/tests/data/ or Roboflow `annotation-mmvj7/new_merge_project` (the
external evaluation set).

    ROBOFLOW_API_KEY=... uv run python data.py   # the key is only needed for a first download
"""

import ctypes
import json
import os
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from checkbox_detection.detection.binarize import binarize, to_clean_gray

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW, FORMS, BUILD = DATA / "raw", DATA / "forms", DATA / "build"
SEED = 21


@dataclass(frozen=True)
class Source:
    """A Roboflow export: fork id (workspace/project/version), class remap, trusted state."""

    fork: str
    remap: dict[str, bool]  # class name -> is_checked; other classes drop the image
    state: bool  # False: its state labels were audited as wrong, so only its boxes are used


# Forks of the public Universe projects in the ticket (their published versions are stretched to
# a square; a fork exports each page at its own size). Licences: all CC BY 4.0.
SOURCES = {
    # universe.roboflow.com/cogniquest-idc1w/checkboxes-xtv3p. Its "checked" class is mostly
    # empty boxes on a contact sheet of 24 random crops, so only its boxes are used.
    "checkboxes-xtv3p": Source(
        "martin-bejarano-s-workspace/checkboxes-xtv3p-y8ofu/1",
        {"checked": True, "unchecked": False},
        state=False,
    ),
    # universe.roboflow.com/myproject-gemsb/checkboxes-2-dyp1n. Six class names from merged
    # uploads; "check" crops are ticked boxes, so it maps to checked like the other three.
    "checkboxes-2-dyp1n": Source(
        "martin-bejarano-s-workspace/checkboxes-2-dyp1n-fl1sp/1",
        {
            "Checkbox-Checked": True,
            "Checked": True,
            "checked": True,
            "check": True,
            "Checkbox-UnChecked": False,
            "unchecked": False,
        },
        state=True,
    ),
    # universe.roboflow.com/tempqqaq/checkbox_v2
    "checkbox_v2": Source(
        "martin-bejarano-s-workspace/checkbox_v2-ti1fj/1",
        {"checked": True, "unchecked": False},
        state=True,
    ),
    # universe.roboflow.com/ujiboo/checkbox-detector-zxwoh (web screenshots dropped by colour)
    "checkbox-detector-zxwoh": Source(
        "martin-bejarano-s-workspace/checkbox-detector-zxwoh-vfhzg/1",
        {"checked": True, "unchecked": False},
        state=True,
    ),
    # universe.roboflow.com/s-workspace-dte1y/government-document-omr-checkbox (handwritten ticks)
    "government-document-omr-checkbox": Source(
        "martin-bejarano-s-workspace/government-document-omr-checkbox-js1fr/1",
        {"checked": True, "unchecked": False},
        state=True,
    ),
    # Not used, after looking at their crops: chris-bromley-bf2yc/tax-form-checkbox-detection
    # labels round radio buttons together with their word (out of scope), and
    # office-8ecbg/checkbox-detection-jiid0 holds 10x10 px crops with no page around them.
}

# Blank forms. Fannie Mae's site answers scripts with a Cloudflare challenge, so the 1004MC is
# saved by hand from a browser; Freddie Mac's copies of the 1004 (form 70) and 1004C (70B) download.
FORM_URLS = {
    "1004": "https://sf.freddiemac.com/docs/pdf/forms/70.pdf",
    "1004c": "https://sf.freddiemac.com/docs/pdf/forms/70B.pdf",
    "1004mc": "https://singlefamily.fanniemae.com/media/14061/display",
}
VAL_FORM = "1004c"  # the held-out form for the synthetic pages

# Real scans: pages whose layouts look alike (same form, rescans, duplicates) are one group, and
# whole groups go to validation, so near-copies never sit on both sides.
VAL_GROUP_SHARE = 0.15
HASH_SIDE = 16  # thumbnail side for the layout hash
SAME_LAYOUT_BITS = 20  # hashes this close (of 256 bits) are one layout

# Filters on real scans (an image failing any of them is dropped whole, so no box goes unlabelled).
MIN_SIDE_PX = 8  # smaller boxes cannot be read; matches OpenCVParams.min_side_px
ASPECT = (0.6, 1.6)  # wider regions are a box plus its label, or several boxes
MAX_COLOUR_SHARE = 0.1  # web screenshots are colourful; scans with blue pen are not
MAX_LONG_SIDE = 4000  # larger pages are shrunk, to keep the build small
# Snapping a loose annotation to its printed square (see snap()).
SNAP_PAD = 0.15  # search this share of the side past the annotation
SNAP_SIDE = (0.55, 1.1)  # the square's side, as a share of the annotation's
SNAP_MAX_OFFSET = 0.25  # how far its centre may sit from the annotation's, in sides

# Synthetic pages.
VARIANTS_PER_PAGE = 10
DPIS = (100, 150, 200, 300)
P_CHECKED = 0.4
P_STRAY = 0.08  # share of empty boxes crossed by a stray stroke (still unchecked)


Page = tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]


def main() -> None:
    rng = random.Random(SEED)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    pdfs = {form: form_pdf(form) for form in FORM_URLS}  # fail before the long real-scan pass
    writers = {split: CocoWriter(split) for split in ("train", "val")}
    real = [scan for name in SOURCES for scan in real_scans(name)]
    val = pick_val_groups([s.layout for s in real], rng)
    for i, scan in enumerate(real):
        writers["val" if i in val else "train"].add(*scan.load())
    for form, pdf in pdfs.items():
        for page in synthetic_pages(form, pdf, rng):
            writers["val" if form == VAL_FORM else "train"].add(*page)
    for writer in writers.values():
        writer.close()


def fetch_roboflow(name: str) -> Path:
    """The export folder of `name`, downloaded once with ROBOFLOW_API_KEY."""
    out = RAW / name
    if out.is_dir():
        return out
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY to download the Roboflow sets")
    url = f"https://api.roboflow.com/{SOURCES[name].fork}/coco?api_key={key}"
    with urllib.request.urlopen(url) as r:
        link = json.load(r)["export"]["link"]
    archive = DATA / f"{name}.zip"
    urllib.request.urlretrieve(link, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    archive.unlink()
    return out


@dataclass(frozen=True)
class Scan:
    """A real page that passed the filters; its pixels are read again only when written."""

    path: Path
    meta: dict[str, Any]
    boxes: list[dict[str, Any]]
    layout: np.ndarray

    def load(self) -> Page:
        image = cv2.imread(str(self.path))
        scale = min(1.0, MAX_LONG_SIDE / max(image.shape[:2]))
        if scale == 1:
            return image, self.meta, self.boxes
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes = [{**b, "bbox": [v * scale for v in b["bbox"]]} for b in self.boxes]
        return image, self.meta, boxes


def real_scans(name: str) -> list[Scan]:
    source = SOURCES[name]
    scans = []
    for coco_file in sorted(fetch_roboflow(name).glob("*/_annotations.coco.json")):
        coco = json.loads(coco_file.read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        for info in coco["images"]:
            anns = by_image[info["id"]]
            if not anns or any(names[a["category_id"]] not in source.remap for a in anns):
                continue
            if not all(usable_box(a["bbox"]) for a in anns):
                continue
            path = coco_file.parent / info["file_name"]
            image = cv2.imread(str(path))
            if image is None or colour_share(image) > MAX_COLOUR_SHARE:
                continue
            gray = to_clean_gray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes = [
                {
                    "bbox": snap(gray, a["bbox"]),
                    "checked": source.remap[names[a["category_id"]]] if source.state else None,
                }
                for a in anns
            ]
            meta = {"source": name, "file": info["file_name"]}
            scans.append(Scan(path, meta, boxes, layout_hash(image)))
    print(f"{name}: {len(scans)} pages")
    return scans


def snap(gray: np.ndarray, bbox: list[float]) -> list[float]:
    """The printed square inside a Roboflow annotation, which is drawn loose: the printed square
    fills 75-90% of its side (median per source), against 95% for the exact synthetic boxes.
    A model trained on loose boxes returns loose boxes, and small boxes then miss IoU 0.5.

    The square is the near-square ink outline whose centre is closest to the annotation's,
    with a side of SNAP_SIDE of the annotation's. Without one (a box touching a rule, a faint
    glyph) the annotation is kept.
    """
    x, y, w, h = bbox
    pad = round(SNAP_PAD * max(w, h))
    x0, y0 = max(0, round(x) - pad), max(0, round(y) - pad)
    crop = gray[y0 : round(y + h) + pad, x0 : round(x + w) + pad]
    if min(crop.shape) < MIN_SIDE_PX:
        return bbox
    contours, _ = cv2.findContours(binarize(crop), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, list[float]] | None = None
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if not (
            SNAP_SIDE[0] * w <= cw <= SNAP_SIDE[1] * w
            and SNAP_SIDE[0] * h <= ch <= SNAP_SIDE[1] * h
        ):
            continue
        if not ASPECT[0] <= ch / cw <= ASPECT[1]:
            continue
        off = abs(x0 + cx + cw / 2 - (x + w / 2)) + abs(y0 + cy + ch / 2 - (y + h / 2))
        if off <= SNAP_MAX_OFFSET * max(w, h) and (best is None or off < best[0]):
            best = (off, [x0 + cx, y0 + cy, cw, ch])
    return best[1] if best else bbox


def usable_box(bbox: list[float]) -> bool:
    _, _, w, h = bbox
    return min(w, h) >= MIN_SIDE_PX and ASPECT[0] <= h / w <= ASPECT[1]


def colour_share(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 1] > 80) & (hsv[..., 2] > 80)))


def layout_hash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIDE, HASH_SIDE), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).ravel()


def pick_val_groups(layouts: list[np.ndarray], rng: random.Random) -> set[int]:
    """Indices of the pages that go to validation: whole layout groups, about VAL_GROUP_SHARE."""
    hashes = np.array(layouts)
    parent = list(range(len(hashes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        close = np.nonzero((hashes[i + 1 :] != hashes[i]).sum(1) <= SAME_LAYOUT_BITS)[0]
        for j in close + i + 1:
            parent[root(int(j))] = root(i)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(hashes)):
        groups[root(i)].append(i)
    order = sorted(groups)
    rng.shuffle(order)
    val: set[int] = set()
    for g in order:
        if len(val) >= VAL_GROUP_SHARE * len(hashes):
            break
        val.update(groups[g])
    print(f"real scans: {len(hashes)} pages in {len(groups)} layout groups, {len(val)} to val")
    return val


def form_pdf(form: str) -> Path:
    path = FORMS / f"{form}.pdf"
    if not path.is_file():
        FORMS.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(FORM_URLS[form], path)
        except OSError as e:
            raise SystemExit(f"Save {FORM_URLS[form]} from a browser as {path} ({e})") from e
    return path


def form_boxes(page: pdfium.PdfPage) -> list[tuple[float, float, float, float]]:
    """Checkboxes in PDF points (left, bottom, right, top): stroked near-squares of 6-12 pt.

    Filled squares of the same size are the black backgrounds of the sidebar letters.
    """
    boxes = []
    for obj in page.get_objects(max_depth=5):
        if obj.type != pdfium_raw.FPDF_PAGEOBJ_PATH:
            continue
        left, bottom, right, top = obj.get_bounds()
        w, h = right - left, top - bottom
        fill, stroke = ctypes.c_int(), ctypes.c_int()
        pdfium_raw.FPDFPath_GetDrawMode(obj.raw, fill, stroke)
        if 6 < w < 12 and 6 < h < 12 and 0.8 < h / w < 1.25 and stroke.value and not fill.value:
            boxes.append((left, bottom, right, top))
    return boxes


def synthetic_pages(form: str, path: Path, rng: random.Random) -> Iterator[Page]:
    for index, page in enumerate(pdfium.PdfDocument(path)):
        boxes_pt = form_boxes(page)
        if not boxes_pt:
            continue
        height_pt = page.get_size()[1]
        for variant in range(VARIANTS_PER_PAGE):
            dpi = rng.choice(DPIS)
            k = dpi / 72
            image = np.array(page.render(scale=k).to_pil().convert("RGB"))[..., ::-1]
            image = np.ascontiguousarray(image)
            boxes = []
            for left, bottom, right, top in boxes_pt:
                x1, y1, x2, y2 = (
                    left * k,
                    (height_pt - top) * k,
                    right * k,
                    (height_pt - bottom) * k,
                )
                checked = rng.random() < P_CHECKED
                if checked:
                    draw_mark(image, (x1, y1, x2, y2), rng)
                elif rng.random() < P_STRAY:
                    draw_stray(image, (x1, y1, x2, y2), rng)
                boxes.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "checked": checked})
            image, boxes = degrade(image, boxes, rng)
            meta = {"source": f"synthetic-{form}", "file": f"{form}-p{index}-v{variant}-{dpi}dpi"}
            yield image, meta, boxes


# BGR: black, dark blue ink, pencil, and a bright blue-violet pen as on scanned housing forms,
# saturated enough that `to_clean_gray` erases it and the locator sees only its dark edges.
PEN_COLOURS = [(20, 20, 20), (140, 40, 20), (60, 60, 60), (190, 60, 70)]


def jitter(points: list[tuple[float, float]], amount: float, rng: random.Random) -> np.ndarray:
    return np.array(
        [(x + rng.gauss(0, amount), y + rng.gauss(0, amount)) for x, y in points], np.int32
    )


def draw_mark(image: np.ndarray, box: tuple[float, ...], rng: random.Random) -> None:
    """An intentional mark: X, tick, full fill, partial fill or scribble mostly inside the box,
    or a big tick or X that spills 1.5 to 3 sides past it, as a hand marks a small box."""
    x1, y1, x2, y2 = box
    s = x2 - x1
    colour = rng.choice(PEN_COLOURS)
    t = max(1, round(s * rng.uniform(0.06, 0.16)))
    m = s * rng.uniform(0.1, 0.25)  # inset of the mark
    a, b, c, d = x1 + m, x2 - m, y1 + m, y2 - m
    kind = rng.choice(
        ["x", "x", "check", "check", "fill", "partial", "scribble", "big_check", "big_x"]
    )
    wob = s * 0.04
    if kind in ("big_check", "big_x"):
        k = rng.uniform(1.5, 3.0)
        t = max(1, round(s * rng.uniform(0.1, 0.25)))
        if kind == "big_x":
            cx, cy = (
                (x1 + x2) / 2 + rng.uniform(-0.3, 0.3) * s,
                (y1 + y2) / 2 + rng.uniform(-0.3, 0.3) * s,
            )
            for dx in (1, -1):
                pts = [(cx - k * s / 2, cy - dx * k * s / 2), (cx + k * s / 2, cy + dx * k * s / 2)]
                cv2.polylines(image, [jitter(pts, wob * k, rng)], False, colour, t, cv2.LINE_AA)
        else:
            # The vertex sits on the box's lower edge; the long arm runs up and right past it.
            vx, vy = x1 + rng.uniform(0.2, 0.8) * s, y2 + rng.uniform(-0.3, 0.4) * s
            pts = [(vx - 0.3 * k * s, vy - 0.35 * k * s), (vx, vy), (vx + 0.5 * k * s, vy - k * s)]
            cv2.polylines(image, [jitter(pts, wob * k, rng)], False, colour, t, cv2.LINE_AA)
    elif kind == "x":
        cv2.polylines(image, [jitter([(a, c), (b, d)], wob, rng)], False, colour, t, cv2.LINE_AA)
        cv2.polylines(image, [jitter([(b, c), (a, d)], wob, rng)], False, colour, t, cv2.LINE_AA)
    elif kind == "check":
        mid = (a + (b - a) * 0.35, d)
        # A hand tick often overshoots the box top-right; most of its ink stays inside.
        end = (b + s * rng.uniform(0, 0.4), c - s * rng.uniform(0, 0.4))
        pts = jitter([(a, (c + d) / 2), mid, end], wob, rng)
        cv2.polylines(image, [pts], False, colour, t, cv2.LINE_AA)
    elif kind == "fill":
        cv2.rectangle(image, (round(a), round(c)), (round(b), round(d)), colour, -1)
    elif kind == "partial":
        frac = rng.uniform(0.4, 0.8)
        cv2.rectangle(
            image, (round(a), round(c)), (round(a + (b - a) * frac), round(d)), colour, -1
        )
    else:
        pts = [(rng.uniform(a, b), rng.uniform(c, d)) for _ in range(rng.randint(4, 8))]
        cv2.polylines(image, [jitter(pts, wob, rng)], False, colour, t, cv2.LINE_AA)


def draw_stray(image: np.ndarray, box: tuple[float, ...], rng: random.Random) -> None:
    """A long stroke that only crosses the box (labeling policy: unchecked)."""
    x1, y1, x2, y2 = box
    s = x2 - x1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    angle = rng.uniform(0, np.pi)
    reach = s * rng.uniform(2.5, 5)
    dx, dy = np.cos(angle) * reach, np.sin(angle) * reach
    t = max(1, round(s * rng.uniform(0.05, 0.12)))
    pts = jitter([(cx - dx, cy - dy), (cx + dx, cy + dy)], s * 0.1, rng)
    cv2.polylines(image, [pts], False, rng.choice(PEN_COLOURS), t, cv2.LINE_AA)


def degrade(
    image: np.ndarray, boxes: list[dict[str, Any]], rng: random.Random
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Look like a scan: colour watermark, slight rotation, blur, noise, JPEG."""
    h, w = image.shape[:2]
    if rng.random() < 0.3:  # a diagonal colour watermark, as on the 1004C sample
        overlay = np.full_like(image, 255)
        size = w / 150  # the word spans most of the page, as on the sample
        colour = rng.choice([(60, 60, 220), (220, 120, 60), (60, 160, 60)])
        cv2.putText(
            overlay,
            "COPY",
            (0, h * 2 // 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            colour,
            max(2, round(size * 3)),
        )
        overlay = cv2.warpAffine(
            overlay,
            cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1),
            (w, h),
            borderValue=(255, 255, 255),
        )
        image = np.minimum(image, cv2.addWeighted(overlay, 0.5, np.full_like(image, 255), 0.5, 0))
    angle = rng.uniform(-1, 1)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, rot, (w, h), borderValue=(255, 255, 255))
    rotated = []
    for b in boxes:
        x, y, bw, bh = b["bbox"]
        corners = np.array([[x, y, 1], [x + bw, y, 1], [x, y + bh, 1], [x + bw, y + bh, 1]]) @ rot.T
        (nx1, ny1), (nx2, ny2) = corners.min(0), corners.max(0)
        rotated.append({**b, "bbox": [nx1, ny1, nx2 - nx1, ny2 - ny1]})
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.0))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0, 8), image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    quality = rng.randint(40, 95)
    image = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], 1)
    return image, rotated


class CocoWriter:
    """data/build/{split}/*.jpg and data/build/annotations/instances_{split}.json.

    Categories are the pretrained YOLOX classes; every box is ChoiceButton (1). The state is kept
    per box as `checked` (None when the source's state is not trusted).
    """

    def __init__(self, split: str) -> None:
        self.split = split
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        (BUILD / split).mkdir(parents=True)

    def add(self, image: np.ndarray, meta: dict[str, Any], boxes: list[dict[str, Any]]) -> None:
        i = len(self.images)
        name = f"{i:05d}.jpg"
        cv2.imwrite(str(BUILD / self.split / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = image.shape[:2]
        self.images.append({"id": i, "file_name": name, "width": w, "height": h, **meta})
        for b in boxes:
            x, y, bw, bh = (round(float(v), 1) for v in b["bbox"])
            self.annotations.append(
                {
                    "id": len(self.annotations),
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "checked": b["checked"],
                }
            )

    def close(self) -> None:
        names = ["Text", "ChoiceButton", "Signature"]
        coco = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
        }
        (BUILD / "annotations").mkdir(exist_ok=True)
        (BUILD / "annotations" / f"instances_{self.split}.json").write_text(json.dumps(coco))
        print(f"{self.split}: {len(self.images)} pages, {len(self.annotations)} boxes")


if __name__ == "__main__":
    main()
