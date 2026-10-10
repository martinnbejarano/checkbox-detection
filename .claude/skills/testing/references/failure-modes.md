# Failure modes checklist

Scan this list when planning tests. For each mode ask: (1) can this code path trigger it? (2) do we own it, or is it
library behavior? (3) can I name the branch or parameter in our code? If any answer is no, skip it.

## Inputs (upload and decode)

| Mode | Look for |
|---|---|
| Oversized body, pixels or pages | size checks before decode / render |
| Wrong or spoofed format | byte sniffing, allow-list |
| Corrupt, truncated or encrypted file | `load()` / pdfium error branches |
| Colour mode surprises | gray stays 2-D, RGBA on white, 16-bit gray, CMYK |
| Orientation | EXIF transpose before detection |
| Missing field / empty file | request validation |

## Detection (stages)

| Mode | Look for |
|---|---|
| Scale | anything derived from image size; a box side or page width outside the samples' |
| Colour overlays | watermark / cell fill only removable when the page is RGB |
| Touching lines | line-removal kernel vs box side |
| Look-alike glyphs | "O", "D", "0", table cells |
| Empty page / text-only page | `[]`, no crash on zero candidates |
| Duplicates | NMS across nested contours and across tiles |
| Tile and page edges | boxes cut by a tile border or touching the page border |
| Ambiguous fill | values near the threshold resolve to unchecked |
| Stray strokes / handwriting | crossing a box must not make it checked |
| Coordinate mapping | upscale ×2, letterbox, tile offsets mapped back to page pixels |

## Labels

| Mode | Look for |
|---|---|
| Nothing to the right | `None`, not "" or garbage |
| Next box on the same row | strip stops before it |
| Box at the right page edge | strip clamped to the page |

## Runtime and errors

| Mode | Look for |
|---|---|
| Missing model file | startup fails with a clear message |
| Unknown stage name | startup `ValueError` listing choices |
| Concurrent requests | pdfium lock; shared ONNX sessions |
| Unexpected exception | 500 problem+json, no stack trace, logged with request id |

## Mapping to tests

Each selected mode becomes a row in the PR's "tests to write" table, or a row in "tests to skip" with the reason.
