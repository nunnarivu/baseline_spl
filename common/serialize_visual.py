'''
serialize_visual.py

Keyframe images for the Demo2Code-VLM variant.

This is the condition where the model reads the demonstration *visually* and receives no
privileged 3-D state — no mesh centroids, no grid cells, no direction labels. It is what
separates the VLM rung from the text rung of the evidence ladder, so anything derived
from ``meshes`` must stay out of this path.

Sizing matters here. Source frames are 720x1280 RGBA and demos run 4-37 keyframes, so a
concept can carry ~60 images across two demonstrations. Frames are downscaled to
``vlm_max_image_px`` on the long side and PNG-encoded.

Subsampling is deliberately NOT the default: in this domain every keyframe is a
placement, so dropping frames deletes construction steps and would cripple the baseline
for a reason unrelated to its capability. ``vlm_max_keyframes`` is a safety valve and
warns loudly when it engages.
'''

from __future__ import annotations

import warnings
from typing import List, Sequence

import cv2
import numpy as np


def encode_frame(rgb: np.ndarray, max_px: int = 512) -> bytes:
    '''One RGB(A) frame -> downscaled PNG bytes.'''
    image = np.asarray(rgb)
    if image.ndim != 3:
        raise ValueError(f"expected an HxWxC frame, got shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]
    scale = max_px / float(max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(width * scale))),
                                   max(1, int(round(height * scale)))),
                           interpolation=cv2.INTER_AREA)

    # cv2 encodes BGR; the dataloader already converted to RGB.
    ok, buffer = cv2.imencode(".png", image[:, :, ::-1])
    if not ok:
        raise RuntimeError("cv2.imencode failed on a keyframe")
    return buffer.tobytes()


def demo_frames(demo: dict, *, max_px: int = 512, max_keyframes: int = 40) -> List[bytes]:
    '''All keyframes of one demonstration as PNG bytes, in order.'''
    rgbs: Sequence = demo.get("rgbs") or []
    if not len(rgbs):
        raise ValueError(
            f"demo {demo.get('demo_id')} has no rgbs; build the dataloader with "
            f"load_images=True for the VLM variant"
        )

    frames = list(rgbs)
    if len(frames) > max_keyframes:
        warnings.warn(
            f"[serialize_visual] demo {demo.get('demo_id')} has {len(frames)} keyframes; "
            f"subsampling to {max_keyframes}. Every keyframe here is a placement, so this "
            f"DELETES construction steps and handicaps the baseline — raise "
            f"vlm_max_keyframes if you can afford the tokens.",
            stacklevel=2,
        )
        keep = np.linspace(0, len(frames) - 1, max_keyframes).round().astype(int)
        frames = [frames[i] for i in sorted(set(keep.tolist()))]

    return [encode_frame(f, max_px=max_px) for f in frames]


def demos_frames(demos: Sequence[dict], *, max_px: int = 512,
                 max_keyframes: int = 40) -> List[List[bytes]]:
    '''Per-demonstration keyframe images.'''
    return [demo_frames(d, max_px=max_px, max_keyframes=max_keyframes) for d in demos]
