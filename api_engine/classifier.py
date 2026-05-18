# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Cart crop classification — Stage 1 (quality) + Stage 2 (fill/bag).

Handles cropping, transform, inference, and temperature-scaled softmax.
Supports batched inference for multiple carts in a single GPU pass.
"""
import cv2
import torch
from PIL import Image

from .config import CLS_TRANSFORM, BAG_CLASSES, EMPTY_OVERRIDE_THRESH
from .models import load_quality_checkpoint, load_fill_checkpoint

_EMPTY_RESULT_UNCLEAR = {
    "is_valid": False, "quality": "unclear",
    "fill": "non-applicable", "bag": "non-applicable",
    "quality_conf": 0.0, "fill_conf": 0.0, "bag_conf": 0.0,
}
_EMPTY_RESULT_TINY = dict(_EMPTY_RESULT_UNCLEAR)
_EMPTY_RESULT_NO_MODEL = {
    "is_valid": True, "quality": "unclassified",
    "fill": "unclassified", "bag": "unclassified",
    "quality_conf": 0.0, "fill_conf": 0.0, "bag_conf": 0.0,
}


class CartClassifier:
    """Stateful wrapper around quality + fill/bag models with lazy loading."""

    __slots__ = (
        "device",
        "_quality_model", "_quality_pt", "_quality_temp",
        "_fill_model", "_fill_pt", "_fill_temp", "_bag_temp",
        "_fill_classes", "_bag_classes", "_n_bag_model",
        "_quality_threshold",
    )

    def __init__(self, device: str):
        self.device = device
        self._quality_model = None
        self._quality_pt = None
        self._quality_temp = 1.0
        self._fill_model = None
        self._fill_pt = None
        self._fill_temp = 1.0
        self._bag_temp = 1.0
        self._fill_classes = ["empty", "partial", "full"]
        self._bag_classes = list(BAG_CLASSES)
        self._n_bag_model = len(BAG_CLASSES)
        self._quality_threshold = 0.75

    @property
    def quality_pt(self):
        return self._quality_pt

    @property
    def fill_pt(self):
        return self._fill_pt

    @property
    def has_quality_model(self):
        return self._quality_model is not None

    def set_quality_threshold(self, v: float):
        self._quality_threshold = v

    def load_quality(self, pt_path: str):
        if pt_path == self._quality_pt:
            return
        print(f"[INFO] Loading quality model: {pt_path}")
        self._quality_model, self._quality_temp = load_quality_checkpoint(pt_path, self.device)
        self._quality_pt = pt_path

    def load_fill(self, pt_path: str):
        if pt_path == self._fill_pt:
            return
        print(f"[INFO] Loading fill/bag model: {pt_path}")
        (self._fill_model, self._fill_classes, self._bag_classes,
         self._n_bag_model, self._fill_temp, self._bag_temp) = load_fill_checkpoint(pt_path, self.device)
        self._fill_pt = pt_path

    def _crop_and_transform(self, frame_bgr, bbox):
        """Crop cart from frame and return transformed tensor, or None if too small."""
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return CLS_TRANSFORM(pil_img)

    @torch.no_grad()
    def classify_batch(self, frame_bgr, bboxes: list, cart_ids: list) -> dict:
        """Classify multiple cart crops in a single batched GPU pass.

        Args:
            frame_bgr: Full frame (BGR numpy array).
            bboxes: List of (x1, y1, x2, y2) for each cart.
            cart_ids: List of display IDs corresponding to each bbox.

        Returns:
            Dict mapping cart_id -> classification result dict.
        """
        if not bboxes or self._quality_model is None:
            fallback = _EMPTY_RESULT_NO_MODEL if self._quality_model is None else _EMPTY_RESULT_TINY
            return {cid: fallback for cid in cart_ids}

        # Prepare all crops on CPU. Carts whose crop is invalid (tiny / OOB)
        # are recorded as _EMPTY_RESULT_TINY in the same pass.
        tensors = []
        valid_indices = []
        results: dict = {}
        for i, bbox in enumerate(bboxes):
            t = self._crop_and_transform(frame_bgr, bbox)
            if t is None:
                results[cart_ids[i]] = _EMPTY_RESULT_TINY
            else:
                tensors.append(t)
                valid_indices.append(i)

        if not tensors:
            return results

        # Single GPU transfer for entire batch
        batch = torch.stack(tensors).to(self.device)

        # Stage 1: Quality (batched)
        logits_q = self._quality_model(batch)
        probs_q = torch.softmax(logits_q / self._quality_temp, dim=1)

        valid_mask = probs_q[:, 0] >= self._quality_threshold
        fill_indices = []

        for j in range(len(tensors)):
            cid = cart_ids[valid_indices[j]]
            if not valid_mask[j]:
                results[cid] = {
                    "is_valid": False, "quality": "unclear",
                    "fill": "non-applicable", "bag": "non-applicable",
                    "quality_conf": float(probs_q[j, 1]),
                    "fill_conf": 0.0, "bag_conf": 0.0,
                }
            else:
                fill_indices.append(j)

        # Stage 2: Fill + Bag (batched, only valid carts)
        if not fill_indices or self._fill_model is None:
            for j in fill_indices:
                cid = cart_ids[valid_indices[j]]
                results[cid] = {
                    "is_valid": True, "quality": "valid_cart",
                    "fill": "unclassified", "bag": "unclassified",
                    "quality_conf": float(probs_q[j, 0]),
                    "fill_conf": 0.0, "bag_conf": 0.0,
                }
            return results

        fill_batch = batch[fill_indices]
        logits_fill, logits_bag = self._fill_model(fill_batch)
        fill_probs = torch.softmax(logits_fill / self._fill_temp, dim=1)
        bag_probs = torch.softmax(logits_bag / self._bag_temp, dim=1)

        empty_idx = self._fill_classes.index("empty") if "empty" in self._fill_classes else 0

        for k, j in enumerate(fill_indices):
            cid = cart_ids[valid_indices[j]]
            fp = fill_probs[k]
            bp = bag_probs[k]
            fill_idx = int(fp.argmax())
            bag_idx = int(bp.argmax())

            if float(fp[empty_idx]) >= EMPTY_OVERRIDE_THRESH:
                fill_idx = empty_idx

            fill_lbl = self._fill_classes[fill_idx]

            if fill_idx == empty_idx:
                bag_lbl = "not_applicable"
                bag_conf = 1.0
            else:
                bag_lbl = self._bag_classes[bag_idx] if bag_idx < len(self._bag_classes) else "unknown"
                bag_conf = float(bp[bag_idx])

            results[cid] = {
                "is_valid": True, "quality": "valid_cart",
                "fill": fill_lbl, "bag": bag_lbl,
                "quality_conf": float(probs_q[j, 0]),
                "fill_conf": float(fp[fill_idx]),
                "bag_conf": bag_conf,
            }

        return results
