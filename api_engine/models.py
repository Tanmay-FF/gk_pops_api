# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Neural network definitions and checkpoint loading utilities.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import BAG_CLASSES
from logging_config import get_logger

log = get_logger("pops_api.models")


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
_BACKBONES = {
    "mobilenet_v3_large":  (lambda: models.mobilenet_v3_large(weights=None),  960),
    "mobilenet_v3_small":  (lambda: models.mobilenet_v3_small(weights=None),  576),
    "efficientnet_b0":     (lambda: models.efficientnet_b0(weights=None),    1280),
    "convnext_tiny":       (lambda: models.convnext_tiny(weights=None),       768),
    "resnet50":            (None,                                            2048),
}


def _build_cls_backbone(name: str):
    if name == "resnet50":
        m = models.resnet50(weights=None)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten()), 2048
    entry = _BACKBONES.get(name)
    if entry is None:
        raise ValueError(f"Unknown backbone: {name}")
    factory, feat_dim = entry
    m = factory()
    return nn.Sequential(m.features, m.avgpool, nn.Flatten()), feat_dim


# ---------------------------------------------------------------------------
# Cart Quality Model (Stage 1)
# ---------------------------------------------------------------------------
class CartQualityModel(nn.Module):
    __slots__ = ("backbone", "feat_dim", "head")

    def __init__(self, backbone_name: str, dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        self.backbone, self.feat_dim = _build_cls_backbone(backbone_name)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Dual-Head Model (Stage 2: fill + bag)
# ---------------------------------------------------------------------------
class DualHeadModel(nn.Module):
    __slots__ = ("n_fill", "backbone", "feat_dim", "head_fill", "head_bag")

    def __init__(self, backbone_name: str, dropout: float = 0.3,
                 n_fill: int = 3, n_bag: int = 3):
        super().__init__()
        self.n_fill = n_fill
        self.backbone, self.feat_dim = _build_cls_backbone(backbone_name)
        self.head_fill = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim, n_fill),
        )
        self.head_bag = nn.Sequential(
            nn.LayerNorm(self.feat_dim + n_fill),
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim + n_fill, n_bag),
        )

    def forward(self, x):
        feats       = self.backbone(x)
        logits_fill = self.head_fill(feats)
        fill_probs  = torch.softmax(logits_fill, dim=1).detach()
        bag_input   = torch.cat([feats, fill_probs], dim=1)
        logits_bag  = self.head_bag(bag_input)
        return logits_fill, logits_bag


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _read_calibration(calib_path: Path, keys: dict) -> dict:
    """Read temperature values from calibration.json. Returns dict of key->float."""
    result = {k: default for k, default in keys.items()}
    if not calib_path.exists():
        return result
    try:
        data = json.loads(calib_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("calibration.read_failed", path=str(calib_path), error=str(e))
        return result
    for k, default in keys.items():
        raw = data.get(k, default)
        try:
            result[k] = float(raw)
        except (TypeError, ValueError):
            log.warning("calibration.bad_value", path=str(calib_path), key=k, raw=raw)
    return result


def _log_state_dict_mismatch(pt_path: str, result) -> None:
    if result.missing_keys or result.unexpected_keys:
        log.warning(
            "checkpoint.state_dict_mismatch",
            checkpoint=pt_path,
            missing_keys=list(result.missing_keys),
            unexpected_keys=list(result.unexpected_keys),
        )


# weights_only=False is required because these checkpoints embed config dicts
# (cfg, fill_classes, calibration) alongside tensors, which torch's safe-loader
# rejects. SAFE ONLY because these weights are baked into the container image
# at build time. DO NOT relax this trust boundary without switching to a safe
# loader if/when checkpoints can come from external sources (GCS uploads,
# user-provided artifacts, etc.) — pickle gives arbitrary code execution.
def load_quality_checkpoint(pt_path: str, device: str):
    """Load a CartQualityModel from checkpoint. Returns (model, temperature)."""
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    cfg      = ckpt.get("cfg", {})
    backbone = cfg.get("model", "mobilenet_v3_large")
    dropout  = cfg.get("dropout", 0.3)
    n_cls    = cfg.get("n_classes", 2)
    model = CartQualityModel(backbone, dropout=dropout, n_classes=n_cls).to(device)
    state = ckpt.get("model_state") or ckpt.get("ema_state")
    _log_state_dict_mismatch(pt_path, model.load_state_dict(state, strict=False))
    model.eval()
    calib = _read_calibration(
        Path(pt_path).parent.parent / "calibration.json",
        {"temperature": 1.0},
    )
    return model, calib["temperature"]


def load_fill_checkpoint(pt_path: str, device: str):
    """Load a DualHeadModel from checkpoint.
    Returns (model, fill_classes, bag_classes, n_bag, fill_temp, bag_temp).
    """
    # See trust note above load_quality_checkpoint re: weights_only=False.
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    fill_classes = ckpt.get("fill_classes")
    bag_classes  = ckpt.get("bag_classes", list(BAG_CLASSES))
    if fill_classes is None:
        hp_path = Path(pt_path).parent.parent / "hparams.json"
        if hp_path.exists():
            try:
                fill_classes = json.loads(hp_path.read_text()).get("fill_classes")
            except (OSError, json.JSONDecodeError) as e:
                log.warning("hparams.read_failed", path=str(hp_path), error=str(e))
    if fill_classes is None:
        fill_classes = ["empty", "partial", "full"]
    fill_classes = list(fill_classes)

    cfg      = ckpt.get("cfg", {})
    backbone = cfg.get("model", "mobilenet_v3_large")
    dropout  = cfg.get("dropout", 0.3)
    n_bag    = cfg.get("n_bag", len(bag_classes))

    bag_classes_display = list(bag_classes)
    if "not_applicable" not in bag_classes_display:
        bag_classes_display.append("not_applicable")

    model = DualHeadModel(backbone, dropout=dropout,
                          n_fill=len(fill_classes), n_bag=n_bag).to(device)
    state = ckpt.get("model_state") or ckpt.get("ema_state")
    _log_state_dict_mismatch(pt_path, model.load_state_dict(state, strict=False))
    model.eval()

    calib = _read_calibration(
        Path(pt_path).parent.parent / "calibration.json",
        {"fill_temperature": 1.0, "bag_temperature": 1.0},
    )
    return (model, fill_classes, bag_classes_display, n_bag,
            calib["fill_temperature"], calib["bag_temperature"])
