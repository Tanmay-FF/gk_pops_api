"""HeadlessTrackingEngine — JSON-only POPS pipeline.

Lifted from engine/tracker.py and stripped of every dependency that doesn't
contribute to the final _tracking.json:

  - No gradio (no progress bar default arg; the caller passes an optional
    progress_cb if it wants per-frame updates).
  - No pose estimation (was always-on in the original but only fed the 3D
    BEV, which this engine doesn't build).
  - No frame drawing (renderer.py).
  - No video writer / re-encode (the original wrote an AVI then transcoded
    to MP4 for display).
  - No 2D / 3D BEV builders.
  - No analytics, no VLM, no case-report, no trajectory cache.

Public entry: `HeadlessTrackingEngine.process_video()`. Returns a single
dict matching the structure documented in the project's _tracking.json
artifact (video_info, frames, events, cart_classifications, pops_summary,
summary, processing_info).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .classifier import CartClassifier
from .config import (
    ABANDON_FRAMES,
    CLASSIFY_EVERY_N_FRAMES,
    FILL_WEIGHT_PATH,
    JSON_EVERY_N_FRAMES,
    POPS_CLASSIFY_EVERY_N_FRAMES,
    MODEL_PATH,
    QUALITY_THRESHOLD,
    QUALITY_WEIGHT_PATH,
    TRACK_HISTORY_MAXLEN,
    TRACKER_CONFIG,
    TRAIL_MAX_LEN,
    WALKAWAY_DIST_THRESH,
    YOLO_IMGSZ,
)
from .linker import PersonCartLinker
from .motion import compute_direction_label, compute_motion
from .renderer import (
    class_color,
    draw_bbox,
    draw_centroid_trail,
    draw_classification_overlay,
    draw_hud,
    draw_link_lines,
    draw_person_overlay,
    event_color,
    outlined_text,
)
from .scoring import (
    HIGH_EVENTS,
    LOGGABLE_EVENTS,
    classify_event,
    compute_pops,
)
from .video_io import create_writer, open_video, reencode_to_mp4

# Carts younger than this many frames don't get scored — early frames are
# noisy and we don't want a 1-frame flicker to trigger a HIGH PRIORITY.
_MIN_CART_FRAMES_FOR_POPS = 10

# Severity ordering for the reconciliation pass at the end.
_EVENT_SEVERITY = {
    "PUSHOUT ALERT": 5,
    "HIGH PRIORITY": 4,
    "ABANDONED CART": 3,
    "MEDIUM PRIORITY": 3,
    "UNLINKED EXIT": 2,
    "LOW PRIORITY": 1,
}

_ABANDON_EVENTS = {"ABANDONED CART"}

# Ranks fill labels for the grab-and-run override; a history entry with a
# higher rank than the current label promotes it.
_FILL_RANK = {"empty": 0, "partial": 1, "full": 2}

ProgressCallback = Callable[[int, int], None]


class HeadlessTrackingEngine:
    """JSON-only tracking engine. Construct once; call process_video() per job."""

    def __init__(self, model_path: str = MODEL_PATH,
                 tracker_config: str = TRACKER_CONFIG,
                 device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)
        self.model.to(self.device)
        # We deliberately do NOT set cudnn.benchmark = True. Gradio leaves it
        # at the default (False), and enabling it caused cuDNN to pick
        # different conv algorithms for the same shapes, which produced
        # bit-different classifier outputs and flipped borderline fill/bag
        # predictions between API and Gradio on identical inputs.
        self.names = self.model.names
        self.tracker_config = tracker_config

        self._classifier = CartClassifier(self.device)

        # Per-run state is reset in _reset(); declared here for clarity.
        self._reset()

    def warmup(self) -> None:
        """Run one synthetic YOLO + BoTSORT pass to prime the GPU + tracker.

        Called from the FastAPI lifespan so /readyz only flips green once
        the first real request won't pay the cold-start tax.

        We deliberately use `model.track()` rather than `model.predict()`
        here so the tracker callbacks are registered EXACTLY ONCE for the
        lifetime of the process. ultralytics' `Model.add_callback()`
        appends to a list — calling `register_tracker()` on a fresh
        predictor for every job (the predictor=None approach) duplicates
        the on_predict_postprocess_end callback, which fires
        `tracker.update()` once per duplicate and corrupts BoTSORT's
        state on every job after the first. Doing the registration once
        in warmup avoids that whole class of bug.

        The dummy frame leaves frame_id=1 in the tracker; the subsequent
        `_reset_trackers()` call zeroes that out so the first real video
        starts cleanly.
        """
        dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
        self.model.track(
            dummy, persist=True, tracker=self.tracker_config,
            imgsz=YOLO_IMGSZ, verbose=False,
        )
        self._reset_trackers()

    def _reset_trackers(self) -> None:
        """Clear BoTSORT per-track state without recreating the predictor.

        Resets frame_id, tracked/lost/removed stracks, the Kalman filter,
        and the global track ID counter (BaseTrack._count). Leaves the
        predictor and its registered callbacks intact, so this is safe to
        call between jobs without re-registering and duplicating
        callbacks.
        """
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", []) or []:
            tracker.reset()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self._display_map: dict[str, dict[int, int]] = {}
        self._next_display: dict[str, int] = {}

        # Centroid trail rendered onto im0 BEFORE the classifier runs. The
        # classifier was trained against frames carrying these pixels (the
        # 5-px filled dot falls inside the cart bbox), so the trail must be
        # painted whether or not the caller wants the annotated video out.
        # Capped at TRAIL_MAX_LEN to match Gradio's render.
        self._track_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRAIL_MAX_LEN))

        # Bounded so long videos don't accumulate one entry per frame per track
        # forever. compute_motion and compute_direction_label only look at the
        # head and tail, so the cap is harmless to downstream logic.
        self._obj_positions: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRACK_HISTORY_MAXLEN))
        self._obj_timestamps: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRACK_HISTORY_MAXLEN))
        self._obj_speeds: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRACK_HISTORY_MAXLEN))
        self._obj_labels: dict[int, str] = {}
        self._obj_bboxes: dict[int, tuple[int, int, int, int]] = {}
        self._obj_first_frame: dict[int, int] = {}
        self._obj_disappeared: dict[int, int] = defaultdict(int)

        self._linker = PersonCartLinker(self._get_display_id)

        self._json_frames: dict[str, dict] = {}
        self._all_people_seen: set[int] = set()
        self._all_carts_seen: set[int] = set()

        # Two caches: the live one (every frame) feeds the per-frame JSON;
        # the pops one (every pcenf frames) feeds POPS scoring + event logging.
        # Decoupled so per-frame JSON stays live without flip-flopping the
        # event-logging quality guard.
        self._cart_cls_cache: dict[int, dict] = {}
        self._cart_cls_pops_cache: dict[int, dict] = {}
        self._pops_cache: dict[int, dict] = {}
        self._event_log: list[dict] = []
        self._max_pops_per_cart: dict[int, int] = {}
        self._peak_pops_snapshot: dict[int, dict] = {}
        self._cart_cls_history: dict[int, list[tuple]] = defaultdict(list)
        self._motion_cache: dict[int, tuple] = {}
        self._walkaway_frames: dict[int, int] = {}

    def _get_display_id(self, label: str, raw_id: int) -> int:
        if label not in self._display_map:
            self._display_map[label] = {}
            self._next_display[label] = 1
        m = self._display_map[label]
        if raw_id not in m:
            m[raw_id] = self._next_display[label]
            self._next_display[label] += 1
        return m[raw_id]

    def _link_label(self, raw_id: int, is_person: bool) -> str | None:
        """Return the '-> Cart:N' / '-> Person:N' label used by overlays.

        Only consulted when writing the annotated video — the JSON output
        already carries this information in the `links` block.
        """
        links = self._linker.links
        gdi = self._get_display_id
        if is_person:
            for cid, pid in links.items():
                if pid == raw_id:
                    return f"-> Cart:{gdi('cart', cid)}"
            pd = gdi('person', raw_id)
            if pd in self._linker.permanently_linked_persons:
                for cid, pid in links.items():
                    if gdi('person', pid) == pd:
                        return f"-> Cart:{gdi('cart', cid)}"
            return None
        if raw_id in links:
            return f"-> Person:{gdi('person', links[raw_id])}"
        cd = gdi('cart', raw_id)
        if cd in self._linker.permanently_linked_carts:
            for cid, pid in links.items():
                if gdi('cart', cid) == cd:
                    return f"-> Person:{gdi('person', pid)}"
        return None

    # ------------------------------------------------------------------
    # Per-frame JSON
    # ------------------------------------------------------------------
    def _build_frame_json(self, frame_idx: int, timestamp: float,
                          frame_detections: list, camera_placement: str) -> dict:
        people: dict[str, dict] = {}
        carts: dict[str, dict] = {}
        frame_persons: dict[int, tuple[float, float]] = {}
        frame_carts: dict[int, tuple[float, float]] = {}
        gdi = self._get_display_id
        links = self._linker.links

        for raw_id, cls, conf, bbox in frame_detections:
            label = self.names[int(cls)]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            is_person = label == 'person'
            is_cart = label == 'cart'
            if not (is_person or is_cart):
                continue

            cached = self._motion_cache.get(raw_id)
            if cached:
                speed, direction, speed_status, accel, dir_label = cached
            else:
                speed, direction, speed_status, accel = compute_motion(
                    self._obj_positions[raw_id], self._obj_timestamps[raw_id],
                    self._obj_speeds[raw_id], 0.0)
                dir_label = compute_direction_label(
                    self._obj_positions[raw_id], camera_placement)

            display_id = gdi('person' if is_person else 'cart', raw_id)
            key = f"{'P' if is_person else 'C'}{display_id}"

            # list(deque)[-5:] — deque slicing isn't supported directly.
            pos_hist = [{"x": round(p[0], 1), "y": round(p[1], 1)}
                        for p in list(self._obj_positions[raw_id])[-5:]]
            spd_hist = [round(s, 2)
                        for s in list(self._obj_speeds[raw_id])[-5:]]

            obj = {
                "id": display_id,
                "centroid": {"x": round(cx, 1), "y": round(cy, 1)},
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "width": x2 - x1, "height": y2 - y1},
                "motion": {"speed": round(speed, 2), "direction": round(direction, 2),
                           "direction_label": dir_label, "speed_status": speed_status,
                           "acceleration": round(accel, 2)},
                "tracking": {"positions_history": pos_hist, "speed_history": spd_hist,
                             "disappeared_frames": 0, "yolo_confidence": round(conf, 4)},
            }

            if is_person:
                obj["linking"] = {"is_linked": False, "linked_cart_id": None, "link_confidence": 0.0}
                people[key] = obj
                frame_persons[raw_id] = (cx, cy)
                self._all_people_seen.add(raw_id)
            else:
                cr = self._cart_cls_cache.get(display_id, {})
                pi = self._pops_cache.get(display_id, {})
                obj["classification"] = {
                    "quality": cr.get("quality", "unclassified"),
                    "fill": cr.get("fill", "unclassified"),
                    "bag": cr.get("bag", "unclassified"),
                    "quality_conf": round(cr.get("quality_conf", 0.0), 4),
                    "fill_conf": round(cr.get("fill_conf", 0.0), 4),
                    "bag_conf": round(cr.get("bag_conf", 0.0), 4),
                }
                obj["pops"] = {"score": pi.get("score", 0), "event": pi.get("event", "CLEAR")}
                obj["linking"] = {"is_linked": False, "linked_person_id": None, "link_confidence": 0.0}
                carts[key] = obj
                frame_carts[raw_id] = (cx, cy)
                self._all_carts_seen.add(raw_id)

        # Link info
        link_data: dict[str, dict] = {}
        active = 0
        seen_pairs: set[tuple[str, str]] = set()
        for cart_raw, person_raw in links.items():
            cd = gdi('cart', cart_raw)
            pd = gdi('person', person_raw)
            ck, pk = f"C{cd}", f"P{pd}"
            if (pk, ck) in seen_pairs:
                continue
            c_in, p_in = ck in carts, pk in people
            if c_in and p_in:
                cp = frame_carts.get(cart_raw, (0, 0))
                pp = frame_persons.get(person_raw, (0, 0))
                dist = math.sqrt((cp[0] - pp[0]) ** 2 + (cp[1] - pp[1]) ** 2)
                carts[ck]["linking"].update({"is_linked": True, "linked_person_id": pd})
                people[pk]["linking"].update({"is_linked": True, "linked_cart_id": cd})
                sf = self._linker.link_start_frames.get(cart_raw, frame_idx)
                link_data[f"{pk}_{ck}"] = {
                    "person_id": pd, "cart_id": cd, "distance": round(dist, 2),
                    "established_frame": sf, "duration_frames": frame_idx - sf,
                }
                active += 1
            elif c_in or p_in:
                if c_in:
                    carts[ck]["linking"].update({"is_linked": True, "linked_person_id": pd})
                if p_in:
                    people[pk]["linking"].update({"is_linked": True, "linked_cart_id": cd})
                active += 1
            seen_pairs.add((pk, ck))

        p_dis = sum(1 for r, l in self._obj_labels.items()
                    if l == 'person' and 0 < self._obj_disappeared[r] < 90)
        c_dis = sum(1 for r, l in self._obj_labels.items()
                    if l == 'cart' and 0 < self._obj_disappeared[r] < 90)

        return {
            "frame_number": frame_idx, "timestamp": round(timestamp, 4),
            "people": people, "carts": carts, "links": link_data,
            "statistics": {"total_people": len(people), "total_carts": len(carts),
                           "active_links": active,
                           "people_disappeared": p_dis, "carts_disappeared": c_dis},
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def process_video(self, source_path: str, *,
                      camera_placement: str = "outside_facing_entrance",
                      json_every_n_frames: int | None = None,
                      classify_every_n_frames: int | None = None,
                      pops_classify_every_n_frames: int | None = None,
                      annotated_video_path: Optional[str] = None,
                      progress_cb: Optional[ProgressCallback] = None) -> dict[str, Any]:
        """Process a video end-to-end and return the _tracking.json structure.

        Args:
            source_path: Path to the input video.
            camera_placement: One of the 5 placement strings the engine recognises.
                Determines INBOUND/OUTBOUND labelling in motion direction.
            json_every_n_frames: Build a JSON entry every N processed frames.
                None falls back to the config default (1).
            classify_every_n_frames: Run the classifier every N frames — controls
                how often _cart_cls_cache (and per-frame JSON) is updated.
                None falls back to the config default (1 = every frame).
            pops_classify_every_n_frames: Sampling cadence for the
                reconciliation vote — controls how often the classifier
                result is snapshotted into `_cart_cls_pops_cache` (which
                POPS scoring reads) and appended to `_cart_cls_history`
                (the post-loop fill/bag vote). POPS scoring + event
                logging themselves run every frame; pcenf only governs
                what classifier data they score against, so transient
                FAST or abandonment frames are still caught the moment
                they happen. None falls back to the config default.
            annotated_video_path: When set, also write an MP4 with the same
                overlay set the Gradio demo produces (bboxes, trails, fill /
                bag / POPS chips, link lines, HUD) to this path. The bbox
                and trail are drawn on every frame regardless — they're
                required for classifier parity with Gradio — so the extra
                cost when this is set is the post-classify overlay pass +
                the AVI writer + the H.264 re-encode.
            progress_cb: Optional `(frame_idx, total_frames) -> None` callback,
                called once per processed frame.

        Returns:
            A dict matching the legacy _tracking.json shape — keys
            `video_info`, `frames`, `events`, `cart_classifications`,
            `pops_summary`, `summary`, `processing_info`.
        """
        self._reset()
        # Clear BoTSORT state from the previous job (or from the warmup's
        # dummy frame, on the very first job). This zeroes frame_id and
        # drops tracked/lost/removed stracks + the global ID counter, so
        # this video's frame 1 is treated as frame_id == 1 by the tracker
        # and unmatched detections are activated immediately with fresh
        # IDs starting at 1. We do NOT recreate the predictor here —
        # ultralytics' Model.add_callback() appends, so recreating the
        # predictor would force register_tracker() to run again and
        # duplicate the tracker callbacks, which corrupts state on every
        # job after the first. Trackers are registered exactly once,
        # during warmup.
        self._reset_trackers()
        self._classifier.set_quality_threshold(QUALITY_THRESHOLD)
        jenf  = json_every_n_frames if json_every_n_frames is not None else JSON_EVERY_N_FRAMES
        cenf  = classify_every_n_frames if classify_every_n_frames is not None else CLASSIFY_EVERY_N_FRAMES
        pcenf = pops_classify_every_n_frames if pops_classify_every_n_frames is not None else POPS_CLASSIFY_EVERY_N_FRAMES
        t_start = time.perf_counter()

        self._classifier.load_quality(QUALITY_WEIGHT_PATH)
        self._classifier.load_fill(FILL_WEIGHT_PATH)

        cap, w, h, fps, total_frames = open_video(source_path)
        names = self.names
        gdi = self._get_display_id
        links = self._linker.links

        # Annotated-video output: write to a sibling AVI first, then re-encode
        # to MP4 at the end. Keep both paths next to the requested final path
        # so cleanup is one directory tree.
        writer = None
        avi_path: Optional[Path] = None
        mp4_path: Optional[Path] = None
        want_video = annotated_video_path is not None
        if want_video:
            mp4_path = Path(annotated_video_path)
            avi_path = mp4_path.with_suffix(".avi")
            writer = create_writer(avi_path, w, h, fps)

        _t_yolo = _t_cls = _t_other = _t_json = _t_draw = 0.0
        frame_idx = 0

        try:
            while True:
                ok, im0 = cap.read()
                if not ok:
                    break
                frame_idx += 1
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # YOLO + BoTSORT. Signature mirrors Gradio's call exactly
                # (no `device=` kwarg, no outer torch.no_grad wrapper) so
                # ultralytics' internal pipeline takes the same code path
                # and the bytes coming out of the conv kernels match.
                _t0 = time.perf_counter()
                results = self.model.track(
                    im0, persist=True, tracker=self.tracker_config,
                    imgsz=YOLO_IMGSZ, verbose=False,
                )
                _t_yolo += time.perf_counter() - _t0

                frame_detections: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu()
                    ids = r.boxes.id.cpu().tolist()
                    clss = r.boxes.cls.tolist()
                    confs = r.boxes.conf.cpu().tolist()

                    # Cart re-ID before we update tracking state.
                    cur_cart_raws = {
                        int(id_) for box, id_, c, _ in zip(boxes, ids, clss, confs)
                        if names[int(c)] == 'cart'
                    }
                    for box, id_, c, conf in zip(boxes, ids, clss, confs):
                        if names[int(c)] != 'cart':
                            continue
                        raw = int(id_)
                        bb = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                        self._linker.try_reidentify_cart(
                            raw, bb, cur_cart_raws, self._display_map,
                            self._obj_positions, self._obj_timestamps,
                            self._obj_speeds, self._obj_disappeared)

                    _t_pre_draw = time.perf_counter()
                    for box, id_, c, conf in zip(boxes, ids, clss, confs):
                        raw = int(id_)
                        label = names[int(c)]
                        if label not in ('person', 'cart'):
                            continue
                        x1, y1, x2, y2 = (float(box[0]), float(box[1]),
                                          float(box[2]), float(box[3]))
                        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                        bb_int = (int(x1), int(y1), int(x2), int(y2))
                        frame_detections.append((raw, c, conf, bb_int))

                        # Paint the bbox + centroid trail onto im0 BEFORE the
                        # classifier crop. The fill/bag head was trained
                        # against frames carrying these pixels (notably the
                        # 5-px filled centroid dot landing inside the cart
                        # crop), so skipping this step silently degrades
                        # classifier accuracy. This is the parity bug fix —
                        # not a video-rendering side effect.
                        disp = gdi(label, raw)
                        col = class_color(label)
                        draw_bbox(im0, box, disp, label, col)
                        track = self._track_history[raw]
                        track.append((cx, cy))
                        draw_centroid_trail(im0, track, cx, cy, col)

                        self._obj_positions[raw].append((cx, cy))
                        self._obj_timestamps[raw].append(timestamp)
                        self._obj_labels[raw] = label
                        self._obj_bboxes[raw] = bb_int
                        if raw not in self._obj_first_frame:
                            self._obj_first_frame[raw] = frame_idx
                        self._obj_disappeared[raw] = 0
                    _t_draw += time.perf_counter() - _t_pre_draw

                # Disappearance tracking
                seen = {d[0] for d in frame_detections}
                for rid in self._obj_labels:
                    if rid not in seen:
                        self._obj_disappeared[rid] += 1

                # Linking
                person_bb: dict[int, tuple] = {}
                cart_bb: dict[int, tuple] = {}
                for raw, c, _, bb in frame_detections:
                    lbl = names[int(c)]
                    if lbl == 'person':
                        person_bb[raw] = bb
                    elif lbl == 'cart':
                        cart_bb[raw] = bb
                self._linker.update(
                    person_bb, cart_bb, frame_idx,
                    self._obj_disappeared, self._obj_positions,
                    self._obj_first_frame)

                # POPS-cadence gate. Classification still runs every cenf frame
                # (so per-frame JSON has live fill/bag/quality), but the POPS
                # scoring + reconciliation snapshots only refresh every pcenf
                # frames. Hoisted above the classifier block so the POPS loop
                # below can see it too.
                is_pops_frame = (frame_idx % pcenf == 0)

                # Classification (batched when possible)
                _t0 = time.perf_counter()
                if frame_idx % cenf == 0 and self._classifier.has_quality_model:
                    cart_bbs: list[tuple] = []
                    cart_ids: list[int] = []
                    for raw, c, _, bb in frame_detections:
                        if names[int(c)] != 'cart':
                            continue
                        cart_bbs.append(bb)
                        cart_ids.append(gdi('cart', raw))
                    if cart_bbs:
                        batch_results = self._classifier.classify_batch(im0, cart_bbs, cart_ids)
                        for cd, result in batch_results.items():
                            self._cart_cls_cache[cd] = result
                            if is_pops_frame:
                                # Snapshot what POPS scoring sees. Held steady
                                # between pcenf-multiple frames so the event
                                # logging quality guard doesn't flip-flop.
                                self._cart_cls_pops_cache[cd] = result
                                if result.get("quality") == "valid_cart":
                                    self._cart_cls_history[cd].append((
                                        result["fill"], result["bag"],
                                        result.get("fill_conf", 0.0),
                                        result.get("bag_conf", 0.0),
                                    ))
                _t_cls += time.perf_counter() - _t0

                # Motion cache (re-computed every frame; cached so JSON build doesn't redo it)
                _t0 = time.perf_counter()
                self._motion_cache.clear()
                for raw, c, _, bb in frame_detections:
                    speed, direction, speed_status, accel = compute_motion(
                        self._obj_positions[raw], self._obj_timestamps[raw],
                        self._obj_speeds[raw], fps)
                    dir_label = compute_direction_label(
                        self._obj_positions[raw], camera_placement)
                    self._motion_cache[raw] = (speed, direction, speed_status, accel, dir_label)
                    self._obj_speeds[raw].append(speed)

                # Sync linked cart direction with person direction
                for cart_raw, person_raw in links.items():
                    if cart_raw in self._motion_cache and person_raw in self._motion_cache:
                        person_dir = self._motion_cache[person_raw][4]
                        if person_dir in ("INBOUND", "OUTBOUND"):
                            old = self._motion_cache[cart_raw]
                            self._motion_cache[cart_raw] = (old[0], old[1], old[2], old[3], person_dir)

                # Walkaway/abandonment counters AND POPS scoring + event
                # logging — both tick every frame so transient FAST or
                # abandonment frames are caught the moment they happen.
                # The sparse cadence (pcenf) only governs *what classifier
                # data POPS scores against*: POPS reads
                # `_cart_cls_pops_cache`, which is only refreshed on
                # pops-cadence frames. So scoring stays stable between
                # classifier snapshots even though it runs every frame.
                for raw, c, _, bb in frame_detections:
                    if names[int(c)] != 'cart':
                        continue
                    cd = gdi('cart', raw)
                    cart_age = frame_idx - self._obj_first_frame.get(raw, frame_idx)
                    if cart_age < _MIN_CART_FRAMES_FOR_POPS:
                        continue
                    speed, _, speed_status, _, dir_label = self._motion_cache[raw]

                    linked = False
                    linked_person_raw = None
                    for cid, pid in links.items():
                        if gdi('cart', cid) == cd:
                            linked = True
                            linked_person_raw = pid
                            break
                    person_gone = (
                        linked and linked_person_raw is not None
                        and self._obj_disappeared.get(linked_person_raw, 0) > ABANDON_FRAMES
                    )
                    person_far = False
                    if linked and linked_person_raw is not None and linked_person_raw in person_bb:
                        pb = person_bb[linked_person_raw]
                        pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                        ccx, ccy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                        dist = ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5
                        if dist > WALKAWAY_DIST_THRESH:
                            self._walkaway_frames[cd] = self._walkaway_frames.get(cd, 0) + 1
                        else:
                            self._walkaway_frames.pop(cd, None)
                        person_far = self._walkaway_frames.get(cd, 0) > ABANDON_FRAMES
                    abandoned = person_gone or person_far

                    # POPS scoring reads the sparse-cadence cache, not the live
                    # one — keeps the quality guard from flipping every frame.
                    # The cache only refreshes on is_pops_frame, so between
                    # snapshots POPS sees the same fill/bag/quality even
                    # though the per-frame JSON sees fresh values.
                    cr = self._cart_cls_pops_cache.get(cd, {})
                    is_valid = cr.get("is_valid", True)
                    fill_lbl = cr.get("fill", "unclassified")
                    bag_lbl = cr.get("bag", "not_applicable")

                    # Grab-and-run override (single-frame view; the final
                    # reconciliation pass below does the more thorough vote).
                    if abandoned and fill_lbl == "empty" and cd in self._cart_cls_history:
                        for h_fill, h_bag, _, _ in self._cart_cls_history[cd]:
                            if _FILL_RANK.get(h_fill, 0) > _FILL_RANK.get(fill_lbl, 0):
                                fill_lbl = h_fill
                                bag_lbl = h_bag

                    pops_score = compute_pops(
                        dir_label, speed_status, is_valid, fill_lbl,
                        bag_label=bag_lbl, cart_detected=True,
                        abandoned=abandoned, linked=linked,
                    )
                    event_name = classify_event(
                        pops_score, linked, dir_label, abandoned=abandoned)

                    self._pops_cache[cd] = {
                        "score": pops_score, "event": event_name,
                        "fill": fill_lbl, "bag": bag_lbl, "direction": dir_label,
                        "speed_status": speed_status, "linked": linked,
                    }

                    prev_max = self._max_pops_per_cart.get(cd, 0)
                    quality_lbl = cr.get("quality", "unclassified")
                    if pops_score >= prev_max:
                        self._max_pops_per_cart[cd] = pops_score
                        self._peak_pops_snapshot[cd] = {
                            "score": pops_score,
                            "event": event_name,
                            "fill": fill_lbl, "bag": bag_lbl, "direction": dir_label,
                            "quality": quality_lbl,
                            "speed_status": speed_status,
                            "linked": linked, "abandoned": abandoned,
                        }

                    if event_name in LOGGABLE_EVENTS:
                        skip = False
                        if quality_lbl in ("unclear", "unclassified") and event_name not in HIGH_EVENTS:
                            skip = True
                        cart_has_high = any(
                            e["cart_id"] == cd and e["event"] in HIGH_EVENTS
                            for e in self._event_log
                        )
                        already_logged = any(
                            e["cart_id"] == cd and e["event"] == event_name
                            for e in self._event_log
                        )
                        if (not skip and not already_logged
                                and not (cart_has_high and event_name not in HIGH_EVENTS)):
                            self._event_log.append({
                                "frame": frame_idx, "timestamp": round(timestamp, 2),
                                "cart_id": cd, "event": event_name, "pops_score": pops_score,
                                "fill": fill_lbl, "bag": bag_lbl,
                                "direction": dir_label, "linked": linked,
                                "speed_status": speed_status, "abandoned": abandoned,
                            })

                _t_other += time.perf_counter() - _t0

                # JSON sample
                if frame_idx % jenf == 0 or frame_idx == 1:
                    _t0 = time.perf_counter()
                    self._json_frames[str(frame_idx)] = self._build_frame_json(
                        frame_idx, timestamp, frame_detections, camera_placement)
                    _t_json += time.perf_counter() - _t0

                # Annotated-video overlays + frame write. Only done when the
                # caller asked for the MP4 — the bbox / trail above are
                # ALWAYS drawn (classifier needs them), but these per-frame
                # chips and HUD are skipped on JSON-only jobs to keep the
                # path light.
                if writer is not None:
                    _t0 = time.perf_counter()
                    p_count = sum(1 for _, c, _, _ in frame_detections
                                  if names[int(c)] == 'person')
                    c_count = sum(1 for _, c, _, _ in frame_detections
                                  if names[int(c)] == 'cart')

                    p_d2r: dict[int, int] = {}
                    c_d2r: dict[int, int] = {}
                    for raw, c, _, _ in frame_detections:
                        lbl = names[int(c)]
                        if lbl == 'person':
                            p_d2r[gdi('person', raw)] = raw
                        elif lbl == 'cart':
                            c_d2r[gdi('cart', raw)] = raw

                    for raw, c, _, bb in frame_detections:
                        if names[int(c)] != 'person':
                            continue
                        cached = self._motion_cache.get(raw)
                        if not cached:
                            continue
                        _, _, status, _, dlbl = cached
                        draw_person_overlay(
                            im0, bb, status, dlbl, self._link_label(raw, True))

                    for raw, c, _, bb in frame_detections:
                        if names[int(c)] != 'cart':
                            continue
                        cd = gdi('cart', raw)
                        cls_result = self._cart_cls_cache.get(cd)
                        pi = self._pops_cache.get(cd)
                        if pi is not None:
                            pi = dict(pi)
                            pi.setdefault("color", event_color(pi.get("event", "CLEAR")))
                        draw_classification_overlay(im0, bb, cls_result, pi)
                        lp = self._link_label(raw, False)
                        if lp:
                            oy = int(bb[3]) + 18 + 16 * 3
                            outlined_text(im0, lp, (int(bb[0]), oy), 0.45,
                                          (0, 255, 0))

                    det_centroids: dict[int, tuple[int, int]] = {}
                    for raw, _c, _conf, bb in frame_detections:
                        det_centroids[raw] = (
                            int((bb[0] + bb[2]) // 2),
                            int((bb[1] + bb[3]) // 2),
                        )
                    active_links = draw_link_lines(
                        im0, links, det_centroids, gdi, p_d2r, c_d2r)

                    draw_hud(im0, p_count, c_count, active_links,
                             frame_idx, total_frames, w)

                    writer.write(im0)
                    _t_draw += time.perf_counter() - _t0

                if progress_cb is not None:
                    progress_cb(frame_idx, total_frames)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # ------------------------------------------------------------------
        # Reconciliation (post-loop): build a coherent final fill/bag/score
        # ------------------------------------------------------------------
        _best_event: dict[int, dict] = {}
        for ev in self._event_log:
            cd = ev["cart_id"]
            sev = _EVENT_SEVERITY.get(ev["event"], 0)
            prev = _best_event.get(cd)
            prev_sev = _EVENT_SEVERITY.get(prev["event"], 0) if prev else -1
            if sev > prev_sev or (sev == prev_sev and ev["frame"] > (prev or {}).get("frame", 0)):
                _best_event[cd] = ev

        for cd in set(list(self._peak_pops_snapshot) + list(self._cart_cls_history)):
            if cd not in self._peak_pops_snapshot:
                continue
            snap = self._peak_pops_snapshot[cd]

            direction = snap.get("direction", "UNKNOWN")
            speed_status = snap.get("speed_status", "STATIC")
            linked = snap.get("linked", False)
            abandoned = snap.get("abandoned", False)

            if cd in _best_event:
                ev = _best_event[cd]
                direction = ev["direction"]
                linked = ev["linked"]
                speed_status = ev.get("speed_status", speed_status)
                abandoned = ev.get("abandoned", abandoned)

            best_fill: Optional[str] = None
            best_bag: Optional[str] = None

            if cd in self._cart_cls_history:
                history = self._cart_cls_history[cd]
                if history:
                    fill_conf: dict[str, float] = defaultdict(float)
                    fill_count: dict[str, int] = defaultdict(int)
                    bag_conf: dict[str, float] = defaultdict(float)
                    bag_count: dict[str, int] = defaultdict(int)
                    for fill, bag, fc, bc in history:
                        fill_conf[fill] += fc
                        fill_count[fill] += 1
                        bag_conf[bag] += bc
                        bag_count[bag] += 1
                    fill_scores = {f: fill_conf[f] * fill_count[f] for f in fill_count}
                    bag_scores = {b: bag_conf[b] * bag_count[b] for b in bag_count}
                    best_fill = max(fill_scores, key=fill_scores.get)
                    best_bag = max(bag_scores, key=bag_scores.get)

                    # Grab-and-run: items in first half → empty in second half.
                    if best_fill == "empty" and abandoned:
                        n = len(history)
                        mid = max(1, n // 2)
                        first_half = history[:mid]
                        second_half = history[mid:]
                        first_fill_count: dict[str, int] = defaultdict(int)
                        for f, b, fc, bc in first_half:
                            first_fill_count[f] += 1
                        second_fill_count: dict[str, int] = defaultdict(int)
                        for f, b, fc, bc in second_half:
                            second_fill_count[f] += 1
                        first_had_items = ((first_fill_count.get("full", 0)
                                            + first_fill_count.get("partial", 0))
                                           >= len(first_half) * 0.5)
                        second_is_empty = (
                            second_fill_count.get("empty", 0) > len(second_half) * 0.7
                        )
                        if first_had_items and second_is_empty:
                            for candidate in ("full", "partial"):
                                if first_fill_count.get(candidate, 0) > 0:
                                    best_fill = candidate
                                    paired_bags: dict[str, float] = defaultdict(float)
                                    for f, b, fc, bc in first_half:
                                        if f == candidate:
                                            paired_bags[b] += bc
                                    if paired_bags:
                                        best_bag = max(paired_bags, key=paired_bags.get)
                                    break

            if best_fill is None:
                continue

            if best_fill in ("partial", "full") and best_bag == "not_applicable":
                if cd in self._cart_cls_history:
                    bag_scores2: dict[str, float] = defaultdict(float)
                    for _, bag, _, bc in self._cart_cls_history[cd]:
                        if bag != "not_applicable":
                            bag_scores2[bag] += bc
                    best_bag = max(bag_scores2, key=bag_scores2.get) if bag_scores2 else "unbagged"
                else:
                    best_bag = "unbagged"

            recomputed = compute_pops(
                direction, speed_status, True, best_fill,
                bag_label=best_bag, cart_detected=True,
                abandoned=abandoned, linked=linked,
            )
            final_score = recomputed
            if best_fill == "partial" and best_bag == "bagged":
                final_score = min(final_score, 55)
            final_event = classify_event(
                final_score, linked, direction, abandoned=abandoned)

            snap.update({
                "fill": best_fill, "bag": best_bag, "quality": "valid_cart",
                "score": final_score, "event": final_event,
                "direction": direction, "speed_status": speed_status,
                "linked": linked, "abandoned": abandoned,
            })
            self._max_pops_per_cart[cd] = final_score

        # Sync last event per cart with the reconciled POPS snapshot
        _last_event: dict[int, dict] = {}
        for ev in self._event_log:
            _last_event[ev["cart_id"]] = ev
        for cd, ev in _last_event.items():
            if cd not in self._peak_pops_snapshot:
                continue
            snap = self._peak_pops_snapshot[cd]
            if ev["event"] in _ABANDON_EVENTS:
                # Abandonment events are truth; POPS adopts them.
                snap["fill"] = ev["fill"]
                snap["bag"] = ev["bag"]
                snap["score"] = ev["pops_score"]
                snap["event"] = ev["event"]
                self._max_pops_per_cart[cd] = ev["pops_score"]
            else:
                # Otherwise the reconciled POPS score wins; backfill the event.
                ev["fill"] = snap["fill"]
                ev["bag"] = snap["bag"]
                ev["pops_score"] = snap["score"]
                ev["event"] = snap["event"]

        # Re-encode the per-job AVI to MP4 if we wrote one. Done after the
        # reconciliation pass so a re-encode failure doesn't lose the JSON
        # result — the caller decides what to do without the video.
        annotated_video_out: Optional[str] = None
        if want_video and avi_path is not None and mp4_path is not None:
            _t_enc0 = time.perf_counter()
            try:
                reencode_to_mp4(avi_path, mp4_path, delete_source=True)
                annotated_video_out = str(mp4_path)
            except Exception:  # noqa: BLE001 — JSON path must survive
                annotated_video_out = None
            _t_enc = round(time.perf_counter() - _t_enc0, 2)
        else:
            _t_enc = 0.0

        t_end = time.perf_counter()

        return {
            "video_info": {
                "video_name": _basename(source_path),
                "width": w, "height": h, "fps": float(fps),
                "total_frames": total_frames,
                "processing_timestamp": datetime.now().isoformat(),
            },
            "frames": self._json_frames,
            "events": self._event_log,
            "cart_classifications": {
                f"C{cid}": self._cart_cls_pops_cache.get(cid, {})
                for cid in self._cart_cls_pops_cache
            },
            "pops_summary": {
                f"C{cid}": {
                    "max_score": self._max_pops_per_cart.get(cid, 0),
                    "peak_event": self._peak_pops_snapshot.get(cid, {}).get("event", "CLEAR"),
                }
                for cid in set(list(self._max_pops_per_cart) + list(self._pops_cache))
            },
            "summary": {
                "total_people_seen": len(self._all_people_seen),
                "total_carts_seen": len(self._all_carts_seen),
                "total_links_established": self._linker.total_links,
                "total_events": len(self._event_log),
                "high_priority": sum(1 for e in self._event_log if e["event"] in HIGH_EVENTS),
                "medium_priority": sum(
                    1 for e in self._event_log
                    if e["event"] in {"MEDIUM PRIORITY", "UNLINKED EXIT", "ABANDONED CART"}
                ),
            },
            "processing_info": {
                "total_frames_processed": frame_idx,
                "json_sampled_frames": len(self._json_frames),
                "json_every_n": jenf,
                "classify_every_n": cenf,
                "pops_classify_every_n": pcenf,
                "device": self.device,
                "model": "YOLOv26m", "tracker": "BoTSORT",
                "quality_model": self._classifier.quality_pt or "None",
                "fill_model": self._classifier.fill_pt or "None",
                "quality_threshold": QUALITY_THRESHOLD,
                "wall_seconds": round(t_end - t_start, 2),
                "annotated_video_path": annotated_video_out,
                "perf_breakdown_seconds": {
                    "yolo_track": round(_t_yolo, 2),
                    "classify": round(_t_cls, 2),
                    "motion_pops": round(_t_other, 2),
                    "json_build": round(_t_json, 2),
                    "draw": round(_t_draw, 2),
                    "video_encode": _t_enc,
                },
            },
        }


def _basename(path: str) -> str:
    # Cheap, OS-agnostic basename; avoids importing os for one call site.
    sep = max(path.rfind("/"), path.rfind("\\"))
    return path[sep + 1:] if sep >= 0 else path
