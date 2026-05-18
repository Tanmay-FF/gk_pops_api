"""OpenCV drawing primitives — bboxes, trails, overlays, link lines, HUD.

Two distinct callers:

  1. The tracker calls draw_bbox / draw_centroid_trail BEFORE the classifier
     runs. The Gradio demo also paints these onto im0 in that order, and the
     fill/bag head was implicitly trained against frames carrying those
     pixels (notably the 5-px filled centroid dot landing inside the cart
     bbox). Skipping them silently degrades the classifier — that's the
     parity bug this module exists to fix.

  2. When the API caller opts into an annotated MP4, the tracker also calls
     draw_classification_overlay / draw_person_overlay / draw_link_lines /
     draw_hud after the per-frame state is finalised so the video carries
     the same overlay set the Gradio output does.

All functions mutate im0 in place and return nothing (or a count, for
draw_link_lines).
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import (
    COLOR_LINK,
    CLR_UNCLEAR, CLR_NA, FILL_COLOR_MAP,
    COLOR_PUSHOUT, COLOR_SUSPICIOUS, COLOR_MONITORING, COLOR_CLEAR,
    COLOR_PERSON, COLOR_CART,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def event_color(event_name: str):
    """Map an event name to the BGR colour Gradio uses for that event."""
    if event_name in ("PUSHOUT ALERT", "HIGH PRIORITY"):
        return COLOR_PUSHOUT
    if event_name in ("ABANDONED CART", "MEDIUM PRIORITY", "UNLINKED EXIT"):
        return COLOR_SUSPICIOUS
    if event_name == "MONITORING":
        return COLOR_MONITORING
    return COLOR_CLEAR


def class_color(label: str):
    """Bbox colour for a YOLO class label."""
    if label == "person":
        return COLOR_PERSON
    if label == "cart":
        return COLOR_CART
    return (255, 255, 255)


def outlined_text(im0, text, pos, scale, color, thickness=1):
    cv2.putText(im0, text, pos, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(im0, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bbox(im0, box, display_id, label, color):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
    text = f"{label}:{display_id}"
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.8, 2)
    bg_x1 = x1
    bg_x2 = bg_x1 + tw + 20
    bg_y2 = y1
    bg_y1 = bg_y2 - th - 16
    cv2.rectangle(im0, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    tx = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
    ty = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2
    cv2.putText(im0, text, (tx, ty), _FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def draw_centroid_trail(im0, track, cx, cy, color):
    cv2.circle(im0, (int(cx), int(cy)), 5, color, -1)
    if len(track) >= 2:
        pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(im0, [pts], False, color, 2)


def draw_classification_overlay(im0, bbox, cls_result, pops_info):
    """Draw quality / fill / bag / POPS below cart bbox.

    pops_info may be None or a dict with keys score, event, color.
    """
    if cls_result is None:
        return
    x1 = int(bbox[0])
    oy = int(bbox[3]) + 18

    quality = cls_result.get("quality")
    if quality == "unclear":
        outlined_text(im0, "UNCLEAR", (x1, oy), 0.45, CLR_UNCLEAR)
        oy += 16
        outlined_text(im0, "Fill: N/A | Bag: N/A", (x1, oy), 0.4, CLR_NA)
        oy += 16
    elif quality == "valid_cart":
        fill_lbl = cls_result.get("fill", "").upper()
        bag_lbl = cls_result.get("bag", "").replace("_", " ").upper()
        fc = FILL_COLOR_MAP.get(fill_lbl, CLR_NA)
        outlined_text(im0, f"{fill_lbl} | {bag_lbl}", (x1, oy), 0.45, fc)
        oy += 16

    if pops_info:
        score = pops_info.get("score", 0)
        event = pops_info.get("event", "CLEAR")
        color = pops_info.get("color") or event_color(event)
        outlined_text(im0, f"POPS:{score} {event}", (x1, oy), 0.45, color)


def draw_person_overlay(im0, bbox, speed_status, dir_label, link_label):
    """Draw speed / direction / link info below person bbox."""
    if dir_label == "OUTBOUND" and speed_status in ("MEDIUM", "FAST"):
        tc = (0, 0, 255)
    elif dir_label == "OUTBOUND":
        tc = (0, 165, 255)
    elif dir_label == "INBOUND":
        tc = (200, 200, 0)
    else:
        tc = (200, 200, 200)

    x1 = int(bbox[0])
    oy = int(bbox[3]) + 18

    lines = [speed_status]
    if dir_label != "UNKNOWN":
        lines.append(dir_label)
    if link_label:
        lines.append(link_label)

    for line in lines:
        c = (0, 255, 0) if "->" in line else tc
        outlined_text(im0, line, (x1, oy), 0.45, c)
        oy += 16


def draw_link_lines(im0, links, det_centroids, get_display_id,
                    person_disp_to_raw, cart_disp_to_raw):
    """Draw magenta lines between linked person-cart pairs. Returns count."""
    drawn = set()
    count = 0
    for cart_raw, person_raw in links.items():
        cd = get_display_id('cart', cart_raw)
        pd = get_display_id('person', person_raw)
        if (pd, cd) in drawn:
            continue
        c_raw = cart_raw if cart_raw in det_centroids else cart_disp_to_raw.get(cd)
        p_raw = person_raw if person_raw in det_centroids else person_disp_to_raw.get(pd)
        if c_raw and c_raw in det_centroids and p_raw and p_raw in det_centroids:
            cpt = det_centroids[c_raw]
            ppt = det_centroids[p_raw]
            cv2.line(im0, cpt, ppt, COLOR_LINK, 2, cv2.LINE_AA)
            mx = (cpt[0] + ppt[0]) // 2
            my = (cpt[1] + ppt[1]) // 2
            outlined_text(im0, "Linked", (mx - 25, my - 8), 0.5, COLOR_LINK)
            count += 1
        elif c_raw or p_raw:
            count += 1
        drawn.add((pd, cd))
    return count


def draw_hud(im0, person_count, cart_count, link_count,
             frame_idx, total_frames, w):
    """Draw frame counter and person / cart / link counts."""
    items = [
        (f"Persons: {person_count}", COLOR_PERSON),
        (f"Carts: {cart_count}", COLOR_CART),
        (f"Links: {link_count}", COLOR_LINK),
    ]
    y_off = 30
    for text, color in items:
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.8, 2)
        cv2.rectangle(im0, (10, y_off - th - 5), (10 + tw + 10, y_off + 5), (0, 0, 0), -1)
        cv2.putText(im0, text, (15, y_off), _FONT, 0.8, color, 2, cv2.LINE_AA)
        y_off += th + 20
    outlined_text(im0, f"Frame: {frame_idx}/{total_frames}", (w - 250, 30), 0.6, (255, 255, 255))
