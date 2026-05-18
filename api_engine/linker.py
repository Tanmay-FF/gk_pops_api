# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Person-cart linking and cart re-identification.

The PersonCartLinker owns all link state and exposes a single update() call
per frame. It does NOT depend on any tracker-specific data structures —
it receives bounding boxes and position histories as plain dicts.
"""
import math
from .config import (
    LINK_CONFIRM_FRAMES, LINK_CONTESTED_FRAMES, LINK_GRACE_FRAMES,
    LINK_DRIFT_FRAMES, STALE_CART_FRAMES, REID_DIST_THRESH, REID_MAX_GONE_FRAMES,
)
from .motion import are_co_moving


# ---------------------------------------------------------------------------
# IoU helper (inlined for speed — called per person per cart per frame)
# ---------------------------------------------------------------------------
def _iou(a, b):
    """Compute IoU between two (x1,y1,x2,y2) boxes. Returns float."""
    ox1 = max(a[0], b[0]); oy1 = max(a[1], b[1])
    ox2 = min(a[2], b[2]); oy2 = min(a[3], b[3])
    if ox2 <= ox1 or oy2 <= oy1:
        return 0.0
    inter = (ox2 - ox1) * (oy2 - oy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------
class PersonCartLinker:
    __slots__ = (
        "_links", "_link_start_frames", "_link_candidates",
        "_perm_persons", "_perm_carts",
        "_person_for_cart", "_person_raw_for_cart",
        "_drift_counter", "_get_display_id",
        "_total_links",
    )

    def __init__(self, get_display_id_fn):
        """get_display_id_fn(label, raw_id) -> display_id"""
        self._get_display_id = get_display_id_fn
        self.reset()

    def reset(self):
        self._links = {}                    # cart_raw -> person_raw
        self._link_start_frames = {}        # cart_raw -> frame_idx
        self._link_candidates = {}          # cart_raw -> (person_raw, count, miss)
        self._perm_persons = set()          # display IDs
        self._perm_carts = set()            # display IDs
        self._person_for_cart = {}          # cart_disp -> person_disp
        self._person_raw_for_cart = {}      # cart_disp -> person_raw
        self._drift_counter = {}            # cart_raw -> frames with zero overlap
        self._total_links = set()           # unique (person_disp, cart_disp) pairs ever linked

    # --- Public properties ---
    @property
    def links(self):
        return self._links

    @property
    def link_start_frames(self):
        return self._link_start_frames

    @property
    def total_links(self):
        return len(self._total_links)

    @property
    def permanently_linked_persons(self):
        return self._perm_persons

    @property
    def permanently_linked_carts(self):
        return self._perm_carts

    @property
    def person_for_cart(self):
        return self._person_for_cart

    @property
    def person_raw_for_cart(self):
        return self._person_raw_for_cart

    # --- Cart re-identification ---
    def try_reidentify_cart(self, new_raw_id, bbox, current_cart_raws,
                            display_map, obj_positions, obj_timestamps,
                            obj_speeds, obj_disappeared):
        """Transfer identity when tracker assigns a new raw ID to same physical cart."""
        cart_map = display_map.get('cart')
        if not cart_map or new_raw_id in cart_map:
            return
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        best_old, best_dist = None, REID_DIST_THRESH
        for old_raw, _disp in cart_map.items():
            if old_raw in current_cart_raws:
                continue
            gone = obj_disappeared.get(old_raw, 999)
            if gone > REID_MAX_GONE_FRAMES:
                continue
            positions = obj_positions.get(old_raw)
            if not positions:
                continue
            lp = positions[-1]
            d = math.sqrt((cx - lp[0]) ** 2 + (cy - lp[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_old = old_raw

        if best_old is None:
            return
        # Transfer display ID
        old_disp = cart_map[best_old]
        cart_map[new_raw_id] = old_disp
        del cart_map[best_old]
        # Transfer link
        if best_old in self._links:
            self._links[new_raw_id] = self._links.pop(best_old)
            self._link_start_frames[new_raw_id] = self._link_start_frames.pop(best_old, 0)
        if best_old in self._link_candidates:
            self._link_candidates[new_raw_id] = self._link_candidates.pop(best_old)
        # Transfer history
        if best_old in obj_positions:
            obj_positions[new_raw_id] = obj_positions.pop(best_old)
        if best_old in obj_timestamps:
            obj_timestamps[new_raw_id] = obj_timestamps.pop(best_old)
        if best_old in obj_speeds:
            obj_speeds[new_raw_id] = obj_speeds.pop(best_old)

    # --- Main per-frame update ---
    def update(self, person_bboxes: dict, cart_bboxes: dict,
               frame_idx: int, obj_disappeared: dict,
               obj_positions: dict, obj_first_frame: dict):
        """Run linking logic for one frame.

        Args:
            person_bboxes: {raw_id: (x1,y1,x2,y2)} for persons in this frame
            cart_bboxes:   {raw_id: (x1,y1,x2,y2)} for carts in this frame
            frame_idx:     current frame number
            obj_disappeared: {raw_id: frames_gone}
            obj_positions:   {raw_id: [(cx,cy), ...]}
            obj_first_frame: {raw_id: first_frame_idx}
        """
        gdi = self._get_display_id

        # Step 0: Purge stale links
        stale = [cid for cid in self._links
                 if cid not in cart_bboxes and obj_disappeared.get(cid, 0) > STALE_CART_FRAMES]
        for cid in stale:
            cd = gdi('cart', cid)
            self._links.pop(cid, None)
            self._link_start_frames.pop(cid, None)
            pd = self._person_for_cart.pop(cd, None)
            self._person_raw_for_cart.pop(cd, None)
            self._perm_carts.discard(cd)
            if pd:
                self._perm_persons.discard(pd)

        # Step 0.5: Drift detection — release link ONLY when the linked
        # person is VISIBLE in the frame but has drifted away from the cart
        # (IoU < 0.05) while someone else is overlapping it.
        # If the linked person has LEFT the frame entirely, that's abandonment
        # — keep the link so POPS abandonment scoring can fire.
        DRIFT_IOU_THRESH = 0.05
        for cart_id, cart_bbox in cart_bboxes.items():
            if cart_id not in self._links:
                self._drift_counter.pop(cart_id, None)
                continue
            pid = self._links[cart_id]

            # Person must be VISIBLE for drift to apply
            if pid not in person_bboxes:
                # Person gone from frame — this is abandonment, NOT drift
                self._drift_counter.pop(cart_id, None)
                continue

            overlap = _iou(cart_bbox, person_bboxes[pid])
            if overlap >= DRIFT_IOU_THRESH:
                self._drift_counter.pop(cart_id, None)  # still engaged
            else:
                # Person is visible but not overlapping the cart.
                # Only count as drift if someone ELSE is overlapping (takeover).
                someone_else = any(
                    _iou(cart_bbox, ob) >= DRIFT_IOU_THRESH
                    for op, ob in person_bboxes.items() if op != pid
                )
                if someone_else:
                    self._drift_counter[cart_id] = self._drift_counter.get(cart_id, 0) + 1
                    if self._drift_counter[cart_id] >= LINK_DRIFT_FRAMES:
                        cd = gdi('cart', cart_id)
                        pd_disp = self._person_for_cart.pop(cd, None)
                        self._person_raw_for_cart.pop(cd, None)
                        self._perm_carts.discard(cd)
                        if pd_disp:
                            self._perm_persons.discard(pd_disp)
                        self._links.pop(cart_id, None)
                        self._link_start_frames.pop(cart_id, None)
                        self._drift_counter.pop(cart_id, None)
                else:
                    self._drift_counter.pop(cart_id, None)

        # Step 1: Handle tracker-ID swaps for linked persons.
        # When the linked person vanishes briefly (tracker swap), find
        # the new person whose centroid is close to the OLD person's last
        # known position.  At ~20fps, an ID swap can take up to ~15 frames
        # (person occluded by cart, re-detected with a new ID).
        # Note: centroid distance, not IoU — we'd need the old bbox plumbed
        # through to gate on IoU, and centroid proximity is a good-enough
        # proxy for "same body reappearing at the same spot."
        PERSON_SWAP_MAX_GONE    = 15  # frames — covers ~0.75s at 20fps
        PERSON_SWAP_MAX_DIST_PX = 80  # max centroid distance for a swap

        claimed = set()
        for cart_id in list(self._links):
            pid = self._links[cart_id]
            if pid in person_bboxes:
                claimed.add(pid)
                continue

            # Linked person gone — check if it's a tracker swap
            person_gone = obj_disappeared.get(pid, 999)
            if person_gone > PERSON_SWAP_MAX_GONE:
                continue  # gone too long — genuine departure, not a swap

            # Get old person's last known bbox from position history
            old_positions = obj_positions.get(pid)
            if not old_positions:
                continue
            # We need the actual bbox, not centroid. Use _obj_bboxes passed via
            # the linker's update signature — but we don't have it here.
            # Instead, compare new person centroids against old person's last centroid.
            # A tracker swap means the new ID appears at nearly the same spot.
            old_cx, old_cy = old_positions[-1]

            best_new_pid = None
            best_dist = PERSON_SWAP_MAX_DIST_PX
            for new_pid, new_bbox in person_bboxes.items():
                if new_pid in claimed:
                    continue
                if new_pid == pid:
                    continue
                new_cx = (new_bbox[0] + new_bbox[2]) * 0.5
                new_cy = (new_bbox[1] + new_bbox[3]) * 0.5
                d = math.sqrt((old_cx - new_cx) ** 2 + (old_cy - new_cy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_new_pid = new_pid

            if best_new_pid is not None:
                cd = gdi('cart', cart_id)
                old_pd = self._person_for_cart.get(cd)
                new_pd = gdi('person', best_new_pid)
                self._links[cart_id] = best_new_pid
                self._person_for_cart[cd] = new_pd
                self._person_raw_for_cart[cd] = best_new_pid
                if old_pd:
                    self._perm_persons.discard(old_pd)
                self._perm_persons.add(new_pd)
                claimed.add(best_new_pid)

        # Step 2: New links for un-linked carts.
        # Accumulate IoU scores for ALL overlapping + co-moving persons over
        # LINK_CONFIRM_FRAMES.  The person with the highest cumulative IoU
        # wins — this ensures the person with the most consistent overlap
        # gets linked, not just whoever appeared first.
        #
        # _link_candidates: cart_raw -> {person_raw: (cumulative_iou, frame_count)}
        for cart_id, cart_bbox in cart_bboxes.items():
            if cart_id in self._links:
                continue
            cd = gdi('cart', cart_id)
            if cd in self._perm_carts:
                continue
            cart_age = frame_idx - obj_first_frame.get(cart_id, frame_idx)
            if cart_age < LINK_GRACE_FRAMES:
                continue

            excluded = set(claimed) | {
                pid for pid in person_bboxes
                if gdi('person', pid) in self._perm_persons
            }

            # Score ALL overlapping + co-moving persons this frame
            candidates = self._link_candidates.get(cart_id, {})
            any_update = False
            for pid, pbbox in person_bboxes.items():
                if pid in excluded:
                    continue
                iou = _iou(cart_bbox, pbbox)
                if iou <= 0:
                    continue
                if not are_co_moving(obj_positions.get(cart_id),
                                     obj_positions.get(pid)):
                    continue
                prev_iou, prev_count = candidates.get(pid, (0.0, 0))
                candidates[pid] = (prev_iou + iou, prev_count + 1)
                any_update = True

            if any_update:
                self._link_candidates[cart_id] = candidates

                # Adaptive threshold: if only 1 candidate ever seen, use
                # fast confirmation (LINK_CONFIRM_FRAMES = 6).
                # If 2+ candidates are competing, use the longer
                # LINK_CONTESTED_FRAMES (20) to give the real pusher time.
                n_total_candidates = len(candidates)
                threshold = LINK_CONTESTED_FRAMES if n_total_candidates >= 2 else LINK_CONFIRM_FRAMES

                qualified = {pid: cum_iou for pid, (cum_iou, count)
                             in candidates.items() if count >= threshold}

                # Tiebreaker: if 2+ qualified candidates, apply "behind the cart"
                # bonus.  The person pushing is behind the cart relative to its
                # direction of movement.  We use the cart's position history to
                # determine which direction it's going, then favour the person
                # whose centroid is on the trailing side.
                if len(qualified) >= 2:
                    cart_positions = obj_positions.get(cart_id, [])
                    if len(cart_positions) >= 5:
                        # Cart movement vector (recent)
                        dx = cart_positions[-1][0] - cart_positions[-5][0]
                        dy = cart_positions[-1][1] - cart_positions[-5][1]
                        cart_cx = (cart_bbox[0] + cart_bbox[2]) * 0.5
                        cart_cy = (cart_bbox[1] + cart_bbox[3]) * 0.5
                        mag = math.sqrt(dx * dx + dy * dy)
                        if mag > 5:  # cart is actually moving
                            # Normalised movement direction
                            ndx, ndy = dx / mag, dy / mag
                            for pid in qualified:
                                pb = person_bboxes.get(pid)
                                if pb is None:
                                    continue
                                pcx = (pb[0] + pb[2]) * 0.5
                                pcy = (pb[1] + pb[3]) * 0.5
                                # Vector from cart center to person center
                                vx = pcx - cart_cx
                                vy = pcy - cart_cy
                                # Dot product with movement direction:
                                # negative = person is BEHIND the cart (trailing)
                                # positive = person is IN FRONT (leading)
                                dot = vx * ndx + vy * ndy
                                if dot < 0:
                                    # Person is behind — bonus 50% of their cum IoU
                                    qualified[pid] *= 1.5

                best_pid, best_score = None, 0.0
                for pid, score in qualified.items():
                    if score > best_score:
                        best_score = score
                        best_pid = pid

                if best_pid is not None:
                    self._links[cart_id] = best_pid
                    self._link_start_frames[cart_id] = frame_idx
                    claimed.add(best_pid)
                    self._link_candidates.pop(cart_id, None)
                    pd = gdi('person', best_pid)
                    self._perm_persons.add(pd)
                    self._perm_carts.add(cd)
                    self._total_links.add((pd, cd))
                    self._person_for_cart[cd] = pd
                    self._person_raw_for_cart[cd] = best_pid
            else:
                self._link_candidates.pop(cart_id, None)
