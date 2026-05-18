# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
Motion analysis — speed, direction labels, co-movement detection.

All functions operate on raw position/timestamp history dicts to avoid
coupling to any particular tracker class.
"""
import math
from .config import (
    SPEED_STATIC, SPEED_SLOW, SPEED_MEDIUM,
    COMOVEMENT_MIN_POSITIONS, COMOVEMENT_WINDOW,
    COMOVEMENT_STATIC_PX, COMOVEMENT_COS_THRESH,
    DIRECTION_MIN_POSITIONS, DIRECTION_MIN_DY,
    PLACEMENT_OUTSIDE_FACING_ENTRANCE, PLACEMENT_INSIDE_FACING_EXIT,
    PLACEMENT_INSIDE_EXIT_ON_RIGHT, PLACEMENT_INSIDE_EXIT_ON_LEFT,
    PLACEMENT_INSIDE_EXIT_ON_BOTH,
)


def compute_motion(positions: list, timestamps: list, speeds: list, fps: float):
    """Compute speed, direction angle, speed status, acceleration.

    Returns (speed, direction_deg, status_str, acceleration).
    """
    n_pos = len(positions)
    if n_pos < 2:
        return 0.0, 0.0, "STATIC", 0.0

    n = min(5, n_pos)
    # list(...) so we accept deque too (tracker now stores positions in deques).
    recent = list(positions)[-n:]
    ts     = list(timestamps)[-n:]
    dt = ts[-1] - ts[0]
    if dt < 0.01:
        return 0.0, 0.0, "STATIC", 0.0

    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    dist = math.sqrt(dx * dx + dy * dy)
    speed = dist / dt
    direction = math.degrees(math.atan2(dy, dx)) % 360

    accel = 0.0
    if len(speeds) >= 2 and dt > 0:
        accel = (speeds[-1] - speeds[-2]) * fps

    if speed < SPEED_STATIC:
        status = "STATIC"
    elif speed < SPEED_SLOW:
        status = "SLOW"
    elif speed < SPEED_MEDIUM:
        status = "MEDIUM"
    else:
        status = "FAST"

    return speed, direction, status, accel


def compute_direction_label(positions: list, camera_placement: str) -> str:
    """Determine INBOUND / OUTBOUND / UNKNOWN from position delta."""
    if len(positions) < DIRECTION_MIN_POSITIONS:
        return "UNKNOWN"
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]

    if camera_placement == PLACEMENT_INSIDE_EXIT_ON_RIGHT:
        if abs(dx) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        return "OUTBOUND" if dx > 0 else "INBOUND"
    elif camera_placement == PLACEMENT_INSIDE_EXIT_ON_LEFT:
        if abs(dx) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        return "OUTBOUND" if dx < 0 else "INBOUND"
    elif camera_placement == PLACEMENT_INSIDE_EXIT_ON_BOTH:
        if abs(dx) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        return "OUTBOUND"  # moving left or right toward either exit
    elif camera_placement in (PLACEMENT_OUTSIDE_FACING_ENTRANCE,
                              PLACEMENT_INSIDE_FACING_EXIT):
        if abs(dy) < DIRECTION_MIN_DY:
            return "UNKNOWN"
        if camera_placement == PLACEMENT_OUTSIDE_FACING_ENTRANCE:
            return "OUTBOUND" if dy > 0 else "INBOUND"
        return "OUTBOUND" if dy < 0 else "INBOUND"
    else:
        raise ValueError(f"unknown camera_placement {camera_placement!r}")


def are_co_moving(pos_a: list, pos_b: list) -> bool:
    """Check if two tracked objects share similar velocity direction.

    A bystander standing still while a cart rolls past will return False.
    """
    min_pos = COMOVEMENT_MIN_POSITIONS
    if not pos_a or not pos_b or len(pos_a) < min_pos or len(pos_b) < min_pos:
        return True  # not enough history — allow overlap-only linking

    n = min(COMOVEMENT_WINDOW, len(pos_a), len(pos_b))
    vax = pos_a[-1][0] - pos_a[-n][0]
    vay = pos_a[-1][1] - pos_a[-n][1]
    vbx = pos_b[-1][0] - pos_b[-n][0]
    vby = pos_b[-1][1] - pos_b[-n][1]

    mag_a = math.sqrt(vax * vax + vay * vay)
    mag_b = math.sqrt(vbx * vbx + vby * vby)

    a_static = mag_a < COMOVEMENT_STATIC_PX
    b_static = mag_b < COMOVEMENT_STATIC_PX

    if a_static and b_static:
        return True
    if a_static != b_static:
        return False

    dot = vax * vbx + vay * vby
    cos_sim = dot / (mag_a * mag_b + 1e-9)
    return cos_sim > COMOVEMENT_COS_THRESH
