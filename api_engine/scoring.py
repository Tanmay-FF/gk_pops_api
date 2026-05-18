# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""
POPS (Push-Out Probability Score) computation and event classification.
"""

# ---------------------------------------------------------------------------
# Score ranges:
#   0-30:  Low Priority  — normal shopping, inbound, employees
#   31-70: Medium Priority — needs quick verify
#   71-100: High Priority — likely theft / pushout
#
# A. Kill Switches:
#   No cart detected → 0,  INBOUND → 5,  UNCLEAR → 5
#
# B. Threat Indicators (additive, OUTBOUND):
#   Base +15, EMPTY -15, PARTIAL +15,
#   FULL+BAGGED +20, FULL+UNBAGGED +45,
#   FAST +15, MEDIUM +5
#
# C. Linked damping (UNKNOWN direction only):
#   If a person is WITH the cart (linked) and direction is UNKNOWN
#   (shopping inside store), subtract 20.
#   Does NOT apply to OUTBOUND — a linked person pushing a cart out
#   the exit is exactly the pushout we want to detect.
#
# D. Abandonment (strongest signal — overrides damping):
#   Floor at 75 if outbound + merchandise (partial/full)
#   Floor at 60 if outbound + empty
#   Otherwise +35
# ---------------------------------------------------------------------------

_FILL_SCORE_OUTBOUND = {"empty": -15}
_FILL_SCORE_UNKNOWN  = {"partial": 8}
_SPEED_SCORE_OUTBOUND = {"FAST": 15, "MEDIUM": 5}
_SPEED_SCORE_UNKNOWN  = {"FAST": 8}

# Linked person is WITH the cart — dampen risk score
_LINKED_DAMPING = 20


def compute_pops(direction_label: str, speed_status: str, is_valid: bool,
                 fill_label: str, bag_label: str = "not_applicable",
                 cart_detected: bool = True, abandoned: bool = False,
                 linked: bool = False) -> int:
    """Compute Push-Out Probability Score (0-100).

    A linked person pushing their cart through the store is normal — the score
    is dampened by 20 points.  Only abandonment (person disappeared) or no link
    at all can push the score into HIGH PRIORITY territory.
    """
    # --- Kill Switches ---
    if not cart_detected:
        return 0
    if direction_label == "INBOUND":
        return 5
    if not is_valid:
        return 5

    score = 0

    if direction_label == "OUTBOUND":
        score += 15  # Base outbound
        # Contents — unbagged adds extra risk at every fill level
        if fill_label == "full":
            score += 50 if bag_label == "unbagged" else 20
        elif fill_label == "partial":
            score += 30 if bag_label == "unbagged" else 15
        else:
            score += _FILL_SCORE_OUTBOUND.get(fill_label, 0)
        # Velocity
        score += _SPEED_SCORE_OUTBOUND.get(speed_status, 0)
        # Combo: rushing with loose items is the classic pushout pattern
        if speed_status == "FAST" and bag_label == "unbagged" and fill_label in ("partial", "full"):
            score += 15
        # NO linked damping for OUTBOUND — a person pushing a cart out
        # the exit IS the scenario we want to catch.  Damping only applies
        # to UNKNOWN direction (shopping inside the store).
        # Abandonment — overrides everything, floor the score high
        if abandoned:
            if fill_label in ("partial", "full"):
                score = max(score, 75)
            elif fill_label == "empty":
                score = max(score, 60)
            else:
                score += 35

    elif direction_label == "UNKNOWN":
        if fill_label == "full":
            score += 25 if bag_label == "unbagged" else 15
        else:
            score += _FILL_SCORE_UNKNOWN.get(fill_label, 0)
        score += _SPEED_SCORE_UNKNOWN.get(speed_status, 0)
        if linked and not abandoned:
            score -= _LINKED_DAMPING
        if abandoned:
            if fill_label in ("partial", "full"):
                score = max(score, 65)
            else:
                score += 25

    # Cap: partial + bagged is low-risk (items are paid for)
    if fill_label == "partial" and bag_label == "bagged":
        score = min(score, 55)

    # Clamp
    if score < 0:
        return 0
    return score if score <= 100 else 100


def classify_event(pops_score: int, linked: bool,
                   direction_label: str, abandoned: bool = False) -> str:
    """Return the event name string based on POPS score + context."""
    if pops_score >= 71:
        return "PUSHOUT ALERT" if abandoned else "HIGH PRIORITY"

    if pops_score >= 31:
        if abandoned:
            return "ABANDONED CART"
        if not linked and direction_label == "OUTBOUND":
            return "UNLINKED EXIT"
        return "MEDIUM PRIORITY"

    if direction_label == "INBOUND":
        return "INBOUND"
    if not linked and direction_label == "OUTBOUND":
        return "UNLINKED EXIT"
    if linked:
        return "MONITORING"
    return "LOW PRIORITY"


# Event names that trigger logging
LOGGABLE_EVENTS = frozenset({
    "PUSHOUT ALERT", "HIGH PRIORITY", "MEDIUM PRIORITY",
    "UNLINKED EXIT", "ABANDONED CART",
})

HIGH_EVENTS = frozenset({"PUSHOUT ALERT", "HIGH PRIORITY"})
