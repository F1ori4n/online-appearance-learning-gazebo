from __future__ import annotations

from enum import Enum


class ExperimentMode(str, Enum):
    """
    Strategy for the experiment mode.
    - TRACKER_ONLY: Only use Tracker
    - ONLINE_REID: Use ReID with one crop for initialization
    - CALIBRATION_REID: Use Reid with multiple crops for initialization
    """
    TRACKER_ONLY = "tracker_only"
    ONLINE_REID = "online_reid"
    CALIBRATION_REID = "calibration_reid"


class TargetState(str, Enum):
    """
    The current state of the target manager.
    Used the for follow, search behavior and for logging
    """
    UNINITIALIZED = "uninitialized"
    CALIBRATING = "calibrating"
    TRACKING = "tracking"
    TRACKING_LOW_CONFIDENCE = "tracking_low_confidence"
    REID_VERIFYING = "reid_verifying"
    TARGET_LOST = "target_lost"
    TARGET_REFOUND = "target_refound"
