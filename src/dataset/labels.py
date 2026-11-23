"""Behavior vocabulary; the first four are normal (label 0), the last four cheat (label 1)."""

import pandas as pd

from dataset.intervals import marked_fraction

NORMAL_BEHAVIORS = ["Checking_Smartwatch", "Looking_Around", "Thinking", "Writing_Answers"]
CHEAT_BEHAVIORS = ["Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
BEHAVIORS = NORMAL_BEHAVIORS + CHEAT_BEHAVIORS
CHEAT_IDS = list(range(len(NORMAL_BEHAVIORS), len(BEHAVIORS)))
IGNORE_BEHAVIOR = -100  # cross-entropy ignore_index


def window_target(row: pd.Series, start: int, length: int) -> tuple[int, int]:
    """(behavior, label) of the window [start, start + length) of a train video. Test labels follow the behavior
    visible in each sequence (FAQ 33/41), and subjects stay neutral until the action starts: a window whose middle lies
    before the video's `onset` frame (when the videos table has that column) is normal with no behavior target. A video
    with manual `spans` (list of [start, end) frames) is labelled by them instead: cheating when the marked frames are
    the majority of the window (the dominant behaviour, FAQ 44.3/63.1), else normal with no behavior target."""
    spans = row.get("spans")
    if isinstance(spans, list):
        return (int(row.behavior_id), int(row.label)) if marked_fraction(spans, start, length) >= 0.5 else (
            IGNORE_BEHAVIOR, 0)
    if "onset" in row and start + length / 2 < row.onset:
        return IGNORE_BEHAVIOR, 0
    return int(row.behavior_id), int(row.label)