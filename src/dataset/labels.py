"""Behavior vocabulary; the first four are normal (label 0), the last four cheat (label 1)."""

NORMAL_BEHAVIORS = ["Checking_Smartwatch", "Looking_Around", "Thinking", "Writing_Answers"]
CHEAT_BEHAVIORS = ["Looking_Under_the_Desk", "Reading_Hidden_Notes", "Repeated_Peeking", "Using_Smartwatch"]
BEHAVIORS = NORMAL_BEHAVIORS + CHEAT_BEHAVIORS
CHEAT_IDS = list(range(len(NORMAL_BEHAVIORS), len(BEHAVIORS)))