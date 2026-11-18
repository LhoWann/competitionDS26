"""Action onset per train video, read off out-of-fold cheat probabilities.

Every recording starts with the subject sitting neutral; the scripted action begins around second 7-9. Out-of-fold
p_cheat of a cheat video stays flat over that neutral start and rises when the action appears, so the onset is the
first second (from `search_from`) reaching half of the rise from the opening baseline to the later peak. Weak curves
borrow the onset of the simultaneous other-camera recording, else take `default`; normal videos take `default` (their
behavior also starts there, which only matters for the behavior target).
"""

import numpy as np
import pandas as pd


def curve_onset(p: np.ndarray, base_secs: int = 5, search_from: int = 5) -> tuple[int, float]:
    """(onset second, rise) of one video's per-second p_cheat curve."""
    base, peak = p[:base_secs].mean(), p[search_from:].max()
    hit = p[search_from:] >= base + 0.5 * (peak - base)
    return int(np.argmax(hit)) + search_from, float(peak - base)


def estimate_onsets(oof: pd.DataFrame, clip_len: int = 30, default: int = 8, min_rise: float = 0.15,
                    bounds: tuple[int, int] = (6, 10)) -> pd.DataFrame:
    """video_id -> onset frame from an OOF frame with `video_id`, `start`, `label`, `p_cheat` (one row per window)."""
    est = {}
    for vid, g in oof.sort_values("start").groupby("video_id"):
        sec, rise = curve_onset(g.p_cheat.to_numpy())
        est[vid] = (int(g.label.max()), sec, rise)
    out = []
    for vid, (label, sec, rise) in est.items():
        other = vid.replace("_front_", "_side_") if "_front_" in vid else vid.replace("_side_", "_front_")
        if not label:
            sec = default
        elif rise < min_rise:
            sec = est[other][1] if other in est and est[other][2] >= min_rise else default
        out.append((vid, int(np.clip(sec, *bounds)) * clip_len))
    return pd.DataFrame(out, columns=["video_id", "onset"])