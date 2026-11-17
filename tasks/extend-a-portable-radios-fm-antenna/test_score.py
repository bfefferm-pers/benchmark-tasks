"""Scorer unit tests: synthetic object states built from the radio's own geometry, no simulator. Created 11_16_26.

Run: python -m pytest test_score.py (plain Python with NumPy and the yam_arena package importable)."""

import math

import numpy as np
import pytest

import task as T
from radio_antenna import mech as M
from yam_arena.task_file import ObjectState

DT = 0.05
FAR_L = (0.10, 0.40, T.TOP + 0.20)  # parked hands, well away from the radio
FAR_R = (0.10, -0.40, T.TOP + 0.20)
FULL = [M.TRAVEL] * 4


def _st(p, q=(0.0, 0.0, 0.0, 1.0), low=None, high=None):
    p = np.asarray(p, float)
    z = np.zeros(3)
    return ObjectState(pos=p, quat_xyzw=np.asarray(q, float), lin_vel=z, ang_vel=z,
                       low=p if low is None else np.asarray(low, float), high=p if high is None else np.asarray(high, float))


def _qy(a):
    return np.array((0.0, math.sin(a / 2), 0.0, math.cos(a / 2)))


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz))


def objects(theta=0.0, slides=FULL, left=FAR_L, right=FAR_R, radio_dxy=(0.0, 0.0), radio_tilt_deg=0.0):
    """A state with the antenna at swivel angle theta and the given slides; the radio optionally shifted and tilted
    (about its y axis, so the antenna tilts with it)."""
    qr = _qy(math.radians(radio_tilt_deg))
    rr = T._rot(qr)
    base = np.array((T.RADIO_XY[0] + radio_dxy[0], T.RADIO_XY[1] + radio_dxy[1], T.TOP))
    pivot = base + rr @ M.PIVOT
    qa = _qmul(qr, _qy(theta))
    length = M.antenna_length(slides)
    tip = pivot + T._rot(qa) @ np.array((0.0, 0.0, length))
    s = list(slides)
    return {
        "left_tcp": _st(left), "right_tcp": _st(right),
        "radio": _st(base, qr),
        "antenna": _st(pivot, qa, low=(theta, length, 0.0), high=tip),
        "slides": _st(s[:3], low=(s[3], 0.0, 0.0), high=(s[3], 0.0, 0.0)),
    }


def run(seq):
    """Feed (objects, seconds) pairs through note_step; return the final score."""
    T.reset_latch()
    last = None
    for obj, secs in seq:
        for _ in range(int(round(secs / DT))):
            T.note_step(obj, DT)
            last = obj
    return T.score(last)


def slides_for(length):
    """Slides (outer joint first) that give this pivot-to-tip length."""
    ext = length - M.RETRACTED
    out = []
    for _ in range(4):
        d = min(M.TRAVEL, max(0.0, ext))
        out.append(d)
        ext -= d
    return out


def test_published_lengths():
    assert M.antenna_length([0.0] * 4) == pytest.approx(0.111)
    assert M.antenna_length(FULL) == pytest.approx(0.432)
    assert M.OVERLAP == pytest.approx(0.01975)


def test_start_scores_zero_doing_nothing():
    assert run([(objects(T.START_THETA, [0.0] * 4), 10.0)]) == 0.0


def test_goal_scores_full():
    assert run([(objects(0.0, FULL), 2.0)]) == 1.0


@pytest.mark.parametrize("length_mm,upright,tilted", [(111, 0.40, 0.0), (187, 0.40, 0.0), (189, 0.55, 0.15), (268, 0.55, 0.15),
                                                      (270, 0.70, 0.30), (348, 0.70, 0.30), (350, 0.85, 0.45),
                                                      (428, 0.85, 0.45), (429.5, 1.0, 0.60), (432, 1.0, 0.60)])
def test_length_bands(length_mm, upright, tilted):
    sl = slides_for(length_mm / 1000.0)
    assert run([(objects(0.0, sl), 2.0)]) == pytest.approx(upright)
    assert run([(objects(math.radians(30.0), sl), 2.0)]) == pytest.approx(tilted)


def test_length_is_what_counts_not_which_section():
    # two sections half out give the same length, and the same credit, as one section all the way out
    one = [M.TRAVEL, 0.0, 0.0, 0.0]
    halves = [M.TRAVEL / 2, M.TRAVEL / 2, 0.0, 0.0]
    assert run([(objects(0.0, one), 2.0)]) == run([(objects(0.0, halves), 2.0)]) == pytest.approx(0.55)


@pytest.mark.parametrize("deg,expect", [(0.0, 1.0), (-14.9, 1.0), (14.9, 1.0), (15.1, 0.6), (-15.1, 0.6), (45.0, 0.6), (90.0, 0.6)])
def test_upright_band(deg, expect):
    assert run([(objects(math.radians(deg), FULL), 2.0)]) == pytest.approx(expect)


def test_collapsed_but_upright_scores_upright_only():
    assert run([(objects(0.0, [0.0] * 4), 2.0)]) == pytest.approx(0.4)


@pytest.mark.parametrize("tilt,expect", [(9.0, 1.0), (11.0, 0.0), (90.0, 0.0)])
def test_radio_tipped(tilt, expect):
    # the radio leans tilt degrees with the antenna standing vertical in the world (swivel at -tilt)
    assert run([(objects(math.radians(-tilt), FULL, radio_tilt_deg=tilt), 2.0)]) == pytest.approx(expect)


def test_radio_lying_on_its_back_with_antenna_vertical_scores_zero():
    assert run([(objects(math.radians(-90.0), FULL, radio_tilt_deg=90.0), 2.0)]) == 0.0


@pytest.mark.parametrize("d_mm,expect", [(0, 1.0), (29, 1.0), (31, 0.5), (99, 0.5), (101, 0.0), (300, 0.0)])
def test_radio_moved(d_mm, expect):
    assert run([(objects(0.0, FULL, radio_dxy=(d_mm / 1000.0, 0.0)), 2.0)]) == pytest.approx(expect)


def test_moved_cap_applies_to_partial_credit_too():
    assert run([(objects(0.0, [0.0] * 4, radio_dxy=(0.05, 0.0)), 2.0)]) == pytest.approx(0.4)


def test_hand_still_on_the_knob_scores_zero():
    p, _ = T.right_pinch(0.0, (M.N_SECTIONS - 1) * M.TRAVEL)
    assert run([(objects(0.0, FULL, right=p), 5.0)]) == 0.0


def test_hand_still_holding_the_radio_scores_zero():
    hold = T.world((0.0, T.HOLD_TIP_Y, T.HOLD_Z)) - 0.013 * T.LEFT_TIPS
    assert run([(objects(0.0, FULL, left=hold), 5.0)]) == 0.0


def test_hand_beside_the_middle_of_the_antenna_is_not_clear():
    mid = T.world(M.PIVOT + 0.2 * M.axis_dir(0.0)) + np.array((0.0, -0.03, 0.0))
    assert run([(objects(0.0, FULL, right=mid), 5.0)]) == 0.0


def test_needs_the_last_second_clear():
    p, _ = T.right_pinch(0.0, (M.N_SECTIONS - 1) * M.TRAVEL)
    assert run([(objects(0.0, FULL, right=p), 2.0), (objects(0.0, FULL), 0.9)]) == 0.0
    assert run([(objects(0.0, FULL, right=p), 2.0), (objects(0.0, FULL), 1.1)]) == 1.0


def test_hand_coming_back_restarts_the_second():
    p, _ = T.right_pinch(0.0, (M.N_SECTIONS - 1) * M.TRAVEL)
    assert run([(objects(0.0, FULL), 3.0), (objects(0.0, FULL, right=p), 0.1), (objects(0.0, FULL), 0.5)]) == 0.0


def test_transient_wobble_that_settles_scores_full():
    # mid-trial the radio rocks 20 degrees and slides 50 mm and the antenna swings 30 degrees; all settle back:
    # only the final state counts, as in the real trial
    seq = [(objects(0.0, FULL), 0.5), (objects(math.radians(30.0), FULL, radio_dxy=(0.05, 0.0), radio_tilt_deg=20.0), 0.5),
           (objects(0.0, FULL), 2.0)]
    assert run(seq) == 1.0


def test_wobble_that_does_not_settle_is_judged_where_it_ends():
    seq = [(objects(0.0, FULL), 2.0), (objects(math.radians(20.0), FULL), 2.0)]
    assert run(seq) == pytest.approx(0.6)


def test_credit_never_exceeds_one_and_parts_sum():
    assert T.LENGTH_CREDIT * len(T.LENGTH_STEPS) + T.UPRIGHT_CREDIT == pytest.approx(1.0)
    assert [round(1000 * s, 1) for s in T.LENGTH_STEPS] == [188.2, 268.5, 348.8, 429.0]
