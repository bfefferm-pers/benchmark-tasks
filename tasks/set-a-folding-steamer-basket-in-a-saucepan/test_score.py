"""Scorer unit tests: synthetic object states built from the steamer's and pan's own geometry, no simulator. Created
11_17_26 (America/New_York).

Run: python -m pytest test_score.py (plain Python with NumPy and the yam_arena package importable)."""

import math

import numpy as np
import pytest

import task as T
from steamer_basket import mech as M
from yam_arena.task_file import ObjectState

DT = 0.05
FAR_L = (0.15, 0.32, T.TOP + 0.16)  # parked hands
FAR_R = (0.15, -0.32, T.TOP + 0.16)
UP = M.HANDLE_TRAVEL
WALL_Q = None  # filled below: the leaf angle that brings a tip to the pan's wall


def _st(p, q=(0.0, 0.0, 0.0, 1.0), low=None, high=None):
    p = np.asarray(p, float)
    z = np.zeros(3)
    return ObjectState(pos=p, quat_xyzw=np.asarray(q, float), lin_vel=z, ang_vel=z,
                       low=p if low is None else np.asarray(low, float), high=p if high is None else np.asarray(high, float))


def _qaxis(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(a / 2)
    return np.array((axis[0] * s, axis[1] * s, axis[2] * s, math.cos(a / 2)))


def _wall_angle():
    """Joint angle at which a leaf standing in the pan touches the wall."""
    for q in np.linspace(0.0, math.radians(M.FOLD_MAX_DEG), 2000):
        if all(math.hypot(c[0], c[1]) <= M.pan_inner_r(M.PAN_FLOOR + M.LEG_H + c[2]) for c in M.leaf_tip_corners(0, q)):
            return float(q)
    raise AssertionError("no wall angle")


WALL_Q = _wall_angle()


def objects(plate_xy=None, plate_h=M.LEG_H, tilt_deg=0.0, slide=0.0, angles=None, left=FAR_L, right=FAR_R,
            pan_dxy=(0.0, 0.0), pan_tilt_deg=0.0, on_table=False):
    """A state: the steamer's plate underside centre at plate_xy (world; default the pan's axis) and plate_h above the
    pan's inside floor (or above the table if on_table), tilted about world x; leaves at `angles` (default: all at the
    wall); handle at `slide`; the pan shifted and tilted about world y."""
    pan_xy = T.PAN_XY + np.asarray(pan_dxy, float)
    pan_q = _qaxis((0, 1, 0), math.radians(pan_tilt_deg))
    pan_pos = np.array((pan_xy[0], pan_xy[1], T.TOP))
    xy = pan_xy if plate_xy is None else np.asarray(plate_xy, float)
    z = T.TOP + plate_h + (0.0 if on_table else M.PAN_FLOOR)
    q = _qaxis((1, 0, 0), math.radians(tilt_deg))
    pos = np.array((xy[0], xy[1], z))
    r = T._rot(q)
    a = np.full(M.N_LEAVES, WALL_Q) if angles is None else np.asarray(angles, float)
    tips = np.array([pos + r @ c for k in range(M.N_LEAVES) for c in M.leaf_tip_corners(k, a[k])])
    top = pos + r @ np.array((0.0, 0.0, M.tbar_top(slide)))
    return {
        "left_tcp": _st(left), "right_tcp": _st(right),
        "steamer": _st(pos, q),
        "leaves": _st(pos, q, low=a, high=tips),
        "handle": _st(top, q, low=(slide, 0.0, 0.0), high=top),
        "pan": _st(pan_pos, pan_q),
    }


def run(seq):
    """Feed (objects, seconds) pairs through note_step; return the final score of the last state."""
    T.reset_latch()
    last = None
    for ob, secs in seq:
        for _ in range(max(1, round(secs / DT))):
            T.note_step(ob, DT)
        last = ob
    return T.score(last)


def test_geometry_matches_published_sizes():
    assert M.open_diameter() == pytest.approx(M.OPEN_D, abs=0.003)
    assert M.closed_diameter() == pytest.approx(M.CLOSED_D, abs=0.002)
    assert M.closed_diameter() < M.PAN_TOP_D  # the folded steamer passes the rim ...
    assert M.open_diameter() > M.PAN_TOP_D  # ... the open one does not
    assert sum(M.MASS.values()) - M.MASS["leaf"] + M.N_LEAVES * M.MASS["leaf"] == pytest.approx(M.MASS_TOTAL, abs=1e-6)


def _sat_gap(a, b):
    """Separating-axis gap (m) between two convex slabs given by their 8 corners (leaf_local_points order): > 0 means
    they do not touch."""
    def dirs(p):
        return [p[4] - p[0], p[6] - p[2], p[2] - p[0], p[1] - p[0]]  # two side edges, the hinge edge, the thickness
    axes = [np.cross(u, v) for u in dirs(a) for v in dirs(a)] + [np.cross(u, v) for u in dirs(b) for v in dirs(b)]
    axes += [np.cross(u, v) for u in dirs(a) for v in dirs(b)]
    best = -1.0
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        pa, pb = a @ (ax / n), b @ (ax / n)
        best = max(best, pb.min() - pa.max(), pa.min() - pb.max())
    return best


def test_leaves_fold_closed_without_touching():
    """Folded together, neighbouring (twisted) leaves never meet from open to fully closed, so a uniform fold is free;
    they overlap seen from above, so the open basket has no gaps between leaves."""
    for deg in np.linspace(0.0, M.FOLD_MAX_DEG, 18):
        q = math.radians(deg)
        assert _sat_gap(M.leaf_points(0, q), M.leaf_points(1, q)) > 0.0
        assert _sat_gap(M.leaf_points(0, q), M.leaf_points(2, q)) > 0.005
    a0 = 2 * math.pi / M.N_LEAVES
    tips = M.leaf_tip_corners(0, 0.0)
    assert max(math.atan2(p[1], p[0]) for p in tips) > a0 / 2  # leaf 0's leading tip corner passes the halfway line
    tips1 = M.leaf_tip_corners(1, 0.0)
    assert min(math.atan2(p[1], p[0]) for p in tips1) < a0 / 2  # and leaf 1's trailing corner comes back past it


def test_handle_heights_against_the_rim():
    plate = M.PAN_FLOOR + M.LEG_H
    assert M.PAN_H - (plate + M.tbar_top(0.0)) > T.HANDLE_BELOW + 0.015  # collapsed: well below the rim
    assert plate + M.tbar_top(UP) > M.PAN_H  # extended: above the rim


def test_full_score_set_in_and_handle_down():
    assert run([(objects(), 2.0)]) == 1.0


def test_doing_nothing_scores_zero():
    start = objects(plate_xy=T.STEAMER_XY, slide=UP, angles=np.zeros(M.N_LEAVES), on_table=True)
    assert run([(start, 60.0)]) == 0.0


def test_handle_left_up_is_half():
    assert run([(objects(slide=UP), 2.0)]) == 0.5


def test_handle_partly_down_above_margin_is_half():
    # T-bar top 3 mm below the rim: short of the 5 mm margin
    plate = M.PAN_FLOOR + M.LEG_H
    s = M.PAN_H - 0.003 - plate - M.HANDLE_DOWN
    assert 0.0 < s < UP
    assert run([(objects(slide=s), 2.0)]) == 0.5


UPRIGHT = np.full(M.N_LEAVES, math.radians(80.0))  # leaves folded nearly upright: tips inside the wall when tilted or shifted


def test_tilt_bands():
    assert run([(objects(tilt_deg=4.0, angles=UPRIGHT), 2.0)]) == 1.0
    assert run([(objects(tilt_deg=10.0, angles=UPRIGHT), 2.0)]) == 0.5
    assert run([(objects(tilt_deg=20.0, angles=UPRIGHT), 2.0)]) == 0.25


def test_leaf_over_the_rim_is_quarter():
    a = np.full(M.N_LEAVES, WALL_Q)
    a[3] = 0.0  # one leaf still open, lying out over the rim
    ob = objects(plate_h=M.LEG_H + 0.080, angles=a)  # perched high enough for that leaf to clear the wall
    assert not T.leaves_inside(ob)
    assert run([(ob, 2.0)]) == 0.25


def test_open_leaf_inside_but_above_rim_is_quarter():
    a = np.full(M.N_LEAVES, WALL_Q)
    a[0] = math.radians(M.FOLD_MAX_DEG)
    ob = objects(plate_h=M.LEG_H + 0.050, angles=a)
    assert run([(ob, 2.0)]) == 0.25  # perched 50 mm up: not standing on its feet


def test_perched_on_the_rim_is_quarter():
    ob = objects(plate_h=M.PAN_H - M.PAN_FLOOR - 0.010, angles=np.full(M.N_LEAVES, math.radians(70.0)))
    assert run([(ob, 2.0)]) == 0.25


def test_off_centre_beyond_tolerance_is_quarter():
    # (with the 168 mm plate in a pan 185 mm across at its height, a plate 17 mm off the axis could not stand on the
    # floor; the synthetic state tests the centre tolerance alone)
    ob = objects(plate_xy=T.PAN_XY + (0.0, 0.017), angles=np.full(M.N_LEAVES, math.radians(M.FOLD_MAX_DEG)))
    assert run([(ob, 2.0)]) == 0.25
    ok = objects(plate_xy=T.PAN_XY + (0.0, 0.008), angles=np.full(M.N_LEAVES, math.radians(M.FOLD_MAX_DEG)))
    assert T.leaves_inside(ok)
    assert run([(ok, 2.0)]) == 1.0


def test_beside_the_pan_scores_zero():
    ob = objects(plate_xy=T.STEAMER_XY, angles=np.zeros(M.N_LEAVES), on_table=True)
    assert run([(ob, 2.0)]) == 0.0


def test_upside_down_in_pan_is_quarter():
    ob = objects(plate_h=0.030, tilt_deg=180.0)
    assert run([(ob, 2.0)]) == 0.25


def test_pan_moved_or_tipped_scores_zero():
    assert run([(objects(pan_dxy=(0.035, 0.0)), 2.0)]) == 0.0
    assert run([(objects(pan_dxy=(0.025, 0.0)), 2.0)]) == 1.0
    assert run([(objects(pan_tilt_deg=8.0), 2.0)]) == 0.0


def test_hand_still_on_the_handle_scores_zero():
    ob = objects()
    on_bar = np.asarray(ob["handle"].high, float) + (0.0, 0.0, -0.002)
    assert run([(objects(left=on_bar), 5.0)]) == 0.0


def test_hand_inside_the_pan_scores_zero():
    pan_mid = (T.PAN_XY[0] + 0.06, T.PAN_XY[1], T.TOP + 0.12)
    assert run([(objects(right=pan_mid), 5.0)]) == 0.0


def test_hands_must_be_clear_for_the_last_second():
    ob = objects()
    near = objects(left=np.asarray(ob["handle"].high, float))
    assert run([(near, 3.0), (ob, 0.5)]) == 0.0
    assert run([(near, 3.0), (ob, 1.0)]) == 1.0


def test_transient_events_do_not_zero_a_good_final_state():
    # a leaf over the rim, a tilt and a hand on the handle earlier in the trial cost nothing if the end state is right
    a = np.zeros(M.N_LEAVES)
    early = objects(plate_h=M.LEG_H + 0.090, tilt_deg=12.0, angles=a, slide=UP)
    assert run([(early, 5.0), (objects(), 1.5)]) == 1.0


def test_score_is_a_listed_band():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ob = objects(plate_xy=T.PAN_XY + rng.uniform(-0.12, 0.12, 2), plate_h=rng.uniform(0.0, 0.15),
                     tilt_deg=rng.uniform(0, 30), slide=rng.uniform(0, UP),
                     angles=rng.uniform(0, math.radians(M.FOLD_MAX_DEG), M.N_LEAVES))
        assert run([(ob, 1.2)]) in (0.0, 0.25, 0.5, 1.0)


def test_success_only_at_full_score():
    T.reset_latch()
    ob = objects()
    for _ in range(40):
        T.note_step(ob, DT)
    assert T.success(ob)
    assert not T.success(objects(slide=UP))
