"""Scorer tests (pure Python; yam_arena must be importable). Created 11_16_26 (America/New_York).

    python -m pytest -q test_score.py

The water valve, the head's stall-and-reverse bookkeeping, the sweep's ends and every scoring band, fed with synthetic
states: the hands, and the head as a heading and a spin."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import task as T
from sprinkler_arc import mech as M

DT = 0.05
FAR = (0.12, 0.34, T.TOP + 0.18)  # a parked hand
NEAR = (M.AXIS_X + 0.05, 0.0, T.HEAD_Z + 0.02)  # a hand at the collars


def state(pos):
    return SimpleNamespace(pos=np.asarray(pos, float), quat_xyzw=np.array((0.0, 0.0, 0.0, 1.0)), ang_vel=np.zeros(3))


def head(deg, spin_deg_s=0.0):
    a = math.radians(deg)
    return SimpleNamespace(pos=np.array((M.AXIS_X, M.AXIS_Y, 0.0)), quat_xyzw=np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2))),
                           ang_vel=np.array((0.0, 0.0, math.radians(spin_deg_s))))


def objs(left=FAR, right=(FAR[0], -FAR[1], FAR[2]), h=0.0, w=0.0):
    return {"left_tcp": state(left), "right_tcp": state(right), "head": head(h, w), "lower": head(0.0), "upper": head(0.0)}


@pytest.fixture(autouse=True)
def fresh():
    T.reset_latch()
    yield
    T.reset_latch()


def open_water():
    """A hand reaches in, then both are parked and still for 2 s: the valve opens."""
    T.note_step(objs(left=NEAR), DT)
    for _ in range(int(T.OPEN_AFTER / DT) + 1):
        T.note_step(objs(), DT)
    assert T._trial["water"]


def sweep(ends, cycles=2, speed=M.DRIVE_SPEED):
    """Drive the bookkeeping through `cycles` full back-and-forth sweeps between ends=(cw, ccw) headings."""
    cw, ccw = ends
    h = cw + 1.0
    target = ccw
    for _ in range(cycles * 2 + 1):
        while (target - h) * (1 if target > h else -1) > 0.5:
            step = math.copysign(speed * DT, target - h)
            h += step
            T.note_step(objs(h=h, w=step / DT), DT)
        for _ in range(int(T.STALL_S / DT) + 1):  # stalled against the tab until the bookkeeping reverses it
            T.note_step(objs(h=target, w=0.0), DT)
        target = cw if target == ccw else ccw


def test_doing_nothing_never_opens_the_water():
    for _ in range(400):
        T.note_step(objs(), DT)
    assert not T._trial["water"]
    assert T.score(objs()) == 0.0
    assert T.drive_command() == (0.0, 0.0)


def test_valve_needs_two_still_clear_seconds():
    T.note_step(objs(left=NEAR), DT)
    for _ in range(int(T.OPEN_AFTER / DT) - 2):
        T.note_step(objs(), DT)
    assert not T._trial["water"]
    for _ in range(3):
        T.note_step(objs(), DT)
    assert T._trial["water"]
    assert T.drive_command() == (M.DRIVE_SPEED, M.DRIVE_TORQUE)


def test_a_moving_hand_keeps_the_valve_shut():
    T.note_step(objs(left=NEAR), DT)
    for k in range(80):
        x = FAR[0] + 0.005 * (k % 2)  # 5 mm a step: not still
        T.note_step(objs(left=(x, FAR[1], FAR[2])), DT)
    assert not T._trial["water"]


def test_hand_above_the_head_counts_as_clear():
    above = (M.AXIS_X, 0.0, T.HEAD_Z + M.HEAD_TOP + T.CLEAR_ABOVE + 0.001)
    assert T.hand_clear(objs(left=above), "left_tcp")
    low = (M.AXIS_X + T.CLEAR_R - 0.01, 0.0, T.HEAD_Z + 0.05)
    assert not T.hand_clear(objs(left=low), "left_tcp")


def test_full_score_for_the_flags_arc():
    open_water()
    sweep((-45.0, 45.0))
    assert T.sweep_ends() == pytest.approx((-45.0, 45.0), abs=0.6)
    assert T.score(objs()) == 1.0
    assert T.success(objs())


def test_no_score_until_each_way_has_reversed():
    open_water()
    h = 0.0
    while h < 44.5:  # the first leg only, counter-clockwise to the stop and reversed
        h += M.DRIVE_SPEED * DT
        T.note_step(objs(h=h, w=M.DRIVE_SPEED), DT)
    for _ in range(int(T.STALL_S / DT) + 1):
        T.note_step(objs(h=45.0, w=0.0), DT)
    assert [s for s, _, _ in T._trial["rev"]] == ["ccw"]
    assert T.sweep_ends() is None and T.score(objs()) == 0.0


def test_latest_reading_each_way_counts():
    open_water()
    sweep((-45.0, 45.0), cycles=1)
    sweep((-60.0, 45.0), cycles=1)  # a collar crept: the latest stop is the one scored
    assert T.sweep_ends()[0] == pytest.approx(-60.0, abs=0.6)
    assert T.score(objs()) == 0.75


@pytest.mark.parametrize("err,credit", [(0.0, 1.0), (9.9, 1.0), (10.0, 1.0), (10.5, 0.5), (20.0, 0.5), (20.5, 0.25), (30.0, 0.25), (30.5, 0.0), (65.0, 0.0)])
def test_end_bands(err, credit):
    assert T.end_credit(err) == credit


@pytest.mark.parametrize("ends,want", [
    ((-45.0, 45.0), 1.0),
    ((-36.0, 54.0), 1.0),  # both 9 deg off: still within 10
    ((-45.0, 60.0), 0.75),  # one end 15 off
    ((-45.0, 110.0), 0.5),  # only the clockwise collar set (the other where it started)
    ((-60.0, 60.0), 0.5),  # both 15 off: an arc 30 too wide
    ((-30.0, 30.0), 0.5),  # both 15 off: an arc 30 too narrow
    ((-70.0, 70.0), 0.25),  # both 25 off
    ((-110.0, 110.0), 0.0),  # the start: 65 off each
    ((135.0, 225.0), 0.0),  # the right width aimed backwards
])
def test_score_ends(ends, want):
    assert T.score_ends(ends) == pytest.approx(want)


def test_the_right_width_in_the_wrong_place_scores_nothing():
    open_water()
    sweep((90.0, 180.0))
    assert T.score(objs()) == 0.0


def test_reaching_back_in_shuts_the_water_and_clears_the_record():
    open_water()
    sweep((-45.0, 45.0))
    assert T.score(objs()) == 1.0
    T.note_step(objs(left=NEAR), DT)
    assert not T._trial["water"] and T._trial["rev"] == []
    assert T.score(objs()) == 0.0
    for _ in range(int(T.OPEN_AFTER / DT) + 1):  # parked again: the valve reopens and the sweep is recorded afresh
        T.note_step(objs(), DT)
    assert T._trial["water"] and T._trial["opened"] == 2


def test_a_head_held_still_records_only_one_reversal_per_side():
    open_water()
    for _ in range(200):  # the head never turns (a hand outside the clear zone can't hold it, but a jam would do this)
        T.note_step(objs(h=0.0, w=0.0), DT)
    assert T.sweep_ends() is None or abs(T.sweep_ends()[0] - T.sweep_ends()[1]) < 1e-6
    assert T.score(objs()) == 0.0


def test_geometry_constants():
    # the start waters 220 deg, each end 65 deg past its flag; the goal tabs give the flags' arc
    assert M.limit_for_tab(M.START["lower"], "cw") == pytest.approx(M.START_LIMITS[0])
    assert M.limit_for_tab(M.START["upper"], "ccw") == pytest.approx(M.START_LIMITS[1])
    assert M.limit_for_tab(M.GOAL_TAB["lower"], "cw") == pytest.approx(M.TARGET[0])
    assert M.limit_for_tab(M.GOAL_TAB["upper"], "ccw") == pytest.approx(M.TARGET[1])
    assert M.DRIVE_TORQUE < M.COLLAR_SLIP and M.BEARING_FRICTION < M.DRIVE_TORQUE
    assert M.HEAD_LEN == pytest.approx(0.111) and M.HEAD_TOP == pytest.approx(0.114)


def test_demo_fits_the_run_limit_and_reaches_every_waypoint():
    wps = T.demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    assert t * 20 < 2400
    pe, re, bad = T.ik_errors(wps)
    assert pe < 0.003 and re < math.radians(1.0), bad[:3]
