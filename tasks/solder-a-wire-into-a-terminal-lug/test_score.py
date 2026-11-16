"""Scoring checks with no simulator: python -m pytest -q test_score.py   (created 11_15_26, local time)"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import task as t
from lug_solder import mech as M

DT = 0.05
PARK = {"left_tcp": (0.12, 0.34, t.PARK[2]), "right_tcp": (0.12, -0.34, t.PARK[2])}


def state(pos, yaw=0.0, v=0.0):
    p = np.array(pos, float)
    return NS(pos=p, quat_xyzw=np.array((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))), lin_vel=np.array((v, 0.0, 0.0)), ang_vel=np.zeros(3), low=p, high=p)


def world(iron="rest", solder="rest", wire=(0.0, 0.0, 0.0), iron_hand="left_tcp", solder_hand="right_tcp"):
    """iron: 'rest', 'on' (tip across the barrel) or 'table'; solder: 'rest', 'joint', 'tip', 'tongue' or 'table'."""
    w = {a: state(p) for a, p in PARK.items()}
    w["wire"] = state(t.WIRE0 + wire)
    if iron == "rest":
        w["iron"] = state(t.IRON0, t.IRON_HEAD)
    elif iron == "table":
        w["iron"] = state((0.2, 0.15, t.TOP + 0.03), t.IRON_HEAD)
    else:
        g = t.iron_tcp(t.iron_target(), t.IRON_HEAD)[0] - (0, 0, t.GRIP_DZ)
        w["iron"] = state(g, t.IRON_HEAD)
        if iron_hand:
            w[iron_hand] = state(g + (0, 0, t.GRIP_DZ))
    if solder == "rest":
        w["solder"] = state(t.SOLDER0, t.STICK_HEAD)
    elif solder == "table":
        w["solder"] = state((0.2, -0.15, t.TOP + 0.03), t.STICK_HEAD)
    else:
        tip = {"joint": t.stick_target(), "tip": t.iron_target() + (0, -0.005, 0.004), "tongue": t.AXIS + (0.022, 0.004, 0.002)}[solder]
        s = t.solder_tcp(tip, t.STICK_HEAD)[0] - (0, 0, t.SLEEVE_DZ)
        w["solder"] = state(s, t.STICK_HEAD)
        if solder_hand:
            w[solder_hand] = state(s + (0, 0, t.SLEEVE_DZ))
    return w


def run(w, seconds, **touch):
    for _ in range(round(seconds / DT)):
        t.note_step(w, DT, {k: DT * v for k, v in touch.items()})
    return w


def job(heat=10.5, feed=5.25, flow=1.0, setting=25.0, feed_touch=None, tidy=True, during_set=None, **kw):
    """The whole job through note_step; returns the final score."""
    run(world(iron="on", **kw), heat, iron_lug=1)
    run(world(iron="on", solder="joint", **kw), feed, **(feed_touch or dict(iron_lug=1, solder_wire=1)))
    run(world(iron="on", **kw), flow, iron_lug=1)
    if during_set:
        during_set()
    else:
        run(world(), setting)
    end = world() if tidy else world(iron="table")
    run(end, 2.0)
    return t.score(end)


@pytest.fixture(autouse=True)
def fresh():
    t.reset_latch()


def test_heat_numbers():
    """Solder flows after about 9 s of contact, and the overheating limit is reached after about 20 s."""
    w = world(iron="on")
    n = 0
    while t.temp()[0] < t.T_FLOW:
        run(w, DT, iron_lug=1)
        n += 1
    assert 7.5 < n * DT < 10.0
    while t.temp()[0] < t.T_BURN:
        run(w, DT, iron_lug=1)
        n += 1
    assert 18.0 < n * DT < 22.0
    run(w, 1.0, iron_lug=1)
    assert t._trial["burnt"]


def test_full_score():
    assert job() == 1.0
    assert 30 <= t._trial["fed"] <= 55 and t.joined() and t.defects() == []


def test_nothing_done():
    w = run(world(), 30.0)
    assert t.score(w) == 0.0 and t.temp()[0] == t.T_AMB


def test_hover_no_contact():
    """The tip held over the lug without touching: no heat, the solder never melts."""
    run(world(iron="on"), 15.0)
    run(world(iron="on", solder="joint"), 6.0, solder_wire=1)
    w = run(world(), 25.0)
    assert t.temp()[0] == t.T_AMB and t._trial["fed"] == 0 and t.score(w) == 0.0


def test_iron_not_held_does_not_heat():
    run(world(iron="on", iron_hand=None), 20.0, iron_lug=1)
    assert t.temp()[0] == t.T_AMB


def test_no_solder():
    run(world(iron="on"), 20.0, iron_lug=1)
    w = run(world(), 30.0)
    assert t.score(w) == 0.0 and t._trial["phase"] == "none"


def test_solder_on_the_iron_only():
    assert job(feed_touch=dict(iron_lug=1, iron_solder=1)) == 0.0
    assert t._trial["fed"] == 0 and t._trial["blob"] > 30


def test_solder_on_the_tongue_is_not_in_the_joint():
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="tongue"), 5.0, iron_lug=1, solder_lug=1)
    w = run(world(), 30.0)
    assert t._trial["fed"] == 0 and t._trial["stray"] > 30 and t.score(w) == 0.0


def test_solder_on_a_cold_joint_does_not_melt():
    run(world(iron="on"), 5.0, iron_lug=1)
    run(world(iron="on", solder="joint"), 3.0, solder_wire=1)
    assert t._trial["fed"] == 0


def test_one_hand_cannot_do_both():
    run(world(iron="on"), 10.5, iron_lug=1)
    w = world(iron="on", solder="joint", solder_hand=None)
    run(w, 5.0, iron_lug=1, solder_wire=1)
    assert t._trial["fed"] == 0


def test_starved_and_excess():
    assert job(feed=2.0) == 0.5 and t.defects() == ["starved"]
    t.reset_latch()
    assert job(feed=8.0) == 0.5 and t.defects() == ["excess"]


def test_too_little_to_hold():
    assert job(feed=1.0) == 0.0 and not t.joined()


def test_overheated():
    assert job(heat=30.0) == 0.5 and t.defects() == ["overheated"]
    t.reset_latch()
    assert job(heat=30.0, feed=2.0) == 0.25


def test_tip_on_the_insulation():
    w = world(iron="on")
    g = w["iron"].pos + (-(t.IRON_AT + M.BARE + 0.010), 0, 0)
    w["iron"], w["left_tcp"] = state(g, t.IRON_HEAD), state(g + (0, 0, t.GRIP_DZ))
    run(w, 1.0, iron_wire=1)
    assert t._trial["ins"] and t.temp()[0] == t.T_AMB
    assert job() == 0.5 and t.defects() == ["overheated"]


def test_iron_off_before_the_solder():
    """The tip leaves and solder is still fed a second later: a cold joint."""
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="joint"), 4.0, iron_lug=1, solder_wire=1)
    run(world(solder="joint"), 1.5, solder_wire=1)
    w = run(world(), 30.0)
    assert t._trial["cold"] and not t.joined() and t.score(w) == 0.0


def test_solder_left_in_while_it_freezes():
    def during():
        run(world(solder="joint"), 30.0, solder_wire=0.0)  # touching nothing: fine
    assert job(during_set=during, tidy=True) in (0.75, 1.0)
    t.reset_latch()
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="joint"), 5.25, iron_lug=1, solder_wire=1)
    run(world(iron="on"), 1.5, iron_lug=1)
    w = world(solder="joint")
    for _ in range(600):  # the stick rests on the joint, too cool to melt, as it freezes
        t.note_step(w, DT, {"solder_wire": DT if t.temp()[0] < t.T_FLOW else 0.0})
    assert t._trial["cold"] and t.score(run(world(), 2.0)) == 0.0


def test_wire_moved_while_freezing():
    def during():
        run(world(), 12.0)
        assert t._trial["phase"] == "freezing"
        run(world(wire=(0.0, 0.001, 0.0)), 20.0)
    assert job(during_set=during) == 0.0 and "moved" in t._trial["why"]


def test_wire_moved_while_liquid_is_fine():
    def during():
        run(world(wire=(0.0, 0.0008, 0.0)), 30.0)
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="joint"), 5.25, iron_lug=1, solder_wire=1)
    run(world(iron="on", wire=(0.0, 0.0008, 0.0)), 1.5, iron_lug=1)
    during()
    assert t.joined()


def test_wire_yanked_out():
    def during():
        run(world(wire=(-0.03, 0.0, 0.0)), 30.0)
    assert job(during_set=during) == 0.0


def test_wire_out_before_freezing_starts():
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="joint"), 5.25, iron_lug=1, solder_wire=1)
    w = run(world(wire=(-0.03, 0.0, 0.0)), 40.0)
    assert not t.joined() and "out of the barrel" in t._trial["why"] and t.score(w) == 0.0


def test_reflow_repairs_a_cold_joint():
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="joint"), 4.0, iron_lug=1, solder_wire=1)
    run(world(solder="joint"), 1.5, solder_wire=1)
    run(world(), 30.0)
    assert not t.joined()
    run(world(iron="on"), 12.0, iron_lug=1)
    assert t._trial["phase"] == "liquid" and not t._trial["cold"]
    w = run(world(), 30.0)
    assert t.joined() and t.score(w) == 1.0


def test_remelting_a_set_joint_unjoins_it():
    job()
    run(world(iron="on"), 12.0, iron_lug=1)
    assert not t.joined()
    assert t.score(world(iron="on")) == 0.0


def test_not_set_by_the_end():
    run(world(iron="on"), 10.5, iron_lug=1)
    run(world(iron="on", solder="joint"), 5.25, iron_lug=1, solder_wire=1)
    w = run(world(), 5.0)
    assert t._trial["phase"] in ("liquid", "freezing") and t.score(w) == 0.0


def test_untidy_bands():
    assert job(tidy=False) == 0.75
    t.reset_latch()
    job()
    w = world()
    w["left_tcp"] = state(t.IRON0 + (0, 0, 0.03))  # a hand still over the iron
    run(w, 2.0)
    assert t.score(w) == 0.75
    w = world()
    w["iron"] = state(t.IRON0, t.IRON_HEAD, v=0.05)  # still moving
    run(w, 2.0)
    assert t.score(w) == 0.75
    w = world()
    w["right_tcp"] = state(t.AXIS + (0, 0, 0.03))  # a hand over the joint
    run(w, 2.0)
    assert t.score(w) == 0.75


def test_tool_off_the_table():
    job()
    w = world()
    w["iron"] = state((0.2, 0.6, 0.2), t.IRON_HEAD)
    run(w, 1.0)
    assert t.score(w) == 0.0


def test_shown():
    job()
    assert t.shown() == {"fx_fill"}
    t.reset_latch()
    job(heat=30.0, feed=2.0)
    assert t.shown() == {"fx_lean", "fx_char"}


def test_demo_plan_fits():
    wps = t.demo()
    dur = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad, _ = t.ik_errors(wps)
    assert dur * 20 < 2000 and not bad
