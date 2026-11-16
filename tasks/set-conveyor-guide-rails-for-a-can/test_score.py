"""The score, band by band, and each shortcut. Created 11_16_26 11:45 EST. Plain Python: `python -m pytest -q test_score.py`
needs NumPy only (no simulator, no robot model)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from guide_rails import mech as M
from guide_rails import scoring as S

TOP = 0.7475
Z3 = np.zeros(3)
RUN_THROUGH = (M.XC, 0.175, TOP + M.CHAIN_TOP + M.CAN_H / 2)  # where the demo leaves the can
AT_INFEED = (M.CAN_START[0], M.CAN_START[1], TOP + M.CAN_START[2])
FAR = {"left_tcp": (0.10, 0.36, TOP + 0.17), "right_tcp": (0.10, -0.36, TOP + 0.17)}


def quat_z(deg):
    a = math.radians(deg) / 2
    return np.array((0.0, 0.0, math.sin(a), math.cos(a)))


def quat_x(deg):
    a = math.radians(deg) / 2
    return np.array((math.sin(a), 0.0, 0.0, math.cos(a)))


def st(pos, quat=None, lin=Z3, ang=Z3):
    return SimpleNamespace(pos=np.array(pos, float), quat_xyzw=quat_z(0) if quat is None else quat, lin_vel=np.array(lin, float), ang_vel=np.array(ang, float))


def world(clear=(0.002, 0.002), knob_cw=(0.0, 0.0), plugs="rest", can=None, hands=None, rail_vel=Z3):
    """A scene: ``clear`` = the lane's clearance at (left, right) bracket (m); ``knob_cw`` = each knob's clockwise turn
    from the start (deg); ``plugs`` = "rest", "lane" or a pair of positions."""
    cl, cr = clear
    yl, yr = M.STATIONS["left"], M.STATIONS["right"]
    xl, xr = M.X_DATUM - M.CAN_D - cl, M.X_DATUM - M.CAN_D - cr  # the face at each bracket
    skew = math.atan2(xl - xr, yl - yr)
    face_mid = np.array(((xl + xr) / 2, 0.0, TOP + M.RAIL_Z))
    R = S.rot(quat_z(math.degrees(-skew)))
    # the face points are pos + R @ (T/2, y, 0): solve for pos so that both bracket faces land where asked
    pos = face_mid - R @ np.array((M.RAIL_T / 2, 0.0, 0.0))
    o = {"rail": st(pos, quat_z(math.degrees(-skew)), lin=rail_vel)}
    for s, cw in zip(S.SIDES, knob_cw):
        o[f"knob_{s}"] = st((M.XB, M.STATIONS[s], TOP + M.KNOB_Z), quat_z(-cw))
    for s in S.SIDES:
        if plugs == "rest":
            p = (*M.REST[s], TOP + M.REST_T + M.PLUG_H / 2)
        elif plugs == "lane":
            x, y, z = M.PLUG_START[s]
            p = (x, y, TOP + z)
        else:
            p = plugs[S.SIDES.index(s)]
        o[f"plug_{s}"] = p if isinstance(p, SimpleNamespace) else st(p)
    o["can"] = can or st(RUN_THROUGH)
    for h, p in {**FAR, **(hands or {})}.items():
        o[h] = st(p)
    return o


def run(o, seconds=1.5, first=None):
    """Play a scene through note_step: ``first`` (a scene) for one step, to set the knobs' starting angle, then ``o``."""
    S.reset()
    S.note_step(first or world(clear=(0.012, 0.012), plugs="lane", can=st(AT_INFEED)), 0.05, TOP)
    for _ in range(int(seconds / 0.05)):
        S.note_step(o, 0.05, TOP)
    return S.score(o, TOP)


def test_clamp_law():
    assert M.knob_torque(M.A_START) == pytest.approx(0.52)  # locked as found: 0.5 N m over the thread's drag
    assert M.hold_force(M.A_START) == pytest.approx(M.F_SLIDE + 187.0 * 0.5)  # 94.1 N
    assert M.hold_force(-30.0) == M.F_SLIDE and M.knob_torque(-30.0) == M.T_FREE  # backed off: only drag
    assert M.hold_force(M.tight_for(10.0)) == pytest.approx(10.0)
    assert 5.0 < M.tight_for(10.0) < 7.0  # about 6 degrees past the seat


def test_geometry():
    o = world(clear=(0.002, 0.001))
    c = S.clearances(o)
    assert c["left"] == pytest.approx(0.002, abs=1e-6) and c["right"] == pytest.approx(0.001, abs=1e-6)
    assert M.PLUG_D - M.CAN_D == pytest.approx(0.002)
    assert M.X_DATUM - M.X_FACE_START == pytest.approx(0.078)
    assert M.PLUG_MASS == pytest.approx(0.307, abs=0.002)
    # the as-found rail: 12 mm of clearance at both brackets
    start = SimpleNamespace(pos=np.array((M.RAIL_START[0], 0.0, TOP + M.RAIL_Z)), quat_xyzw=quat_z(0), lin_vel=Z3, ang_vel=Z3)
    c0 = S.clearances({"rail": start})
    assert c0["left"] == pytest.approx(0.012, abs=1e-6) and c0["right"] == pytest.approx(0.012, abs=1e-6)


def test_nothing_done_scores_zero():
    assert run(world(clear=(0.012, 0.012), plugs="lane", can=st(AT_INFEED))) == 0.0


def test_full_score():
    assert run(world()) == 1.0
    assert run(world(clear=(0.0005, 0.003))) == 1.0  # the band's two ends, the rail 2.5 mm out of parallel
    assert run(world(knob_cw=(10.0, -40.0))) == 1.0  # tighter than found, and 40 degrees less: 20 past the seat still holds


def test_band_edges():
    assert run(world(clear=(0.0004, 0.002))) == 0.25  # 0.4 mm: too tight at one bracket
    assert run(world(clear=(0.0031, 0.002))) == 0.25  # 3.1 mm: too slack at one bracket
    assert run(world(clear=(0.0, 0.0))) == 0.0  # touching the can at both: pinched
    assert run(world(clear=(-0.0005, 0.002))) == 0.25  # squeezing the can at one bracket
    assert run(world(clear=(0.0035, 0.0045))) == 0.0


def test_one_bracket_only():
    assert run(world(clear=(0.002, 0.012))) == 0.25
    assert run(world(clear=(0.012, 0.002))) == 0.25


def test_unclamped_is_capped():
    assert run(world(knob_cw=(-95.0, -95.0))) == 0.25  # both knobs left slack: set, but nothing holds it
    assert run(world(knob_cw=(-56.0, -56.0))) == 0.25  # 4 degrees past the seat: 6.8 N, under the 10 N test
    assert run(world(knob_cw=(-53.0, -53.0))) == 1.0  # 7 degrees past: 11.5 N


def test_one_clamp():
    assert run(world(knob_cw=(0.0, -95.0))) == 0.5
    assert run(world(knob_cw=(-95.0, 5.0))) == 0.5


def test_knob_turned_and_turned_back():
    """The tightness follows the knob both ways: slackened 95 degrees and locked 110 is 15 degrees tighter than found."""
    S.reset()
    o = world()
    S.note_step(o, 0.05, TOP)
    for cw in list(np.arange(0.0, -95.1, -5.0)) + list(np.arange(-95.0, 15.1, 5.0)):
        S.note_step(world(knob_cw=(cw, cw)), 0.05, TOP)
    assert S.tightness("left") == pytest.approx(M.A_START + 15.0, abs=0.01)
    sent = dict(S.take_commands())  # the last limits handed to the joints
    assert sent["knob_left"] == pytest.approx(M.knob_torque(75.0), abs=0.005) and sent["slide_right"] == pytest.approx(M.hold_force(75.0), abs=0.25)


def test_knob_more_than_half_a_turn():
    S.reset()
    S.note_step(world(), 0.05, TOP)
    for cw in np.arange(0.0, -400.1, -20.0):
        S.note_step(world(knob_cw=(cw, 0.0)), 0.05, TOP)
    assert S.tightness("left") == pytest.approx(M.A_START - 400.0, abs=0.01) and S.tightness("right") == pytest.approx(M.A_START)


def test_plugs_left_in_the_lane():
    assert run(world(plugs="lane")) == 0.75


def test_plug_off_its_rest_or_tipped():
    rest = (*M.REST["right"], TOP + M.REST_T + M.PLUG_H / 2)
    off = (M.REST["left"][0] + 0.020, M.REST["left"][1], TOP + M.REST_T + M.PLUG_H / 2)
    assert run(world(plugs=(off, rest))) == 0.75  # 20 mm off the rest's centre
    near = (M.REST["left"][0] + 0.014, M.REST["left"][1], TOP + M.REST_T + M.PLUG_H / 2)
    assert run(world(plugs=(near, rest))) == 1.0
    lying = st((M.REST["left"][0], M.REST["left"][1], TOP + M.REST_T + M.PLUG_D / 2), quat_x(90))
    assert run(world(plugs=(lying, rest))) == 0.75
    held_up = (M.REST["left"][0], M.REST["left"][1], TOP + M.REST_T + M.PLUG_H / 2 + 0.01)
    assert run(world(plugs=(held_up, rest))) == 0.75
    both_on_one = (rest, (rest[0] + 0.01, rest[1], rest[2]))
    assert run(world(plugs=both_on_one)) == 0.75
    swapped = ((*M.REST["right"], rest[2]), (*M.REST["left"], rest[2]))
    assert run(world(plugs=swapped)) == 1.0  # either plug on either rest


def test_can_not_run_through():
    assert run(world(can=st(AT_INFEED))) == 0.75  # left at the infeed end
    half = st((M.XC, 0.05, TOP + M.CHAIN_TOP + M.CAN_H / 2))
    assert run(world(can=half)) == 0.75  # stopped mid-lane, short of the left bracket
    short = st((M.XC, M.STATIONS["left"] + 0.029, TOP + M.CHAIN_TOP + M.CAN_H / 2))
    assert run(world(can=short)) == 0.75
    edge = st((M.XC, M.STATIONS["left"] + 0.030, TOP + M.CHAIN_TOP + M.CAN_H / 2))
    assert run(world(can=edge)) == 1.0
    lying = st((M.XC, 0.175, TOP + M.CHAIN_TOP + M.CAN_D / 2), quat_x(90))
    assert run(world(can=lying)) == 0.75  # knocked over
    lifted = st((M.XC, 0.175, TOP + M.CHAIN_TOP + M.CAN_H / 2 + 0.02))
    assert run(world(can=lifted)) == 0.75  # carried, not run on the chain
    away = st((0.20, 0.30, TOP + M.CAN_H / 2))
    assert run(world(can=away)) == 0.75  # carried round onto the table


def test_can_run_before_the_clamps_held():
    """Pushing the can through the open lane first, then setting the rail, is not a run through the set lane."""
    S.reset()
    S.note_step(world(clear=(0.012, 0.012), knob_cw=(0.0, 0.0), plugs="rest", can=st(AT_INFEED)), 0.05, TOP)
    S.note_step(world(clear=(0.012, 0.012), knob_cw=(-95.0, -95.0), plugs="rest", can=st(AT_INFEED)), 0.05, TOP)  # slackened
    S.note_step(world(clear=(0.012, 0.012), knob_cw=(-95.0, -95.0), plugs="rest"), 0.05, TOP)  # can run through, open lane
    for _ in range(40):
        S.note_step(world(knob_cw=(15.0, 15.0)), 0.05, TOP)  # then set and locked
    assert S.score(world(knob_cw=(15.0, 15.0)), TOP) == 0.75


def test_part_off_the_table():
    fell = (0.9, 0.0, TOP - 0.3)
    rest = (*M.REST["right"], TOP + M.REST_T + M.PLUG_H / 2)
    assert run(world(plugs=(fell, rest))) == 0.0
    S.reset()  # latched: back on the table afterwards does not restore the score
    S.note_step(world(plugs=(fell, rest)), 0.05, TOP)
    for _ in range(40):
        S.note_step(world(), 0.05, TOP)
    assert S.score(world(), TOP) == 0.0


def test_hand_still_on_the_rail():
    hold = {"right_tcp": (M.X_FACE_SET - M.RAIL_T / 2, -0.065, TOP + M.RAIL_Z)}
    assert run(world(hands=hold)) == 0.25  # the rail held in place by a gripper is not a set rail
    assert run(world(knob_cw=(-95.0, -95.0), hands=hold)) == 0.25


def test_rail_still_moving():
    assert run(world(rail_vel=(0.02, 0.0, 0.0))) == 0.25


def test_hand_on_a_knob_or_a_plug():
    on_knob = {"left_tcp": (M.XB, M.STATIONS["left"], TOP + M.KNOB_Z)}
    assert run(world(hands=on_knob)) == 0.75
    on_plug = {"left_tcp": (*M.REST["left"], TOP + M.REST_T + M.PLUG_H / 2 + 0.01)}
    assert run(world(hands=on_plug)) == 0.75


def test_not_yet_at_rest_for_a_second():
    assert run(world(), seconds=0.5) == 0.75
    assert run(world(), seconds=1.05) == 1.0


def test_scores_are_the_five_levels():
    seen = set()
    for cl in (-0.001, 0.0004, 0.0005, 0.002, 0.003, 0.0031, 0.012):
        for cr in (0.0004, 0.002, 0.012):
            for kn in ((0.0, 0.0), (-95.0, 0.0), (-95.0, -95.0)):
                for pl in ("rest", "lane"):
                    seen.add(run(world(clear=(cl, cr), knob_cw=kn, plugs=pl), seconds=1.2))
    assert seen == {0.0, 0.25, 0.5, 0.75, 1.0}
