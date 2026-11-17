"""`yam_steamer_basket`: the stock yam_arena embodiment plus the folding steamer (one articulation whose free root is its
base plate) and the sauce pan (one rigid body), both built by steamer_basket.mech. Created 11_17_26 (America/New_York).

They are built in place of the task file's placeholder box "steamer". The scorer gets, besides the stock objects:
`left_tcp` and `right_tcp` (each gripper's fingertip midpoint); `steamer` (the base plate: pos = the centre of its
underside, quat = its orientation); `leaves` (low = the 16 hinge angles in rad, high = the 4 tip-edge corners of each leaf in the
world, a 64 x 3 array); `handle` (low = (slide in m, 0, 0), high = the T-bar top's centre in the world); and `pan` (pos = the
middle of its outside bottom, quat = its orientation). They are read through the PhysX tensor views. The harness is
unchanged."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("STEAMER_DEBUG") == "1"
PLACEHOLDER = "steamer"


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


def _np(a):
    """A tensor-view getter's result as a NumPy copy (the view may hand back warp, torch or NumPy arrays)."""
    if hasattr(a, "numpy"):
        try:
            return np.array(a.numpy(), dtype=np.float32)
        except TypeError:
            pass
    return np.array(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float32)


def _put(setter, data, like, n_idx=1):
    """Call a tensor-view setter with a NumPy array, in the kind of array and on the device the getter returned."""
    idx = np.arange(n_idx, dtype=np.int32)
    if type(like).__module__.startswith("warp"):
        import warp as wp

        setter(wp.from_numpy(data.astype(np.float32), dtype=wp.float32, device=like.device),
               wp.from_numpy(idx, dtype=wp.int32, device=like.device))
    elif hasattr(like, "detach"):
        import torch

        setter(torch.tensor(data, dtype=torch.float32, device=like.device), torch.tensor(idx, dtype=torch.int32, device=like.device))
    else:
        setter(data.astype(np.float32), idx)


def rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SteamerBasketEmbodiment(YamArenaEmbodiment):
    _sim_view = None
    _art_view = None
    _pan_view = None
    _root = None
    _dof = None
    _link = None

    def reset(self, scene, *, seed=None):
        self._ensure_sim(None)
        self.task.module.reset_latch()
        self._start_pose()
        return super().reset(scene, seed=seed)

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

        if self._env is not None:
            return super()._ensure_sim(goal)
        if base._APP is None:  # Kit must be up before Arena is imported
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        from yam_arena import scene

        original = scene.task_object
        owner = self
        m = self.task.module

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            from . import mech

            stage = omni.usd.get_context().get_stage()
            prim = _place(stage, prim_path, translation, orientation)
            owner._root = prim_path
            mech.build_pan(stage, f"{prim_path}/pan_root", (0.0, 0.0))
            mech.build_steamer(stage, f"{prim_path}/steamer", tuple(m.STEAMER_XY - m.PAN_XY), m.STEAMER_YAW)
            return prim

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != PLACEHOLDER:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

    # -- PhysX tensor views ----------------------------------------------------------------------------------------------
    def _views(self):
        if self._art_view is None:
            import omni.physics.tensors as tensors

            from . import mech

            sim_view = None
            try:
                from isaaclab.sim import SimulationContext

                sim_view = SimulationContext.instance().physics_sim_view
            except Exception:  # noqa: BLE001
                sim_view = None
            if sim_view is None:
                sim_view = tensors.create_simulation_view("torch")
                sim_view.set_subspace_roots("/")
            self._sim_view = sim_view
            self._art_view = sim_view.create_articulation_view(self._root + "/steamer/plate")
            self._pan_view = sim_view.create_rigid_body_view(self._root + "/pan_root/pan")
            meta = self._art_view.shared_metatype
            names = list(meta.dof_names)
            self._dof = {"handle": names.index("j_handle")}
            self._dof.update({k: names.index(f"j_leaf{k:02d}") for k in range(mech.N_LEAVES)})
            links = list(meta.link_names)
            self._link = {n: links.index(n) for n in ["plate", "handle"] + [f"leaf{k:02d}" for k in range(mech.N_LEAVES)]}
            v = self._art_view
            fr = _np(v.get_dof_friction_properties())
            efforts = {"handle": mech.HANDLE_FRICTION_N, **{k: mech.LEAF_FRICTION_NM for k in range(mech.N_LEAVES)}}
            for j, e in efforts.items():
                fr[0, self._dof[j], :] = (e, e * mech.DYNAMIC_FRACTION, 0.0)
            _put(v.set_dof_friction_properties, fr, v.get_dof_friction_properties())
            if DEBUG:
                print("STEAMER dofs", names, "links", links, flush=True)
        return self._art_view

    def _start_pose(self):
        v = self._views()
        m = self.task.module
        pos = _np(v.get_dof_positions())
        for j, a in m.START_Q.items():
            pos[0, self._dof[j]] = a
        _put(v.set_dof_positions, pos, v.get_dof_positions())
        _put(v.set_dof_velocities, pos * 0.0, v.get_dof_velocities())

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        from . import mech

        base = self._base
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            q = arr(base.scene[robot].data.joint_pos.torch[0, arm_ids])
            p = np.asarray(ik.tcp_pose(q, np.array(BASES[arm]))[0], float)
            _, fing_ids = self._ids[robot]
            fw = arr(base.scene[robot].data.joint_pos.torch[0, fing_ids])  # finger joints (m), for the debug line
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=np.array((*fw[:2], 0.0)))
        v = self._views()
        tf = _np(v.get_link_transforms())[0].astype(float)
        vel = _np(v.get_link_velocities())[0].astype(float)
        qd = _np(v.get_dof_positions())[0].astype(float)
        pl = tf[self._link["plate"]]
        pv = vel[self._link["plate"]]
        out["steamer"] = ObjectState(pos=pl[:3], quat_xyzw=pl[3:7], lin_vel=pv[:3], ang_vel=pv[3:6], low=pl[:3], high=pl[:3])
        rp = rot(pl[3:7])
        angles = np.array([qd[self._dof[k]] for k in range(mech.N_LEAVES)])
        tips = np.array([pl[:3] + c @ rp.T for k in range(mech.N_LEAVES) for c in mech.leaf_tip_corners(k, angles[k])])
        out["leaves"] = ObjectState(pos=pl[:3], quat_xyzw=pl[3:7], lin_vel=zero, ang_vel=zero, low=angles, high=tips)
        slide = float(qd[self._dof["handle"]])
        top = pl[:3] + rp @ np.array((0.0, 0.0, mech.tbar_top(slide)))
        out["handle"] = ObjectState(pos=top, quat_xyzw=pl[3:7], lin_vel=zero, ang_vel=zero,
                                    low=np.array((slide, 0.0, 0.0)), high=top)
        pt = _np(self._pan_view.get_transforms())[0].astype(float)
        pvel = _np(self._pan_view.get_velocities())[0].astype(float)
        out["pan"] = ObjectState(pos=pt[:3], quat_xyzw=pt[3:7], lin_vel=pvel[:3], ang_vel=pvel[3:6], low=pt[:3], high=pt[:3])
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        m = self.task.module
        extra = ("left_tcp", "right_tcp", PLACEHOLDER, "leaves", "handle", "pan")

        def object_states(env, names):
            names = [n for n in names if n not in extra]
            return {**(original(env, names) if names else {}), **self._extra_states()}

        scene.object_states = object_states
        try:
            st0 = object_states(self._env, [])
            m.note_step(st0, 1.0 / self.info.control_hz)
            result = super()._task_step()
        finally:
            scene.object_states = original
        if DEBUG:
            print(m.debug_line(st0, self._t), flush=True)
        return result


def make_embodiment(**kwargs):
    return SteamerBasketEmbodiment(**kwargs)
