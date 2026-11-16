"""The conveyor section, its adjustable guide rail, the two clamp knobs, the setting plugs and the can.
Created 11_16_26 10:10 EST. The geometry and the clamp law at the top are NumPy only, so the task file and the tests
use them without Isaac Sim; the USD builders import pxr when called.

Frames (m): x away from the arms, y to the left, z up from the station table's top. The lane runs along y.

- Conveyor (one fixed body, 560 mm long): two side frames and a bed carrying an 82.6 mm (3.25 in) tabletop chain whose top is
  CHAIN_TOP above the table; the fixed datum rail on the far side on two fixed brackets; on the near side two bracket
  posts, each carrying a cross block with a bore for the adjustable rail's rod; two rests for the setting plugs.
- Rail (one loose body): the adjustable guide rail (a 1 x 0.69 in flat rail, 360 mm) with a 1/2 in rod screwed square
  into its back at each station. Each rod slides along x in its cross block; the two stations slide independently, so
  the rail can stand skewed. Body origin: the rail bar's centre; its inner (product) face is at local x +RAIL_T/2.
- Knob (two loose bodies): a 40 mm star knob on a vertical M8 clamp screw on top of each cross block. Turned clockwise
  (seen from above) it closes the block's split bore on the rod.
- Plug (two loose bodies): a setting plug, a 68.0 mm acetal cylinder: the 66.0 mm can plus the 2.0 mm running clearance.
- Can (loose): a full 355 ml (12 fl oz) drinks can, 66.0 mm body, 122.2 mm tall.

The clamp, as a declared law over real joints (the thread and the split block's elastic closing are not meshed):
- tightness a (deg) = the knob's clockwise turn past the point where the block first touches the rod (the seat).
- torque to turn the knob either way: T(a) = T_FREE + K_TURN * max(0, a) (N m). It is the friction limit of the knob's
  hinge, so the gripper has to supply it, and a knob let go stays where it is.
- the rod's holding force along its axis: F(a) = F_SLIDE + HOLD_PER_NM * K_TURN * max(0, a) (N): the friction limit
  of that station's slide. Below the seat the rod slides against F_SLIDE alone.
"""

import math

import numpy as np

MM = 0.001

# -- product and setting plugs -----------------------------------------------------------------------------------------
CAN_D, CAN_H, CAN_MASS = 0.0660, 0.1222, 0.38  # 2.60 in body, 4.81 in tall; full: 355 ml of drink plus a 13 g can (est)
CLEAR_NOM = 0.002  # the running clearance the plugs set
PLUG_D, PLUG_H = CAN_D + CLEAR_NOM, 0.060
PLUG_MASS = 1410.0 * math.pi * (PLUG_D / 2) ** 2 * PLUG_H  # solid acetal: 0.307 kg
SIDES = 32  # the round colliders are 32-sided prisms, flats across the lane: 68.00 across flats, 68.33 across corners

# -- conveyor ----------------------------------------------------------------------------------------------------------
XC = 0.335  # the chain's centre line
CHAIN_W, CHAIN_TOP, CHAIN_PITCH, CHAIN_PLATE = 0.0826, 0.050, 0.0381, 0.004  # 820-series K325 tabletop chain
SEC_LEN = 0.56  # the section, along y
FRAME_T, FRAME_TOP = 0.015, 0.046  # side frames; the chain's plates stand 4 mm proud of them
X_DATUM = XC + PLUG_D / 2  # the datum rail's product face: the lane, set, is centred on the chain
RAIL_T, RAIL_H, RAIL_LEN = 0.0175, 0.0254, 0.34  # 0.69 x 1 in flat guide rail
RAIL_Z = CHAIN_TOP + 0.040  # the rails' centre line above the table: 40 mm above the chain
RAIL_MASS = 0.63  # 0.26 kg of rail (0.71 kg/m), two 0.14 kg rods, two rail clamps
RAIL_INERTIA = (0.0075, 0.0012, 0.0085)
STATIONS = {"left": 0.11, "right": -0.11}  # y of each bracket
ROD_D, ROD_LEN = 0.0127, 0.140  # 1/2 in stainless rod
XB = 0.215  # the cross blocks' centre
BLOCK = (0.045, 0.051, 0.060)  # cross block, x, y, z; centred on the rod
BORE = 0.016  # the collider's square channel for the rod (the real bore is 12.7 mm, split)
KNOB_D, KNOB_H, KNOB_STEM = 0.040, 0.022, 0.006  # star knob over the block's top
KNOB_Z = RAIL_Z + BLOCK[2] / 2 + KNOB_STEM + KNOB_H / 2  # its centre above the table
KNOB_MASS = 0.035
LANE_START = 0.078  # the lane as found: set for a 76 mm can
X_FACE_START = X_DATUM - LANE_START  # the adjustable rail's product face, as found
X_FACE_SET = X_DATUM - PLUG_D
RAIL_START = (X_FACE_START - RAIL_T / 2, 0.0, RAIL_Z)
TRAVEL = (-0.022, 0.016)  # the rods' stops, from the start: out to a 100 mm lane, in to a 62 mm one
SKEW_MAX_DEG = 6.0
PLUG_START = {s: (X_DATUM - PLUG_D / 2, y, CHAIN_TOP + PLUG_H / 2) for s, y in STATIONS.items()}  # against the datum rail
REST_R, REST_T = 0.050, 0.003  # the plugs' rests: 100 mm discs on the table
REST = {"left": (0.165, 0.285), "right": (0.250, 0.385)}  # both beyond the lane's left (outfeed) end
CAN_START = (XC, -0.240, CHAIN_TOP + CAN_H / 2)  # the sample can: on the chain at the right-hand end, just outside the rails

# -- the clamp ---------------------------------------------------------------------------------------------------------
A_START = 60.0  # deg past the seat, as found: locked at 0.5 N m
T_FREE = 0.02  # N m: thread drag below the seat
K_TURN = 0.5 / 60.0  # N m per degree past the seat (est: an eighth to a quarter turn from snug to hand tight)
HOLD_PER_NM = 187.0  # N of axial hold per N m: M8 screw, nut factor 0.2 (625 N of clamp load per N m), two lines of contact at 0.15
F_SLIDE = 0.6  # N per station: the loose rod's sliding drag (half the 0.63 kg rail assembly's weight at 0.2)
A_MIN, A_MAX = -400.0, 150.0  # the knob's stops: about a turn out; 1.25 N m in
LOCK_N = 10.0  # a rod whose clamp holds this much or more is locked solid in the simulation (a real one slips at its holding force)


def knob_torque(a_deg):
    """N m to turn the knob (either way) at tightness ``a_deg``."""
    return T_FREE + K_TURN * max(0.0, float(a_deg))


def hold_force(a_deg):
    """N along the rod before it slips in its block at tightness ``a_deg``."""
    return F_SLIDE + HOLD_PER_NM * K_TURN * max(0.0, float(a_deg))


def tight_for(force):
    """The tightness (deg past the seat) at which the rod holds ``force`` N."""
    return max(0.0, (force - F_SLIDE) / (HOLD_PER_NM * K_TURN))


COLORS = {  # sRGB base colour, metallic, roughness
    "frame": ((0.62, 0.64, 0.66), 0.9, 0.4), "bed": ((0.30, 0.31, 0.33), 0.6, 0.6), "chain": ((0.78, 0.66, 0.46), 0.0, 0.5),
    "gap": ((0.16, 0.14, 0.11), 0.0, 0.9), "rail": ((0.72, 0.74, 0.76), 1.0, 0.3), "wear": ((0.94, 0.94, 0.90), 0.0, 0.6),
    "rod": ((0.80, 0.81, 0.83), 1.0, 0.25), "block": ((0.07, 0.07, 0.08), 0.0, 0.55), "knob": ((0.05, 0.05, 0.05), 0.0, 0.45),
    "mark": ((0.92, 0.80, 0.10), 0.0, 0.6), "plug": ((0.95, 0.45, 0.08), 0.0, 0.5), "rest": ((0.15, 0.42, 0.25), 0.0, 0.7),
    "can": ((0.80, 0.82, 0.84), 1.0, 0.25), "band": ((0.10, 0.30, 0.65), 0.3, 0.4), "lid": ((0.70, 0.71, 0.73), 1.0, 0.35),
    "post": ((0.55, 0.56, 0.58), 1.0, 0.35),
}
MU = {"steel": 0.4, "slide": 0.2, "knob": 1.0, "plastic": 0.35}  # acetal and UHMW on steel or aluminium: 0.2


# -- USD -------------------------------------------------------------------------------------------------------------
def _physx(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


class _Builder:
    def __init__(self, stage, root):
        self.stage, self.root = stage, root
        self.looks, self.phys = {}, {}

    def look(self, name):
        from pxr import Gf, Sdf, UsdShade

        if name not in self.looks:
            color, metal, rough = COLORS[name]
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Looks/{name}")
            sh = UsdShade.Shader.Define(self.stage, f"{self.root}/Looks/{name}/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_lin(color)))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.looks[name] = mat
        return self.looks[name]

    def surface(self, name):
        from pxr import UsdPhysics, UsdShade

        if name not in self.phys:
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
            pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm.CreateStaticFrictionAttr(MU[name])
            pm.CreateDynamicFrictionAttr(MU[name])
            pm.CreateRestitutionAttr(0.0)
            self.phys[name] = mat
        return self.phys[name]

    def _finish(self, g, look, collide, offset=0.001):
        from pxr import UsdGeom, UsdPhysics, UsdShade

        prim = g.GetPrim()
        if look is None:
            UsdGeom.Imageable(prim).MakeInvisible()
        else:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(look))
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            _physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": float(offset), "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide), UsdShade.Tokens.weakerThanDescendants, "physics")
        return g

    def box(self, path, lo, hi, look, collide=None):
        from pxr import Gf, UsdGeom

        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, (lo + hi) / 2)))
        c.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
        return self._finish(c, look, collide)

    def cyl(self, path, center, r, h, look, axis="Z"):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        return self._finish(c, look, None)

    def prism(self, path, center, across_flats, h, collide, n=SIDES, look=None):
        """An upright n-sided prism, a flat facing +x and -x: the collider of a round part (one convex hull)."""
        from pxr import Gf, UsdGeom, UsdPhysics, Vt

        rc = across_flats / 2 / math.cos(math.pi / n)
        m = UsdGeom.Mesh.Define(self.stage, path)
        pts = [Gf.Vec3f(rc * math.cos(2 * math.pi * (k + 0.5) / n), rc * math.sin(2 * math.pi * (k + 0.5) / n), z) for z in (-h / 2, h / 2) for k in range(n)]
        counts, idx = [], []
        for k in range(n):
            j = (k + 1) % n
            counts.append(4)
            idx += [k, j, n + j, n + k]
        counts += [n, n]
        idx += list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
        m.CreatePointsAttr(Vt.Vec3fArray(pts))
        m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        m.CreateSubdivisionSchemeAttr("none")
        m.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        self._finish(m, look, collide)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        return m

    def rigid(self, prim, kinematic=False, mass=None, com=None, inertia=None, iters=32):
        from pxr import Gf, UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        attrs = {"physxRigidBody:solverPositionIterationCount": iters, "physxRigidBody:solverVelocityIterationCount": 2,
                 "physxRigidBody:maxDepenetrationVelocity": 0.2}
        if not kinematic:
            attrs["physxRigidBody:enableCCD"] = True
            attrs["physxRigidBody:sleepThreshold"] = 0.0
        _physx(prim, "PhysxRigidBodyAPI", **attrs)
        if mass is not None:
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(float(mass))
            m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com or (0.0, 0.0, 0.0))))
            if inertia is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))


def build_conveyor(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, kinematic=True)
    x0, x1 = XC - CHAIN_W / 2, XC + CHAIN_W / 2
    y0, y1 = -SEC_LEN / 2, SEC_LEN / 2
    b.box(f"{path}/frame_near", (x0 - FRAME_T, y0, 0.0), (x0, y1, FRAME_TOP), "frame", "steel")
    b.box(f"{path}/frame_far", (x1, y0, 0.0), (x1 + FRAME_T, y1, FRAME_TOP), "frame", "steel")
    b.box(f"{path}/bed", (x0, y0, 0.0), (x1, y1, CHAIN_TOP - CHAIN_PLATE), "bed", "steel")
    b.box(f"{path}/chain", (x0, y0, CHAIN_TOP - CHAIN_PLATE), (x1, y1, CHAIN_TOP), "gap", "slide")
    n = int(SEC_LEN / CHAIN_PITCH)
    for k in range(n):  # the chain's top plates, drawn only: 38.1 mm pitch, 1.5 mm gaps
        ya = y0 + (SEC_LEN - n * CHAIN_PITCH) / 2 + k * CHAIN_PITCH
        b.box(f"{path}/plate{k}", (x0, ya + 0.00075, CHAIN_TOP - 0.001), (x1, ya + CHAIN_PITCH - 0.00075, CHAIN_TOP + 0.0002), "chain")
    for k, yy in enumerate((y0, y1)):  # end caps over the chain's return
        ya, yb = sorted((yy, yy - math.copysign(0.004, yy)))
        b.box(f"{path}/end{k}", (x0 - FRAME_T, ya, 0.0), (x1 + FRAME_T, yb, FRAME_TOP), "frame")
    # the datum rail: fixed, on two fixed brackets on the far frame
    r0, r1 = RAIL_Z - RAIL_H / 2, RAIL_Z + RAIL_H / 2
    b.box(f"{path}/datum_rail", (X_DATUM + 0.003, -RAIL_LEN / 2, r0), (X_DATUM + RAIL_T, RAIL_LEN / 2, r1), "rail", "slide")
    b.box(f"{path}/datum_wear", (X_DATUM, -RAIL_LEN / 2, r0), (X_DATUM + 0.003, RAIL_LEN / 2, r1), "wear", "slide")
    xf = x1 + FRAME_T
    for s, y in STATIONS.items():
        b.box(f"{path}/datum_post_{s}", (xf, y - 0.010, 0.0), (xf + 0.020, y + 0.010, r1 + 0.004), "block", "plastic")
        b.box(f"{path}/datum_arm_{s}", (X_DATUM + RAIL_T, y - 0.010, RAIL_Z - 0.008), (xf, y + 0.010, RAIL_Z + 0.008), "block", "plastic")
    # the adjustable rail's brackets: a foot on the table, a 1/2 in post, a cross block with a channel for the rod
    bx0, bx1 = XB - BLOCK[0] / 2, XB + BLOCK[0] / 2
    z0, z1 = RAIL_Z - BLOCK[2] / 2, RAIL_Z + BLOCK[2] / 2
    for s, y in STATIONS.items():
        by0, by1 = y - BLOCK[1] / 2, y + BLOCK[1] / 2
        b.box(f"{path}/foot_{s}", (XB - 0.030, y - 0.030, 0.0), (x0 - FRAME_T, y + 0.030, 0.008), "frame", "steel")
        b.box(f"{path}/post_{s}", (XB - 0.00635, y - 0.00635, 0.008), (XB + 0.00635, y + 0.00635, z0), "post", "steel")
        b.box(f"{path}/block_{s}_under", (bx0, by0, z0), (bx1, by1, RAIL_Z - BORE / 2), "block", "plastic")
        b.box(f"{path}/block_{s}_over", (bx0, by0, RAIL_Z + BORE / 2), (bx1, by1, z1), "block", "plastic")
        b.box(f"{path}/block_{s}_a", (bx0, by0, RAIL_Z - BORE / 2), (bx1, y - BORE / 2, RAIL_Z + BORE / 2), "block", "plastic")
        b.box(f"{path}/block_{s}_b", (bx0, y + BORE / 2, RAIL_Z - BORE / 2), (bx1, by1, RAIL_Z + BORE / 2), "block", "plastic")
        b.cyl(f"{path}/stem_{s}", (XB, y, z1 + KNOB_STEM / 2), 0.007, KNOB_STEM, "rod")
        rx, ry = REST[s]
        b.cyl(f"{path}/rest_{s}", (rx, ry, REST_T / 2), REST_R, REST_T, "rest")
        b.box(f"{path}/rest_{s}_c", (rx - REST_R, ry - REST_R, 0.0), (rx + REST_R, ry + REST_R, REST_T), None, "slide")
    return prim


def build_rail(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=RAIL_MASS, inertia=RAIL_INERTIA)
    ht, hl, hh = RAIL_T / 2, RAIL_LEN / 2, RAIL_H / 2
    b.box(f"{path}/bar", (-ht, -hl, -hh), (ht - 0.003, hl, hh), "rail", "slide")
    b.box(f"{path}/wear", (ht - 0.003, -hl, -hh), (ht, hl, hh), "wear", "slide")
    r = ROD_D / 2
    for s, y in STATIONS.items():
        b.box(f"{path}/rod_{s}_c", (-ht - ROD_LEN, y - r, -r), (-ht, y + r, r), None, "steel")
        b.cyl(f"{path}/rod_{s}", (-ht - ROD_LEN / 2, y, 0.0), r, ROD_LEN, "rod", axis="X")
        b.box(f"{path}/clip_{s}", (-ht - 0.012, y - 0.014, -hh - 0.002), (-ht, y + 0.014, hh + 0.002), "block", "plastic")
    return prim


def build_knob(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    b.rigid(prim, mass=KNOB_MASS, inertia=(6e-6, 6e-6, 9e-6))
    b.prism(f"{path}/grip", (0.0, 0.0, 0.0), KNOB_D - 0.002, KNOB_H, "knob", n=16)
    b.cyl(f"{path}/body", (0.0, 0.0, 0.0), KNOB_D / 2 - 0.004, KNOB_H, "knob")
    for k in range(6):  # six lobes, drawn; a yellow index mark on one, so the knob's turn shows
        a = math.pi / 3 * k
        b.cyl(f"{path}/lobe{k}", ((KNOB_D / 2 - 0.006) * math.cos(a), (KNOB_D / 2 - 0.006) * math.sin(a), 0.0), 0.006, KNOB_H, "knob")
    b.box(f"{path}/index", (0.006, -0.002, KNOB_H / 2), (KNOB_D / 2 - 0.001, 0.002, KNOB_H / 2 + 0.0006), "mark")
    return prim


def build_plug(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    r = PLUG_D / 2
    b.rigid(prim, mass=PLUG_MASS, inertia=(PLUG_MASS * (3 * r * r + PLUG_H ** 2) / 12,) * 2 + (PLUG_MASS * r * r / 2,))
    b.prism(f"{path}/hull", (0.0, 0.0, 0.0), PLUG_D, PLUG_H, "slide")
    b.cyl(f"{path}/body", (0.0, 0.0, 0.0), r, PLUG_H, "plug")
    b.cyl(f"{path}/ring", (0.0, 0.0, PLUG_H / 2), r - 0.008, 0.0006, "wear")
    return prim


def build_can(stage, path):
    b = _Builder(stage, path)
    prim = stage.GetPrimAtPath(path)
    r = CAN_D / 2
    b.rigid(prim, mass=CAN_MASS, inertia=(CAN_MASS * (3 * r * r + CAN_H ** 2) / 12,) * 2 + (CAN_MASS * r * r / 2,))
    b.prism(f"{path}/hull", (0.0, 0.0, 0.0), CAN_D, CAN_H, "slide")
    b.cyl(f"{path}/body", (0.0, 0.0, -0.006), r, CAN_H - 0.012, "can")
    b.cyl(f"{path}/band", (0.0, 0.0, -0.010), r + 0.0002, 0.040, "band")
    b.cyl(f"{path}/neck", (0.0, 0.0, CAN_H / 2 - 0.006), 0.027, 0.012, "lid")
    return prim


BUILDERS = {"conveyor": build_conveyor, "rail": build_rail, "knob_left": build_knob, "knob_right": build_knob,
            "plug_left": build_plug, "plug_right": build_plug, "can": build_can}
JOINTS = ("slide_left", "slide_right", "knob_left", "knob_right")


def _joint(stage, kind, path, conveyor, body, at, local):
    """A joint from the conveyor (fixed) to a loose part: ``at`` in the conveyor's frame (z above the table), ``local``
    in the part's."""
    from pxr import Gf, Sdf

    j = kind.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(conveyor)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, at)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, local)))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0))
    j.CreateCollisionEnabledAttr(True)
    j.CreateExcludeFromArticulationAttr(True)
    return j


def _friction(j, axis, limit, damping):
    """A joint's friction: a velocity drive to rest that can exert at most ``limit`` (N, or N m about a hinge)."""
    from pxr import UsdPhysics

    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(0.0)
    d.CreateDampingAttr(float(damping))
    d.CreateTargetVelocityAttr(0.0)
    d.CreateMaxForceAttr(float(limit))


def add_joints(stage, name, path):
    """The mechanism's joints for the part just built at ``path`` (the conveyor is its sibling prim).
    Rail: each rod in its cross block is a slide along x, and the rail may skew about the vertical; the slide's friction
    limit is the clamp law's holding force. Knob: a hinge about the vertical on its clamp screw, whose friction limit is
    the clamp law's turning torque."""
    from pxr import UsdPhysics

    conveyor = path.rsplit("/", 1)[0] + "/conveyor"
    if name == "rail":
        for s, y in STATIONS.items():
            j = _joint(stage, UsdPhysics.Joint, f"{path}/slide_{s}", conveyor, path, (RAIL_START[0], y, RAIL_Z), (0.0, y, 0.0))
            for axis in ("transZ", "rotY") + (("transY", "rotX") if s == "left" else ()):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)  # low above high: locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transX")
            lim.CreateLowAttr(TRAVEL[0])
            lim.CreateHighAttr(TRAVEL[1])
            if s == "left":
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotZ")
                lim.CreateLowAttr(-SKEW_MAX_DEG)
                lim.CreateHighAttr(SKEW_MAX_DEG)
            _friction(j, "transX", hold_force(A_START), 2.0e5)
            # the clamp locked solid: the slide's x fixed where the rod stands when the clamp bites (anchor set then)
            k = _joint(stage, UsdPhysics.Joint, f"{path}/lock_{s}", conveyor, path, (RAIL_START[0], y, RAIL_Z), (0.0, y, 0.0))
            lim = UsdPhysics.LimitAPI.Apply(k.GetPrim(), "transX")
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
            k.CreateJointEnabledAttr(True)
    elif name.startswith("knob_"):
        s = name.split("_", 1)[1]
        j = _joint(stage, UsdPhysics.RevoluteJoint, f"{path}/{name}", conveyor, path, (XB, STATIONS[s], KNOB_Z), (0.0, 0.0, 0.0))
        j.CreateAxisAttr("Z")
        j.CreateLowerLimitAttr(-(A_MAX - A_START))  # clockwise (tightening) is the negative turn about +z
        j.CreateUpperLimitAttr(A_START - A_MIN)
        _friction(j, "angular", knob_torque(A_START), 20.0)


def controls(stage):
    """What the clamp law sets, found on the stage: the friction limit attribute of each of JOINTS (N for a slide,
    N m for a knob), and each station's lock joint (``lock_left``, ``lock_right``)."""
    from pxr import UsdPhysics

    out = {}
    for prim in stage.Traverse():
        n = prim.GetName()
        if "/envs/" not in str(prim.GetPath()) or not prim.GetTypeName().endswith("Joint"):
            continue
        if n in JOINTS:
            out[n] = prim.GetAttribute(f"drive:{'transX' if n.startswith('slide') else 'angular'}:physics:maxForce")
        elif n in ("lock_left", "lock_right"):
            out[n] = UsdPhysics.Joint(prim)
    return out


def set_lock(joint, side, x):
    """Lock that station's rod where it stands (``x``: the rail's centre line at the station, conveyor frame), or free it
    (``x`` None)."""
    from pxr import Gf

    if x is None:
        joint.GetJointEnabledAttr().Set(False)
        return
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(float(x), float(STATIONS[side]), float(RAIL_Z)))
    joint.GetJointEnabledAttr().Set(True)
