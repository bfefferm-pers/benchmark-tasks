"""The soldering bench: jig, lug, wire, iron and solder stick. Created 11_15_26 (local time).

The constants at the top are NumPy only, so the task file and the tests use them without Isaac Sim; the USD builders
import pxr when called. All lengths in metres, true size.

Frames:
- Bench: x, y as the world's, z from the station table's top. One kinematic body: a steel base plate, a phenolic post
  with the clamp bolt that holds the lug's tongue, a loose wire cradle, the iron's stand and the solder stick's rest.
- Lug: origin at the centre of the barrel's mouth, +x into the barrel (toward the tongue). Its own kinematic body, so
  the simulator reports contacts with the lug separately from the jig. A solder-type lug: the ILSCO copper soldering lug,
  catalogue number 1/4 (8 AWG, one hole, #10 bolt), from the maker's catalogue table: seamless 1/4 in copper tube,
  barrel 6.35 mm outside, 4.76 mm bore (3/16 in), 8.73 mm long (11/32 in); tongue 9.5 mm wide (3/8 in), 1.6 mm thick;
  26.2 mm overall (1-1/32 in); hole 5.16 mm (13/64 in), its centre 5.56 mm (7/32 in) from the tongue's end.
- Wire: origin on the conductor's axis where it crosses the mouth when seated, +x into the barrel. 150 mm of 8 AWG
  19-strand copper building wire: conductor 3.63 mm, insulation 5.5 mm outside, stripped 16.2 mm and pre-tinned.
- Iron: origin at the centre of its grip block, +x toward the tip. A 67 W pencil iron, 6.5 mm chisel tip with 18 mm
  exposed; the grip block sits on top of the handle so that a downward grip closes on its front and back faces.
- Solder: origin at the centre of its grip sleeve, +x its heading in plan. A straight 1.5 mm stick of solder wire
  leaves the sleeve's underside sloping 20 degrees down.
Colliders are boxes: a round part is a square bar of the same height (the barrel's top and the tip's underside are at
their true heights); the bore is a square channel 4.2 mm across round a 3.3 mm square conductor (0.45 mm radial play;
the real parts have 0.57 mm).
"""

import math

import numpy as np

# -- lug (lug frame) -------------------------------------------------------------------------------------------------
B_OD, B_ID, B_LEN = 0.00635, 0.00476, 0.00873
BORE, COND_SQ = 0.0042, 0.0033  # collider bore and conductor, square
STOP = 0.0028  # the flattened transition closing the barrel's far end
TONGUE = dict(x0=B_LEN + STOP, x1=0.0262, w=0.0095, t=0.0016)
HOLE_X, HOLE_D = 0.0262 - 0.00556, 0.00516
LUG_BOXES = {  # name: (low, high)
    "top": ((0.0, -0.0025, BORE / 2), (B_LEN, 0.0025, B_OD / 2)),
    "bottom": ((0.0, -0.0025, -B_OD / 2), (B_LEN, 0.0025, -BORE / 2)),
    "left": ((0.0, BORE / 2, -BORE / 2), (B_LEN, B_OD / 2, BORE / 2)),
    "right": ((0.0, -B_OD / 2, -BORE / 2), (B_LEN, -BORE / 2, BORE / 2)),
    "stop": ((B_LEN, -B_OD / 2, -B_OD / 2), (B_LEN + STOP, B_OD / 2, B_OD / 2)),
    "tongue": ((TONGUE["x0"], -TONGUE["w"] / 2, -TONGUE["t"] / 2), (TONGUE["x1"], TONGUE["w"] / 2, TONGUE["t"] / 2)),
}

# -- wire (wire frame) -----------------------------------------------------------------------------------------------
COND_D, INS_D, WIRE_LEN = 0.00363, 0.0055, 0.150
BARE = 0.008  # bare conductor showing outside the mouth (limit: two insulated-wire diameters, 11 mm)
SEAT = 0.0082  # conductor end, inside the barrel, at the start (0.5 mm short of the stop)
INS_SQ = 0.0050
WIRE_MASS = 0.014
WIRE_COM = np.array((-0.068, 0.0, 0.0))
WIRE_DROP = (BORE - COND_SQ) / 2  # the conductor lies on the bore's floor, this far below the lug's axis
WIRE_BOXES = {
    "cond": ((-BARE, -COND_SQ / 2, -COND_SQ / 2), (SEAT, COND_SQ / 2, COND_SQ / 2)),
    "ins": ((SEAT - WIRE_LEN, -INS_SQ / 2, -INS_SQ / 2), (-BARE, INS_SQ / 2, INS_SQ / 2)),
}

# -- jig (bench frame) -----------------------------------------------------------------------------------------------
MOUTH = np.array((0.270, 0.0))  # the barrel mouth's centre in plan; the barrel runs along +x
AXIS_Z = 0.045  # the lug's axis above the table
BASE = dict(x=(0.105, 0.335), y=(-0.035, 0.035), t=0.006)
POST = dict(x=(MOUTH[0] + 0.0130, MOUTH[0] + 0.0290), y=(-0.011, 0.011))
CRADLE = dict(x=(0.150, 0.170), gap=INS_SQ + 0.0012, cheek=0.004)

# -- iron (iron frame) -----------------------------------------------------------------------------------------------
GRIP = dict(x=0.030, y=0.036, z=0.026)
HANDLE = dict(r=0.011, x0=-0.070, x1=0.050)
HEATER = dict(r=0.0045, x0=0.050, x1=0.127)
TIP_D, TIP_X0, TIP_X1 = 0.0065, 0.127, 0.145  # the chisel tip's exposed 18 mm
CORD = dict(r=0.004, x0=-0.105, x1=-0.070)
IRON_Z = -(GRIP["z"] / 2 + HANDLE["r"])  # the iron's axis below the grip block's centre
TIP_MID = np.array(((TIP_X0 + TIP_X1) / 2, 0.0, IRON_Z))
TIP_A, TIP_B = np.array((TIP_X0, 0.0, IRON_Z)), np.array((TIP_X1, 0.0, IRON_Z))
IRON_MASS = 0.085
IRON_COM = np.array((0.004, 0.0, -0.020))
IRON_BOXES = {
    "grip": ((-GRIP["x"] / 2, -GRIP["y"] / 2, -GRIP["z"] / 2), (GRIP["x"] / 2, GRIP["y"] / 2, GRIP["z"] / 2)),
    "handle": ((HANDLE["x0"], -HANDLE["r"], IRON_Z - HANDLE["r"]), (HANDLE["x1"], HANDLE["r"], IRON_Z + HANDLE["r"])),
    "heater": ((HEATER["x0"], -HEATER["r"], IRON_Z - HEATER["r"]), (HEATER["x1"], HEATER["r"], IRON_Z + HEATER["r"])),
    "tip": ((TIP_X0, -TIP_D / 2, IRON_Z - TIP_D / 2), (TIP_X1, TIP_D / 2, IRON_Z + TIP_D / 2)),
}
IRON_HULL = (np.array((CORD["x0"], -GRIP["y"] / 2, IRON_Z - HANDLE["r"])), np.array((TIP_X1, GRIP["y"] / 2, GRIP["z"] / 2)))

# -- solder stick (solder frame) ---------------------------------------------------------------------------------------
SLEEVE = 0.025
SOLDER_D = 0.0015
SLOPE = math.radians(20.0)
STICK = 0.130  # sleeve centre to tip, along the stick
DROP = 0.010
S_DIR = np.array((math.cos(SLOPE), 0.0, -math.sin(SLOPE)))
S_TIP = STICK * S_DIR - (0.0, 0.0, DROP)
S_BACK = -0.008 * S_DIR - (0.0, 0.0, DROP)
S_EXIT = (SLEEVE / 2 - DROP) / math.sin(SLOPE) * S_DIR - (0.0, 0.0, DROP)
SOLDER_MASS = 0.021
SLEEVE_BOX = ((-SLEEVE / 2,) * 3, (SLEEVE / 2,) * 3)

# -- rests (bench frame) -----------------------------------------------------------------------------------------------
IRON_REST = dict(x=0.200, y=0.250, heading=-math.pi / 2)  # grip centre; the tip points along -y, toward the work
IRON_POST_H = 0.045
IRON_REST_Z = IRON_POST_H + HANDLE["r"] - IRON_Z
IRON_POSTS = ((-0.058, -0.038), (0.018, 0.038))  # under the handle, along the iron's axis from the grip centre
SOLDER_REST = dict(x=0.200, y=-0.260, heading=math.pi / 2)  # sleeve centre; the stick points along +y
SOLDER_REST_Z = 0.060
SOLDER_POST_X = (-0.011, 0.002)
SOLDER_FORK_X = 0.040  # a support under the stick, this far ahead of the sleeve centre (plan)
SOLDER_FORK_Z = SOLDER_REST_Z - DROP - SOLDER_FORK_X * math.tan(SLOPE) - SOLDER_D / 2

COLORS = {  # sRGB, metallic, roughness
    "steel": ((0.36, 0.37, 0.39), 0.9, 0.45), "phenolic": ((0.55, 0.36, 0.18), 0.0, 0.6), "bolt": ((0.62, 0.63, 0.65), 1.0, 0.3),
    "copper": ((0.80, 0.45, 0.28), 1.0, 0.3), "tinned": ((0.78, 0.79, 0.80), 1.0, 0.3), "ins": ((0.75, 0.08, 0.08), 0.0, 0.5),
    "grip": ((0.90, 0.45, 0.05), 0.0, 0.55), "handle": ((0.08, 0.20, 0.50), 0.0, 0.5), "heater": ((0.70, 0.70, 0.72), 1.0, 0.35),
    "tip": ((0.82, 0.82, 0.84), 1.0, 0.2), "cord": ((0.04, 0.04, 0.04), 0.0, 0.7), "sleeve": ((0.10, 0.50, 0.25), 0.0, 0.5),
    "white": ((0.95, 0.95, 0.92), 0.0, 0.7), "solder": ((0.86, 0.87, 0.90), 1.0, 0.25), "fillet": ((0.88, 0.89, 0.92), 1.0, 0.12),
    "cold": ((0.45, 0.45, 0.44), 0.4, 0.95), "char": ((0.05, 0.04, 0.03), 0.0, 0.95), "mat": ((0.16, 0.20, 0.26), 0.0, 0.9),
}
MU = {"steel": 0.5, "grip": 1.0, "rubber": 0.9, "copper": 0.4}
FX = ("fx_lean", "fx_fill", "fx_fat", "fx_cold", "fx_char", "fx_wet")  # on the lug, switched from the task's record


def _physx(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


class Kit:
    """Boxes, cylinders and cones under one body prim, with looks and friction."""

    def __init__(self, stage, root):
        self.stage, self.root, self.looks, self.phys = stage, root, {}, {}

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
            self.phys[name] = mat
        return self.phys[name]

    def _finish(self, g, look, collide, offset, show=True):
        from pxr import UsdGeom, UsdPhysics, UsdShade

        prim = g.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(look))
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            _physx(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": float(offset), "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide), UsdShade.Tokens.weakerThanDescendants, "physics")
        if not show:
            UsdGeom.Imageable(prim).MakeInvisible()
        return g

    def box(self, name, lo, hi, look, collide=None, rot_y=0.0, offset=0.0006, show=True):
        from pxr import Gf, UsdGeom

        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, f"{self.root}/{name}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, (lo + hi) / 2)))
        if rot_y:
            c.AddRotateYOp().Set(float(rot_y))
        c.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
        return self._finish(c, look, collide, offset, show)

    def cyl(self, name, center, r, h, look, axis="X", rot_y=0.0, show=True):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, f"{self.root}/{name}")
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if rot_y:
            c.AddRotateYOp().Set(float(rot_y))
        return self._finish(c, look, None, 0.0, show)

    def cone(self, name, x0, r0, x1, r1, look, show=True, n=20, z=0.0):
        """A closed frustum along x from x0 (radius r0) to x1 (radius r1), on the axis at height z."""
        from pxr import Gf, UsdGeom, Vt

        m = UsdGeom.Mesh.Define(self.stage, f"{self.root}/{name}")
        pts = [Gf.Vec3f(x, r * math.cos(2 * math.pi * k / n), z + r * math.sin(2 * math.pi * k / n)) for x, r in ((x0, r0), (x1, r1)) for k in range(n)]
        pts += [Gf.Vec3f(x0, 0, z), Gf.Vec3f(x1, 0, z)]
        counts, idx = [], []
        for k in range(n):
            j = (k + 1) % n
            counts += [4, 3, 3]
            idx += [k, j, n + j, n + k, 2 * n, j, k, 2 * n + 1, n + k, n + j]
        m.CreatePointsAttr(Vt.Vec3fArray(pts))
        m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDoubleSidedAttr(True)
        return self._finish(m, look, None, 0.0, show)

    def rigid(self, kinematic=False, mass=None, com=None, report=False):
        from pxr import Gf, PhysxSchema, UsdPhysics

        prim = self.stage.GetPrimAtPath(self.root)
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _physx(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 32, "physxRigidBody:solverVelocityIterationCount": 2,
                                             "physxRigidBody:maxDepenetrationVelocity": 0.2, "physxRigidBody:enableCCD": not kinematic})
        if mass is not None:
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(float(mass))
            if com is not None:
                m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        if report:
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        return prim


def build_lug(stage, path):
    k = Kit(stage, path)
    prim = k.rigid(kinematic=True)
    for name, (lo, hi) in LUG_BOXES.items():
        k.box(f"c_{name}", lo, hi, "copper", "copper", offset=0.0004, show=name == "tongue")
    k.cyl("v_barrel", (B_LEN / 2, 0, 0), B_OD / 2, B_LEN, "copper")
    k.cone("v_neck", B_LEN, B_OD / 2, B_LEN + STOP, TONGUE["t"] / 2 + 0.0004, "copper")
    k.cyl("v_bore", (0.0001, 0, 0), B_ID / 2, 0.0004, "char")
    # the joint as the task's record has it: a fillet round the conductor at the mouth (lean, full or fat), dull if
    # cold, a charred ring if overheated, and a bright ring while liquid
    r = COND_D / 2
    z = -WIRE_DROP
    k.cone("fx_lean", -0.0012, r + 0.0002, 0.0003, B_ID / 2, "fillet", show=False, z=z / 2)
    k.cone("fx_fill", -0.0030, r + 0.0003, 0.0003, B_OD / 2 - 0.0004, "fillet", show=False, z=z / 2)
    k.cone("fx_fat", -0.0060, B_OD / 2 + 0.0008, 0.0010, B_OD / 2 + 0.0012, "fillet", show=False)
    k.cone("fx_cold", -0.0032, r + 0.0006, 0.0004, B_OD / 2 - 0.0002, "cold", show=False, z=z / 2)
    k.cone("fx_char", 0.0004, B_OD / 2 + 0.0002, 0.0030, B_OD / 2 + 0.0002, "char", show=False)
    k.cone("fx_wet", 0.0002, B_OD / 2 + 0.0003, 0.0012, B_OD / 2 + 0.0003, "white", show=False)
    return prim


def build_wire(stage, path, lug_path):
    from pxr import Gf, UsdPhysics

    k = Kit(stage, path)
    prim = k.rigid(mass=WIRE_MASS, com=WIRE_COM)
    for name, (lo, hi) in WIRE_BOXES.items():
        k.box(f"c_{name}", lo, hi, "ins", "copper" if name == "cond" else "rubber", offset=0.0004, show=False)
    k.cyl("v_cond", ((SEAT - BARE) / 2, 0, 0), COND_D / 2, SEAT + BARE, "tinned")
    k.cyl("v_ins", ((SEAT - WIRE_LEN - BARE) / 2, 0, 0), INS_D / 2, WIRE_LEN - SEAT - BARE, "ins")
    k.cone("fx_ins", -BARE - 0.006, INS_D / 2 + 0.0003, -BARE, INS_D / 2 + 0.0003, "char", show=False)
    # the soldered joint: a fixed joint from the lug to the wire, off until a joint sets (the embodiment switches it)
    j = UsdPhysics.FixedJoint.Define(stage, f"{path}/soldered")
    j.CreateBody0Rel().SetTargets([lug_path])
    j.CreateBody1Rel().SetTargets([path])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(-WIRE_DROP)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateJointEnabledAttr(False)
    j.CreateExcludeFromArticulationAttr(True)
    return prim


def build_iron(stage, path):
    k = Kit(stage, path)
    prim = k.rigid(mass=IRON_MASS, com=IRON_COM, report=True)
    for name, (lo, hi) in IRON_BOXES.items():
        k.box(f"c_{name}", lo, hi, "grip", "grip" if name == "grip" else "steel", show=name == "grip")
    k.cyl("v_handle", ((HANDLE["x0"] + HANDLE["x1"]) / 2, 0, IRON_Z), HANDLE["r"], HANDLE["x1"] - HANDLE["x0"], "handle")
    k.cyl("v_guard", (HANDLE["x1"] - 0.004, 0, IRON_Z), HANDLE["r"] + 0.003, 0.004, "cord")
    k.cyl("v_heater", ((HEATER["x0"] + HEATER["x1"]) / 2, 0, IRON_Z), HEATER["r"], HEATER["x1"] - HEATER["x0"], "heater")
    k.cyl("v_tip", ((TIP_X0 + TIP_X1 - 0.004) / 2, 0, IRON_Z), TIP_D / 2, TIP_X1 - TIP_X0 - 0.004, "tip")
    k.cone("v_chisel", TIP_X1 - 0.004, TIP_D / 2, TIP_X1, 0.0008, "tip", z=IRON_Z)
    k.cyl("v_cord", ((CORD["x0"] + CORD["x1"]) / 2, 0, IRON_Z), CORD["r"], CORD["x1"] - CORD["x0"], "cord")
    k.cone("fx_blob", TIP_X1 - 0.010, TIP_D / 2 + 0.0012, TIP_X1 - 0.003, TIP_D / 2 + 0.0004, "fillet", show=False, z=IRON_Z)
    return prim


def build_solder(stage, path):
    k = Kit(stage, path)
    prim = k.rigid(mass=SOLDER_MASS, report=True)
    k.box("c_sleeve", *SLEEVE_BOX, "sleeve", "grip")
    mid, L, deg = (S_BACK + S_TIP) / 2, float(np.linalg.norm(S_TIP - S_BACK)), math.degrees(SLOPE)
    k.box("c_stick", mid - (L / 2, SOLDER_D / 2, SOLDER_D / 2), mid + (L / 2, SOLDER_D / 2, SOLDER_D / 2), "solder", "steel", rot_y=deg, offset=0.0004, show=False)
    k.cyl("v_stick", mid, SOLDER_D / 2, L, "solder", rot_y=deg)
    k.box("v_band", (-SLEEVE / 2 - 0.0005, -SLEEVE / 2 - 0.0005, -0.003), (SLEEVE / 2 + 0.0005, SLEEVE / 2 + 0.0005, 0.003), "white")
    return prim


def build_bench(stage, path):
    k = Kit(stage, path)
    prim = k.rigid(kinematic=True)
    (bx0, bx1), (by0, by1), bt = BASE["x"], BASE["y"], BASE["t"]
    k.box("base", (bx0, by0, 0.0), (bx1, by1, bt), "steel", "steel")
    # the post under the tongue and the #10 screw and washer through the stud hole
    (px0, px1), (py0, py1) = POST["x"], POST["y"]
    k.box("post", (px0, py0, bt), (px1, py1, AXIS_Z - TONGUE["t"] / 2), "phenolic", "steel")
    hx, top = MOUTH[0] + HOLE_X, AXIS_Z + TONGUE["t"] / 2
    k.cyl("washer", (hx, 0, top + 0.0005), 0.0055, 0.0010, "bolt", axis="Z")
    k.cyl("bolt", (hx, 0, top + 0.0030), 0.0040, 0.0040, "bolt", axis="Z")
    k.box("c_bolt", (hx - 0.004, -0.004, top), (hx + 0.004, 0.004, top + 0.0050), "bolt", "steel", show=False)
    # the wire cradle: a floor and two cheeks, open on top, nothing clamped
    (cx0, cx1), g, ch = CRADLE["x"], CRADLE["gap"], CRADLE["cheek"]
    floor = AXIS_Z - WIRE_DROP - INS_SQ / 2
    k.box("cradle", (cx0, -g / 2 - ch, bt), (cx1, g / 2 + ch, floor), "phenolic", "rubber")
    for s, n in ((1, "a"), (-1, "b")):
        k.box(f"cheek_{n}", (cx0, min(s * g / 2, s * (g / 2 + ch)), floor), (cx1, max(s * g / 2, s * (g / 2 + ch)), floor + 0.004), "phenolic", "rubber")
    # the iron's stand: two forks under the handle on a steel foot, the tip in the air
    ix, iy = IRON_REST["x"], IRON_REST["y"]
    for n, (a0, a1) in enumerate(IRON_POSTS):
        k.box(f"iron_post{n}", (ix - 0.016, iy - a1, 0.0), (ix + 0.016, iy - a0, IRON_POST_H), "steel", "steel")
    k.box("iron_foot", (ix - 0.025, iy - 0.045, 0.0), (ix + 0.025, iy + 0.065, 0.004), "steel", "steel")
    # the solder stick's rest: a post under the sleeve and a fork under the stick
    sx, sy = SOLDER_REST["x"], SOLDER_REST["y"]
    k.box("solder_post", (sx - 0.016, sy + SOLDER_POST_X[0], 0.0), (sx + 0.016, sy + SOLDER_POST_X[1], SOLDER_REST_Z - SLEEVE / 2), "steel", "steel")
    k.box("solder_fork", (sx - 0.010, sy + SOLDER_FORK_X - 0.004, 0.0), (sx + 0.010, sy + SOLDER_FORK_X + 0.004, SOLDER_FORK_Z), "steel", "rubber")
    return prim
