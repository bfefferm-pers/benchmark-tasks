"""An OXO Good Grips stainless steel steamer with extendable handle (model 1067247) and a Vigor SS1 4.5 qt stainless
sauce pan (473SSAUCP4), both true size. Created 11_17_26 (America/New_York).

Geometry is NumPy only, so the task file and the tests use it without Isaac Sim; the USD builders import pxr when called.

Sources (README "Where the numbers come from"):
- Steamer: closed 7 in across x 3 in tall, open 10.5 in across x 5 in tall (OXO's answer on the Target listing; the same
  figures on the Fante's and Conway Kitchen listings); 0.69 lb (Target specification tab); legs 1.5 in tall (Target
  Q&A; America's Test Kitchen's specification list); handle 4 in extended and 2.5 in collapsed (America's Test Kitchen
  review); fits pots 8 in across and up (Williams-Sonoma, Sizzle and Sear).
- Pan: top diameter 8 in, bottom diameter 7 1/16 in, height 5 1/4 in, handle 7 7/8 in, 22 gauge stainless, 4 mm
  aluminium-clad bottom (Vigor's sauce pan specification sheet and the WebstaurantStore listing).
- Not published, chosen and stated: 16 leaves (counted on the retailer's product photo), each 47 mm long, 36 mm wide at the
  hinge and 54 mm at the tip, twisted 12 degrees so each overlaps the next, and hinged on a 168 mm circle (the simplest
  fit to the published closed and open diameters with the leaves resting 15 degrees above level when open); three wire feet; the T-bar 74 mm long (scaled from the product photo); every friction value.

Steamer frame (m): origin at the centre of the base plate's underside, z up; the feet stand 38 mm below it. Leaf k's
hinge sits at angle 2 pi k / N on the plate's rim; its joint angle is 0 at the open stop (the leaf 15 degrees above
level) and grows as the leaf folds up, to FOLD_MAX (100 degrees above level, leaning in). The handle slides on the post
along z: 0 is fully collapsed (T-bar top 64 mm above the plate), HANDLE_TRAVEL fully extended (102 mm).

Pan frame (m): origin at the middle of the pan's outside bottom on the table, z up; the handle points along +x.
"""

import math

import numpy as np

IN = 0.0254

# -- the steamer (OXO 1067247) ------------------------------------------------------------------------------------------
CLOSED_D = 7.0 * IN  # 177.8 mm, published
OPEN_D = 10.5 * IN  # 266.7 mm, published
MASS_TOTAL = 0.69 * 0.45359237  # 313 g, published
LEG_H = 1.5 * IN  # 38.1 mm: the plate's underside above the table, published leg height
HANDLE_UP = 4.0 * IN  # T-bar top above the plate, extended (ATK)
HANDLE_DOWN = 2.5 * IN  # collapsed (ATK)
HANDLE_TRAVEL = HANDLE_UP - HANDLE_DOWN  # 38.1 mm
N_LEAVES = 16
PLATE_T = 0.0008  # 430 stainless sheet, with the leaf knuckles
R_HINGE = 0.084  # hinge circle radius: 168 mm across; with the knuckles and the folded leaves, 178 mm closed
LEAF_LEN = 0.047
LEAF_T = 0.0012  # collider thickness (the sheet is about 0.5 mm; 1.2 mm keeps PhysX contacts stable)
LEAF_W_IN = 0.036  # at the hinge (33 mm of hinge circle per leaf, plus the overlap onto the next leaf)
LEAF_W_OUT = 0.054  # at the tip, with the overlap onto the next leaf
# Each leaf is twisted this far about its own length, its leading edge (toward leaf k+1) raised, so it lies over the
# next leaf's trailing edge all round, as the real leaves are shingled. With this twist and these widths, neighbours
# fold together from open (15 degrees above level) to fully closed (100 degrees) without touching (test_score.py
# test_leaves_fold_closed_without_touching), and touch, push and slide on one another when one folds ahead of the next.
LEAF_TWIST_DEG = 12.0
OPEN_DEG = 15.0  # an open leaf rests this far above level: 268 mm across the open tips' corners
FOLD_MAX_DEG = 85.0  # joint range: up to 100 degrees above level
POST_R = 0.006  # fixed lower tube
HANDLE_R = 0.0048  # sliding upper tube
POST_TOP = 0.060  # the fixed tube's top above the plate
TBAR = dict(len=0.074, w=0.014, h=0.012)  # length, width across, height
FOOT_R = 0.060  # wire feet: radius of each foot's leg from the plate's centre
FOOT_WIRE = 0.004
FOOT_SPAN = 0.030  # each foot's bottom bar, tangential
MASS = dict(plate=0.075, post=0.030, feet=0.035, handle=0.045)
MASS["leaf"] = (MASS_TOTAL - sum(MASS.values())) / N_LEAVES  # 8 g each
# Coulomb friction (PhysX joint friction). Estimates: a leaf's rivet hinge turns freely (it falls open under its own
# 0.002 N m of weight); the handle's push lock is modelled as friction that holds the steamer's own weight when carried
# collapsed and gives to a 3 N push (README, Reset, gives the spring-scale test).
LEAF_FRICTION_NM = 0.0004
HANDLE_FRICTION_N = 3.0
DYNAMIC_FRACTION = 0.8
MU_STEEL = 0.3

# -- the pan (Vigor SS1 473SSAUCP4) ---------------------------------------------------------------------------------
PAN_TOP_D = 8.0 * IN  # 203.2 mm across the top, published
PAN_BOT_D = (7 + 1 / 16) * IN  # 179.4 mm across the bottom, published
PAN_H = 5.25 * IN  # 133.4 mm, published
PAN_HANDLE_LEN = (7 + 7 / 8) * IN  # 200 mm, published
PAN_WALL = 0.0008  # 22 gauge
PAN_FLOOR = 0.005  # 4 mm aluminium disc under 0.8 mm steel: the inside floor's height above the table
PAN_WALL_COLLIDE = 0.003
PAN_SEGMENTS = 48
PAN_MASS = 1.30  # estimate: 4 lb shipped with its cover and box
MU_TABLE = 0.4


def pan_inner_r(z):
    """Inside radius of the pan's wall at height z (m) above the table (the published diameters are outside)."""
    t = float(np.clip(z / PAN_H, 0.0, 1.0))
    return PAN_BOT_D / 2 + t * (PAN_TOP_D - PAN_BOT_D) / 2 - PAN_WALL


RIM_R = PAN_TOP_D / 2  # 101.6 mm
WALL_TILT = math.atan2((PAN_TOP_D - PAN_BOT_D) / 2, PAN_H)  # 5.1 degrees out from vertical


def leaf_hinge(k):
    """Hinge point and unit radial and tangential directions of leaf k, in the steamer frame."""
    a = 2 * math.pi * k / N_LEAVES
    radial = np.array((math.cos(a), math.sin(a), 0.0))
    tangent = np.array((-math.sin(a), math.cos(a), 0.0))
    return R_HINGE * radial + np.array((0.0, 0.0, PLATE_T)), radial, tangent


def leaf_dir(k, q):
    """Unit direction from leaf k's hinge to its tip at joint angle q (rad), in the steamer frame."""
    _, radial, _ = leaf_hinge(k)
    b = math.radians(OPEN_DEG) + q
    return math.cos(b) * radial + np.array((0.0, 0.0, math.sin(b)))


def leaf_tip(k, q):
    """The middle of leaf k's tip edge at joint angle q, in the steamer frame."""
    h, _, _ = leaf_hinge(k)
    return h + LEAF_LEN * leaf_dir(k, q)


def leaf_local_points():
    """The 8 corners of a leaf's (twisted, trapezoidal) sheet in its body frame: x from the hinge to the tip, y along the
    hinge (toward leaf k+1), z its normal (up, into the bowl, when open). Order: hinge-end then tip-end, each
    (-y bottom, -y top, +y bottom, +y top)."""
    g = math.radians(LEAF_TWIST_DEG)
    ty = np.array((0.0, math.cos(g), math.sin(g)))
    nz = np.array((0.0, -math.sin(g), math.cos(g)))
    pts = []
    for s, w in ((0.0, LEAF_W_IN / 2), (LEAF_LEN, LEAF_W_OUT / 2)):
        for side in (-1, 1):
            for z in (0.0, LEAF_T):
                pts.append(np.array((s, 0.0, 0.0)) + side * w * ty + z * nz)
    return np.array(pts)


def leaf_rot(k, q):
    """Rotation of leaf k's body frame in the steamer frame at joint angle q: columns (to the tip, hinge, normal)."""
    d = leaf_dir(k, q)
    _, _, t = leaf_hinge(k)
    return np.column_stack((d, t, np.cross(d, t)))


def leaf_points(k, q):
    """The 8 corners of leaf k's sheet at joint angle q, in the steamer frame."""
    h, _, _ = leaf_hinge(k)
    return h + leaf_local_points() @ leaf_rot(k, q).T


def leaf_tip_corners(k, q):
    """The 4 corners of leaf k's tip edge (both sides, both faces) at joint angle q, in the steamer frame: the points of
    the leaf furthest out and, folded, highest."""
    return leaf_points(k, q)[4:]


def tbar_top(slide):
    """Height of the T-bar's top above the plate's underside for a handle slide (m, 0 collapsed)."""
    return HANDLE_DOWN + float(np.clip(slide, 0.0, HANDLE_TRAVEL))


def open_diameter():
    """Across the open leaves' tip corners."""
    return 2 * max(float(np.hypot(p[0], p[1])) for k in range(N_LEAVES) for p in leaf_tip_corners(k, 0.0))


def closed_diameter():
    """Across the folded steamer: the hinge circle plus the knuckles and the folded leaves' thickness."""
    return 2 * (R_HINGE + 0.0035 + LEAF_T)


# -- USD ----------------------------------------------------------------------------------------------------------
COLORS = {  # sRGB base colour, metallic, roughness
    "steel": ((0.78, 0.79, 0.80), 1.0, 0.28), "pan": ((0.72, 0.73, 0.74), 1.0, 0.22),
    "black": ((0.07, 0.07, 0.08), 0.0, 0.5), "dark": ((0.30, 0.30, 0.32), 1.0, 0.35),
}


def _schema(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def _quat_from_matrix(r):
    """(w, x, y, z) of a rotation matrix."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((0.25 * s), (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        return ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    if i == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        return ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    return ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)


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

    def surface(self, name, mu):
        from pxr import UsdPhysics, UsdShade

        if name not in self.phys:
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
            pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm.CreateStaticFrictionAttr(mu)
            pm.CreateDynamicFrictionAttr(mu)
            pm.CreateRestitutionAttr(0.0)
            self.phys[name] = mat
        return self.phys[name]

    def _finish(self, g, look, collide, mu=MU_STEEL, visible=True):
        from pxr import UsdGeom, UsdPhysics, UsdShade

        prim = g.GetPrim()
        if look:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(look))
        if not visible:
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            _schema(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.0015, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide, mu), UsdShade.Tokens.weakerThanDescendants,
                                                         "physics")
        return g

    def box(self, path, center, size, look, collide=None, rot=None, mu=MU_STEEL, visible=True):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if rot is not None:
            rot = np.asarray(rot, float)
            if np.linalg.det(rot) < 0.999:
                raise ValueError(f"{path}: rotation is not proper")
            w, x, y, z = _quat_from_matrix(rot)
            c.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
        c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        return self._finish(c, look, collide, mu, visible)

    def cyl(self, path, center, r, h, look, collide=None, axis="Z", mu=MU_STEEL):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        return self._finish(c, look, collide, mu)

    def mesh(self, path, points, faces, look):
        from pxr import Gf, UsdGeom

        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
        m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr("none")
        return self._finish(m, look, None)

    def sheet(self, path, corners, look, collide, mu=MU_STEEL):
        """A thin convex slab from its 8 corners (leaf_local_points order), drawn and, as its convex hull, collided."""
        from pxr import Gf, UsdGeom, UsdPhysics

        faces = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6)]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in corners])
        m.CreateFaceVertexCountsAttr([4] * len(faces))
        m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr("none")
        self._finish(m, look, collide, mu)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        return m

    def rigid(self, prim, mass, com, inertia):
        from pxr import Gf, UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim)
        _schema(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 32,
                                              "physxRigidBody:solverVelocityIterationCount": 4,
                                              "physxRigidBody:maxDepenetrationVelocity": 0.3,
                                              "physxRigidBody:angularDamping": 0.05, "physxRigidBody:linearDamping": 0.05})
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(mass))
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*(max(float(v), 1e-7) for v in inertia)))


def _xform(stage, path, pos=(0.0, 0.0, 0.0), yaw=0.0):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if yaw:
        x.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    return x.GetPrim()


def _joint(stage, kind, path, body0, body1, pos0, rot0, pos1, rot1, axis, lo, hi):
    from pxr import Gf, Sdf, UsdPhysics

    j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos1)))
    j.CreateLocalRot0Attr(Gf.Quatf(*map(float, rot0)))
    j.CreateLocalRot1Attr(Gf.Quatf(*map(float, rot1)))
    j.CreateLowerLimitAttr(float(lo))
    j.CreateUpperLimitAttr(float(hi))
    return j.GetPrim()


def _leaf_frame(k):
    """Rotation of leaf k's body frame at joint angle 0, in the steamer frame: x along the leaf (hinge to tip), y along
    the hinge axis (tangential), z the leaf's normal. Folding up is a positive turn about -y (the joint in build_steamer)."""
    return leaf_rot(k, 0.0)


def build_steamer(stage, path, xy=(0.0, 0.0), yaw=0.0):
    """The steamer under `path`, standing on its feet on the table at `xy` (in the parent Xform's frame), turned `yaw`
    about z. One articulation: the base plate (with the fixed post and the feet) is its free root; the 16 leaves hang
    from it on revolute hinges and the handle on a prismatic slide. Joints start at 0 (leaves open, handle collapsed);
    the embodiment sets the start pose through the joints."""
    from pxr import UsdPhysics

    b = _Builder(stage, path)
    root = _xform(stage, path, (xy[0], xy[1], LEG_H), yaw)
    plate = f"{path}/plate"
    pp = _xform(stage, plate)
    m_root = MASS["plate"] + MASS["post"] + MASS["feet"]
    r = R_HINGE
    b.rigid(pp, m_root, (0.0, 0.0, -0.004), (m_root * r * r / 4, m_root * r * r / 4, m_root * r * r / 2))
    UsdPhysics.ArticulationRootAPI.Apply(pp)
    _schema(pp, "PhysxArticulationAPI", **{"physxArticulation:solverPositionIterationCount": 32,
                                           "physxArticulation:solverVelocityIterationCount": 4,
                                           "physxArticulation:enabledSelfCollisions": True})  # leaf on leaf
    b.cyl(f"{plate}/disc", (0.0, 0.0, PLATE_T / 2), R_HINGE, PLATE_T, "steel", "steel")
    b.cyl(f"{plate}/rimring", (0.0, 0.0, PLATE_T + 0.0015), R_HINGE + 0.002, 0.003, "dark", None)
    b.cyl(f"{plate}/post", (0.0, 0.0, POST_TOP / 2), POST_R, POST_TOP, "steel", "steel")
    for i in range(3):
        a = 2 * math.pi * i / 3 + math.pi / 6
        c = np.array((math.cos(a), math.sin(a), 0.0))
        t = np.array((-math.sin(a), math.cos(a), 0.0))
        rot = np.column_stack((t, -c, (0.0, 0.0, 1.0)))  # right-handed: t x (-c) = z
        for s in (-1, 1):
            p = FOOT_R * c + s * (FOOT_SPAN / 2) * t + np.array((0.0, 0.0, -LEG_H / 2))
            b.box(f"{plate}/foot{i}_leg{'ab'[s > 0]}", p, (FOOT_WIRE, FOOT_WIRE, LEG_H), "steel", "steel", rot)
        b.box(f"{plate}/foot{i}_bar", FOOT_R * c + np.array((0.0, 0.0, -LEG_H + FOOT_WIRE / 2)),
              (FOOT_SPAN + FOOT_WIRE, FOOT_WIRE, FOOT_WIRE), "steel", "steel", rot)
    # handle: sliding tube and T-bar; link frame at the plate's origin, built collapsed
    hl = f"{path}/handle"
    hp = _xform(stage, hl)
    top = HANDLE_DOWN
    b.rigid(hp, MASS["handle"], (0.0, 0.0, top - 0.02), (MASS["handle"] * 0.003, MASS["handle"] * 0.003, 2e-6))
    tube_len = top - TBAR["h"] - 0.012
    b.cyl(f"{hl}/tube", (0.0, 0.0, 0.012 + tube_len / 2), HANDLE_R, tube_len, "steel", "steel")
    b.box(f"{hl}/tbar", (0.0, 0.0, top - TBAR["h"] / 2), (TBAR["len"], TBAR["w"], TBAR["h"]), "black", "grip", mu=0.9)
    b.cyl(f"{hl}/hub", (0.0, 0.0, top - TBAR["h"] - 0.006), 0.008, 0.012, "black", None)
    _joint(stage, "prismatic", f"{path}/j_handle", plate, hl, (0.0, 0.0, 0.0), (1, 0, 0, 0), (0.0, 0.0, 0.0), (1, 0, 0, 0),
           "Z", 0.0, HANDLE_TRAVEL)
    # leaves
    for k in range(N_LEAVES):
        h, radial, tang = leaf_hinge(k)
        rot = _leaf_frame(k)
        lk = f"{path}/leaf{k:02d}"
        lp = _xform(stage, lk)
        from pxr import Gf, UsdGeom

        xf = UsdGeom.Xformable(lp)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, h)))
        w, x, y, z = _quat_from_matrix(rot)
        xf.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
        m = MASS["leaf"]
        b.rigid(lp, m, (LEAF_LEN / 2, 0.0, 0.0), (m * LEAF_W_OUT ** 2 / 12, m * LEAF_LEN ** 2 / 3, m * LEAF_LEN ** 2 / 3))
        b.sheet(f"{lk}/sheet", leaf_local_points(), "steel", "steel")  # drawn and collided as the same twisted slab
        # hinge axis: the leaf's +y (tangential). A positive turn about +y tips +x (the leaf) down toward -z, so the
        # joint uses -y: folding up is positive. Joint frames: rot0 = the leaf frame in the plate frame, rot1 = identity.
        # The joint's local frame has its x axis along the hinge; build it so that joint x = -y_leaf.
        jr = rot @ np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))  # columns: -y_leaf, x_leaf, z_leaf
        w0 = _quat_from_matrix(jr)
        w1 = _quat_from_matrix(np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))))
        _joint(stage, "revolute", f"{path}/j_leaf{k:02d}", plate, lk, h, w0, (0.0, 0.0, 0.0), w1, "X", 0.0, FOLD_MAX_DEG)
    return root


def build_pan(stage, path, xy=(0.0, 0.0)):
    """The pan, one rigid body standing on the table at `xy` (parent frame), handle along +x."""
    b = _Builder(stage, path)
    root = _xform(stage, path, (xy[0], xy[1], 0.0))
    body = f"{path}/pan"
    bp = _xform(stage, body)
    R = PAN_BOT_D / 2
    b.rigid(bp, PAN_MASS, (0.012, 0.0, 0.035), (PAN_MASS * R * R / 2, PAN_MASS * R * R / 2, PAN_MASS * R * R))
    b.cyl(f"{body}/floor", (0.0, 0.0, PAN_FLOOR / 2), R, PAN_FLOOR, "pan", "pantable", mu=MU_TABLE)
    slant = PAN_H / math.cos(WALL_TILT)
    n = PAN_SEGMENTS
    for i in range(n):
        a = 2 * math.pi * i / n
        c = np.array((math.cos(a), math.sin(a), 0.0))
        t = np.array((-math.sin(a), math.cos(a), 0.0))
        up = math.cos(WALL_TILT) * np.array((0.0, 0.0, 1.0)) + math.sin(WALL_TILT) * c
        nrm = np.cross(t, up)  # outward
        rmid = (PAN_BOT_D + PAN_TOP_D) / 4 - PAN_WALL / 2
        centre = rmid * c + np.array((0.0, 0.0, PAN_H / 2)) + nrm * (PAN_WALL_COLLIDE / 2 - PAN_WALL / 2)
        width = 2 * math.pi * (rmid + 0.004) / n + 0.003
        rot = np.column_stack((t, np.cross(up, t), up))  # right-handed: x along the wall, y inward (thickness), z up the slant
        b.box(f"{body}/wall{i:02d}", centre, (width, PAN_WALL_COLLIDE, slant), "pan", "steel", rot)
    b.cyl(f"{body}/rim", (0.0, 0.0, PAN_H - 0.0012), RIM_R + 0.0012, 0.0024, "pan", None)
    # welded handle, rising 12 degrees, from the wall just under the rim
    hang = math.radians(12.0)
    hd = np.array((math.cos(hang), 0.0, math.sin(hang)))
    start = np.array((RIM_R - 0.002, 0.0, PAN_H - 0.022))
    rot = np.column_stack((hd, (0.0, 1.0, 0.0), np.cross(hd, (0.0, 1.0, 0.0))))
    b.box(f"{body}/handle", start + hd * PAN_HANDLE_LEN / 2, (PAN_HANDLE_LEN, 0.024, 0.012), "pan", "steel", rot)
    b.box(f"{body}/bracket", (RIM_R - 0.004, 0.0, PAN_H - 0.030), (0.010, 0.030, 0.030), "pan", None)
    return root
