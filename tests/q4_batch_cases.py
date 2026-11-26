"""Model builders and result hashing shared by the batched Q4 wiring tests (plate, Scordelis-Lo,
pinched cylinder, irregular plate, mixed Q4/S3 plate and shell, beam-stiffened plate)."""
import numpy as np
from anyfesolver import FEModel, LoadCase, solve_linear
from anyfesolver.elements import create_shell_element
from anyfesolver.boundary import BoundaryCondition
from anyfesolver.assembly import compute_stresses


def grid_model(name, P, t, E, nu, rho=None):
    m = FEModel(name)
    if rho is None:
        m.add_material("mat", E, nu)
    else:
        m.add_material("mat", E, nu, density=rho)
    ni, nj = P.shape[:2]
    N = np.zeros((ni, nj), int); k = 0
    for i in range(ni):
        for j in range(nj):
            k += 1; m.add_node(k, *map(float, P[i, j])); N[i, j] = k
    El = []; e = 0
    for i in range(ni - 1):
        for j in range(nj - 1):
            e += 1
            m.add_element(e, create_shell_element(e, [int(N[i, j]), int(N[i+1, j]), int(N[i+1, j+1]), int(N[i, j+1])], "mat", thickness=t))
            El.append(e)
    return m, N, El


def fixer(m, specs):
    bc = {}
    for nodes, dofs in specs:
        for n in np.atleast_1d(nodes):
            d = bc.setdefault(int(n), {})
            for q in dofs:
                d[q] = 0.0
    for n, d in bc.items():
        m.add_boundary_condition(BoundaryCondition(f"bc{n}", [n], d))


def scordelis(n):
    R, L, t, E = 25.0, 50.0, 0.25, 4.32e8
    th = np.radians(40.0)
    P = np.array([[(L/2*i/n, R*np.sin(th*j/n), R*np.cos(th*j/n)) for j in range(n+1)] for i in range(n+1)])
    m, N, El = grid_model("sl", P, t, E, 0.0, rho=90.0/t)
    fixer(m, [(N[0, :], ["uy", "uz"]), (N[-1, :], ["ux", "ry", "rz"]), (N[:, 0], ["uy", "rx", "rz"])])
    lc = LoadCase("g"); lc.set_gravity(0.0, 0.0, -1.0)
    return m, lc


def plate(n):
    E, nu, t, a = 1.7472e7, 0.3, 1e-4, 2.0
    xs = np.linspace(0, a/2, n+1)
    P = np.array([[(x, y, 0.0) for y in xs] for x in xs])
    m, N, El = grid_model("pl", P, t, E, nu)
    edge = list(N[-1, :]) + list(N[:, -1])
    fixer(m, [(N.ravel(), ["ux", "uy", "rz"]), (edge, ["uz"]), (N[0, :], ["ry"]), (N[:, 0], ["rx"])])
    lc = LoadCase("uniform")
    for e in El:
        lc.add_pressure_load(int(e), 1e-4)
    return m, lc


def pinched(n):
    R, L, t, E, nu = 300.0, 600.0, 3.0, 3.0e6, 0.3
    P = np.array([[(L/2*i/n, R*np.sin(np.pi/2*j/n), R*np.cos(np.pi/2*j/n)) for j in range(n+1)] for i in range(n+1)])
    m, N, El = grid_model("pc", P, t, E, nu)
    fixer(m, [(N[0, :], ["uy", "uz"]), (N[-1, :], ["ux", "ry", "rz"]), (N[:, 0], ["uy", "rx", "rz"]), (N[:, -1], ["uz", "rx", "ry"])])
    lc = LoadCase("p"); lc.add_nodal_load(int(N[-1, 0]), [0, 0, -0.25, 0, 0, 0])
    return m, lc


def rplate(n):
    """Flat plate with seeded in-plane jitter of interior nodes: every element geometry distinct."""
    E, nu, t, a = 2.1e11, 0.3, 0.01, 1.0
    xs = np.linspace(0, a, n+1)
    rng = np.random.default_rng(12345)
    P = np.array([[(x, y, 0.0) for y in xs] for x in xs])
    h = a / n
    P[1:-1, 1:-1, :2] += rng.uniform(-0.2*h, 0.2*h, size=(n-1, n-1, 2))
    m, N, El = grid_model("rp", P, t, E, nu)
    edge = list(N[0, :]) + list(N[-1, :]) + list(N[:, 0]) + list(N[:, -1])
    fixer(m, [(edge, ["ux", "uy", "uz", "rx", "ry", "rz"])])
    lc = LoadCase("p")
    for e in El:
        lc.add_pressure_load(int(e), 1e4)
    return m, lc


def _mixed_model(name, P, t, E, nu, rho=None):
    """Grid where every second quad is split into two S3 triangles (facet reference normals)."""
    m = FEModel(name)
    if rho is None:
        m.add_material("mat", E, nu)
    else:
        m.add_material("mat", E, nu, density=rho)
    ni, nj = P.shape[:2]
    N = np.zeros((ni, nj), int); k = 0
    for i in range(ni):
        for j in range(nj):
            k += 1; m.add_node(k, *map(float, P[i, j])); N[i, j] = k
    El = []; e = 0; q = 0
    for i in range(ni - 1):
        for j in range(nj - 1):
            ids = [int(N[i, j]), int(N[i+1, j]), int(N[i+1, j+1]), int(N[i, j+1])]
            q += 1
            if q % 2:
                e += 1
                m.add_element(e, create_shell_element(e, ids, "mat", thickness=t)); El.append(e)
            else:
                for tri in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])):
                    p = [np.asarray(P.reshape(-1, 3)[v - 1]) for v in tri]
                    nrm = np.cross(p[1] - p[0], p[2] - p[0]); nrm /= np.linalg.norm(nrm)
                    e += 1
                    m.add_element(e, create_shell_element(e, list(tri), "mat", thickness=t,
                                                          reference_normal=tuple(float(v) for v in nrm)))
                    El.append(e)
    return m, N, El


def mixed_plate(n):
    E, nu, t, a = 2.1e11, 0.3, 0.01, 1.0
    xs = np.linspace(0, a, n+1)
    P = np.array([[(x, y, 0.0) for y in xs] for x in xs])
    m, N, El = _mixed_model("mp", P, t, E, nu)
    edge = list(N[0, :]) + list(N[-1, :]) + list(N[:, 0]) + list(N[:, -1])
    fixer(m, [(edge, ["ux", "uy", "uz", "rx", "ry", "rz"])])
    lc = LoadCase("p")
    for e in El:
        lc.add_pressure_load(int(e), 1e4)
    return m, lc


def mixed_sl(n):
    R, L, t, E = 25.0, 50.0, 0.25, 4.32e8
    th = np.radians(40.0)
    P = np.array([[(L/2*i/n, R*np.sin(th*j/n), R*np.cos(th*j/n)) for j in range(n+1)] for i in range(n+1)])
    m, N, El = _mixed_model("msl", P, t, E, 0.0, rho=90.0/t)
    fixer(m, [(N[0, :], ["uy", "uz"]), (N[-1, :], ["ux", "ry", "rz"]), (N[:, 0], ["uy", "rx", "rz"])])
    lc = LoadCase("g"); lc.set_gravity(0.0, 0.0, -1.0)
    return m, lc


def stiffened(n):
    """Clamped Q4 plate with concentric beam stiffeners along every 4th grid line in x."""
    from anyfesolver.elements import BeamElement
    E, nu, t, a = 2.1e11, 0.3, 0.01, 1.0
    xs = np.linspace(0, a, n+1)
    P = np.array([[(x, y, 0.0) for y in xs] for x in xs])
    m, N, El = grid_model("st", P, t, E, nu)
    sec = {"area": 1.0e-3, "Iy": 2.0e-7, "Iz": 1.0e-8, "J": 1.0e-9}
    e = len(El)
    for j in range(4, n, 4):
        for i in range(n):
            e += 1
            m.add_element(e, BeamElement(e, [int(N[i, j]), int(N[i+1, j])], "mat", dict(sec)))
    edge = list(N[0, :]) + list(N[-1, :]) + list(N[:, 0]) + list(N[:, -1])
    fixer(m, [(edge, ["ux", "uy", "uz", "rx", "ry", "rz"])])
    lc = LoadCase("p")
    for el in El:
        lc.add_pressure_load(int(el), 1e4)
    return m, lc


CASES = {"sl": scordelis, "plate": plate, "pinch": pinched, "rplate": rplate,
         "mplate": mixed_plate, "msl": mixed_sl, "stiff": stiffened}


def feed(h, obj):
    if isinstance(obj, np.ndarray):
        h.update(str(obj.dtype).encode()); h.update(str(obj.shape).encode()); h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode()); feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            feed(h, v)
        h.update(b"]")
    elif isinstance(obj, float):
        h.update(np.float64(obj).tobytes())
    else:
        h.update(repr(obj).encode())

