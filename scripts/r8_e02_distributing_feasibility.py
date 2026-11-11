"""Small common-action area comparison, not a qualification adjudicator.

The resolved-solid secant-director observable is a distributing idealization;
it is not a welded face. The fixed physical pad uses paired 10x4, 20x8 and
40x16 Cartesian geometries; through-thickness resolution stays fixed.
Build/parse never launch a solver. Native execution requires the fixed installed
artifact and retains the free GE stub, checking its actual fields and work.
"""
from dataclasses import asdict
import hashlib
import json
import math
import re

import numpy as np

from scripts.r8_e02_distributing_solid_area import compile_area_map
from scripts.r8_e02_rigid_tip_reference import (
    CCX_SHA256, CCX_VERSION, ReferenceParseError, _HEADER, _nset,
    _printed_number, build_deck,
)

BOUNDS = (.8, 1., -.05, 0.)
PORT = (.9, -.025, .05)
LOADS = {"Mx": (0., 0., 0., 1., 0., 0.),
         "Fy": (0., 10., 0., 0., 0., 0.)}


def _hash(value):
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def build_case(name, nx=10, ny=4):
    """No rigid-body equations from the geometry donor enter this new deck."""
    if name not in LOADS:
        raise ValueError("fixed Mx or eccentric Fy feasibility case required")
    _, donor = build_deck(nx, ny, 1, "Mx")
    nodes, elements = donor["physical_nodes"], donor["elements"]
    return _build_geometry_case(name, nodes, elements, donor["root_nodes"], nx, ny, 2,
        "R8_E02_DISTRIBUTING_FEASIBILITY_V1",
        "single coarse common-action screen; no refinement or acceptance"
        if (nx, ny) == (10, 4) else "paired in-plane refinement; thickness fixed; no acceptance")


def _build_geometry_case(name, nodes, elements, root, nx, ny, nz, schema, scope):
    """Common deck syntax; each caller owns its separately bounded geometry."""
    if name not in LOADS:
        raise ValueError("fixed Mx or eccentric Fy feasibility case required")
    area = compile_area_map(nodes, elements, BOUNDS, .02, PORT)
    action = area.action(LOADS[name][:3], LOADS[name][3:])
    deck = ["*HEADING", "R8 distributing common-action feasibility only", "*NODE"]
    deck.extend(str(n) + "," + ",".join(format(v, ".17g") for v in p)
                for n, p in nodes.items())
    deck.append("*ELEMENT,TYPE=C3D20R,ELSET=SOLID")
    for e, ids in elements.items():
        deck.extend((str(e)+","+",".join(map(str, ids[:15])),
                     ",".join(map(str, ids[15:]))))
    deck.extend(_nset("ALL", sorted(nodes)))
    deck.extend(_nset("ROOT", root))
    deck.extend(("*MATERIAL,NAME=ELASTIC", "*ELASTIC", "210000000000.,0.3",
        "*SOLID SECTION,ELSET=SOLID,MATERIAL=ELASTIC", "*BOUNDARY", "ROOT,1,3,0.",
        "*STEP", "*STATIC", "1.,1.,0.000001,1.", "*CLOAD"))
    deck.extend(f"{n},{j},{v:.17g}" for n, f in zip(action.node_ids, action.forces_n)
                for j, v in enumerate(f, 1) if v)
    deck.extend(("*NODE PRINT,NSET=ALL", "U", "*NODE PRINT,NSET=ROOT", "RF", "*END STEP"))
    text = "\n".join(deck)+"\n"
    metadata = dict(schema=schema, scope=scope,
        case=name, ccx_version=CCX_VERSION, executable_sha256=CCX_SHA256,
        nx=nx, ny=ny, nz=nz, E_Pa=210e9, nu=.3, thickness_m=.02,
        bounds_xy_m=BOUNDS, port_m=PORT, applied_wrench=LOADS[name],
        physical_nodes=nodes, elements=elements, root_nodes=root,
        area_map_identity=area.identity,
        deck_sha256=hashlib.sha256(text.encode("ascii")).hexdigest())
    return text, metadata, area


def parse_result(dat, stdout, metadata):
    """Require completed, correctly identified, finite and fully observed output.

    Decimal print bounds are output-resolution bounds, not numerical-error or
    physical-accuracy estimates. No balance/accuracy acceptance is inferred.
    """
    try:
        _, expected, area = build_case(metadata["case"], metadata["nx"], metadata["ny"])
        if _hash(expected) != _hash(metadata):
            raise ReferenceParseError("changed reference metadata")
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceParseError("invalid reference metadata") from exc
    return _parse_observed(dat, stdout, expected, area)


def _parse_observed(dat, stdout, expected, area):
    """Shared output checks after the caller reconstructs its exact case."""
    if re.search(r"\b(?:warnings?|errors?|fatal|failed|failure|nan|infinity)\b", dat+"\n"+stdout, re.I):
        raise ReferenceParseError("solver diagnostic or nonfinite output")
    if len(re.findall(r"^\s*Job finished\s*$", stdout, re.M)) != 1:
        raise ReferenceParseError("incomplete or multiple job")
    if re.findall(r"CalculiX Version\s+(\S+)", stdout) not in ([CCX_VERSION], [CCX_VERSION+","]):
        raise ReferenceParseError("wrong executable version")
    cpu = [line for line in stdout.splitlines() if "cpu" in line.lower()]
    if not cpu or any(re.findall(r"\b(\d+)\s+cpu", line, re.I) != ["1"] for line in cpu):
        raise ReferenceParseError("effective single-CPU reporting required")
    headers = list(_HEADER.finditer(dat))
    required = {("displacements", "ALL"): set(expected["physical_nodes"]),
                ("forces", "ROOT"): set(expected["root_nodes"])}
    parsed, uncertainty = {}, {}
    for i, h in enumerate(headers):
        key = h[1].lower(), h[2].upper()
        if key not in required or key in parsed or _printed_number(h[3])[0] != 1.:
            raise ReferenceParseError("unexpected output section/time")
        end = headers[i+1].start() if i+1 < len(headers) else len(dat)
        rows, errors = {}, {}
        for line in dat[h.end():end].splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4 or not re.fullmatch(r"[1-9]\d*", parts[0]):
                raise ReferenceParseError("malformed output row")
            n = int(parts[0])
            if n in rows:
                raise ReferenceParseError("duplicate output node")
            values = [_printed_number(s) for s in parts[1:]]
            rows[n], errors[n] = tuple(v[0] for v in values), tuple(v[1] for v in values)
        if set(rows) != required[key]:
            raise ReferenceParseError("missing or unexpected output nodes")
        parsed[key], uncertainty[key] = rows, errors
    if set(parsed) != set(required):
        raise ReferenceParseError("missing observed output")
    u = parsed["displacements", "ALL"]
    ub = uncertainty["displacements", "ALL"]
    rf = parsed["forces", "ROOT"]
    rfb = uncertainty["forces", "ROOT"]
    action = area.action(expected["applied_wrench"][:3], expected["applied_wrench"][3:])
    observed = area.recover(u)
    forces = np.array(list(rf.values()))
    positions = np.array([expected["physical_nodes"][n] for n in rf])
    balance = np.r_[forces.sum(axis=0), np.cross(positions, forces).sum(axis=0)]
    wrench = np.array(expected["applied_wrench"])
    applied = np.r_[wrench[:3], wrench[3:]+np.cross(PORT, wrench[:3])]
    root_bounds = np.array(list(rfb.values()))
    moment_bounds = np.array([
        (abs(p[1])*b[2]+abs(p[2])*b[1], abs(p[2])*b[0]+abs(p[0])*b[2],
         abs(p[0])*b[1]+abs(p[1])*b[0]) for p, b in zip(positions, root_bounds)])
    work_bound = math.fsum(abs(f[j])*ub[n][j]
        for n, f in zip(action.node_ids, action.forces_n) for j in range(3))
    return dict(status="PARSED_DEVELOPMENT_ONLY", observable=asdict(observed),
        external_final_pairing_J=action.work(u), pairing_print_bound_J=work_bound,
        root_wrench_about_origin=balance.tolist(), root_wrench_error=(balance+applied).tolist(),
        root_wrench_print_bounds=np.r_[root_bounds.sum(axis=0), moment_bounds.sum(axis=0)].tolist(),
        root_displacement_max=max(abs(v) for n in expected["root_nodes"] for v in u[n]),
        precision_scope="printed-decimal resolution, no 1e-8 certification")


def native_cases(nx=10, ny=4):
    """Run one fixed physical pad through the installed public GE facade."""
    return _native_load_cases(nx, ny, LOADS)


def _native_load_cases(nx, ny, loads, *, geometry_builder=None):
    """Local fixture seam; each bounded caller owns its explicit load inventory."""
    from anyfesolver import b3_ge, stiffened_shell as ss
    from scipy.spatial.transform import Rotation
    from scripts.r8_e02_native_rigid_tip import geometry
    if geometry_builder is not None:
        geometry = geometry_builder
    nodes, cells, _, _, root = geometry(nx, ny)
    # Integer grid ownership holds the material rectangle fixed at all levels.
    hosts = tuple(1+j*nx+i for j in range(ny//4,ny//2) for i in range(4*nx//5,nx))
    if len(hosts) != nx*ny//20:
        raise ValueError("fixed physical-pad host inventory mismatch")
    e, g = 210e9, 210e9/2.6
    section = b3_ge.EllipsoidalGeneralizedSection(np.diag((e*.0004, g*.0004*5/6,
        g*.0004*5/6, g*2.249e-8, e*1.3333333333333335e-8,
        e*1.3333333333333335e-8)), np.eye(6), 1e20, 1e9)
    xyz = np.array((PORT, (.95, PORT[1], PORT[2]), (1., PORT[1], PORT[2])))
    member = b3_ge.define_beam("b3-ge", 1, (100001,100002,100003), xyz,
        np.tile(np.eye(3), (3,1,1)), section, np.eye(6))
    definition = ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        ss.ShellPanel(tuple(nodes), tuple(nodes.values()), cells, .02, e, .3, 7850.),
        (member,), (), tuple(ss.FixedCoordinate("shell", n, d) for n in root for d in range(6)),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        distributing_footprints=(ss.GEReferenceDistributingFootprint("area", hosts, 100001, BOUNDS),))
    owner = ss.create_analysis(definition=definition)
    records = {}
    try:
        ns = 6*len(nodes)
        for name, wrench in loads.items():
            result = owner.solve_linear((ss.NodalAction("beam",100001,wrench),))
            if result.status != "completed":
                raise RuntimeError("native solve failed")
            q = np.r_[result.shell_displacement,result.beam_coordinates]
            beam = result.beam_coordinates
            port = beam[:6]
            transfer, = owner.recover(selection=ss.GEReferenceDistributingFootprintSelection(("area",)))
            # Actual retained strain/resultant recovery and rigid-field checks,
            # beyond an unloaded-stub assumption or small quadratic work alone.
            element, _, origin = owner._beam_parts[0]
            stations = element.operator.recover(Rotation.from_rotvec(beam[18:24].reshape(2,3).copy()).as_matrix(),
                beam[24:], origin=origin)
            force = owner._K[ns:,ns:]@beam
            pairing = float(np.array(wrench)@port)
            root_force, root_moment = np.zeros(3), np.zeros(3)
            for node in root:
                f = result.reactions[6*(node-1):6*(node-1)+3]
                m = result.reactions[6*(node-1)+3:6*node]
                root_force += f
                root_moment += m+np.cross(nodes[node],f)
            root_error = np.r_[root_force+np.array(wrench[:3]),
                root_moment+np.array(wrench[3:])+np.cross(PORT,wrench[:3])]
            expected_centroid = np.r_[wrench[:3],np.array(wrench[3:])+
                np.cross(np.array(PORT)-transfer.centroid_m,wrench[:3])]
            transfer_error = transfer.shell_action-expected_centroid
            rigid = np.array([np.r_[port[:3]+np.cross(port[3:],x-xyz[0]),port[3:]] for x in xyz]).ravel()
            checks = dict(stub_coordinate_rigid_error=float(np.max(np.abs(beam[:18]-rigid))),
                stub_cell_rotation_error=float(np.max(np.abs(beam[18:24]-np.tile(port[3:],2)))),
                stub_retained_resultant_max=float(np.max(np.abs(beam[24:]))),
                stub_station_strain_max=max(float(np.max(np.abs(s["strain"]+s["strain_low"]))) for s in stations),
                stub_station_resultant_max=max(float(np.max(np.abs(s["resultants"]+s["resultants_low"]))) for s in stations),
                stub_covector_max=float(np.max(np.abs(force))),
                stub_quadratic_work_J=.5*float(beam@force),
                root_wrench_error=root_error.tolist(),
                transfer_wrench_error=transfer_error.tolist(),
                full_quadratic_pairing_J=float(q@(owner._K@q)),
                equilibrium_ratio=result.free_equilibrium_ratio, constraint_max=result.constraint_max)
            if (checks["stub_coordinate_rigid_error"] > 1e-10 or
                checks["stub_cell_rotation_error"] > 1e-10 or
                checks["stub_station_strain_max"] > 1e-10 or
                checks["stub_station_resultant_max"] > 1e-7 or
                checks["stub_covector_max"] > 1e-7 or
                np.max(np.abs(root_error)) > 1e-7 or
                np.max(np.abs(transfer_error)) > 1e-8 or
                abs(checks["stub_quadratic_work_J"]) > 1e-10*abs(pairing) or
                abs(checks["full_quadratic_pairing_J"]-pairing) > 1e-8*abs(pairing)):
                raise RuntimeError("free-stub/work controls failed: "+repr(checks))
            records[name] = dict(status=result.status, port_translation_m=port[:3].tolist(),
                port_rotation_rad=port[3:].tolist(), external_final_pairing_J=pairing,
                root_force_N=root_force.tolist(),root_moment_Nm=root_moment.tolist(),
                transfer=asdict(transfer), controls=checks)
        return dict(scope="reference-linear common-action, acceptance separate", nx=nx,ny=ny,hosts=hosts,
            shell_nodes=nodes, cells=cells, root=root, port_m=PORT, bounds_xy_m=BOUNDS,
            factors=owner.preflight()["reference_linear_factorizations"], results=records)
    finally:
        owner.close()
