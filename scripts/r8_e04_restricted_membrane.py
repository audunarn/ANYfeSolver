"""Supplemental nonlinear material/axial-patch feasibility, never E04 acceptance.

Execution requires a separately authorised single-job resource supervisor.
No roof, capacity, rotating connector, or mandatory CalculiX-row claim is made.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
import json
import math
import os
from pathlib import Path
import sys

E = 210e9
NU = .3
G = E / (2 * (1 + NU))
BULK = E / (3 * (1 - 2 * NU))
SIG0 = 1e6
H = 1e8
LENGTH = 1.
WIDTH = .1
THICKNESS = .002
OFFSET = .01
B = .01
HEIGHT = .004
AREA = B * HEIGHT
FORCE = 400.
TARGETS = (.05, .15, .35, .6, 1., .6, .2, 1.)
SHELL_IDS = (1, 2, 3, 4, 5, 6)
SHELL_XYZ = ((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.),
             (0., .1, 0.), (.5, .1, 0.), (1., .1, 0.))
QUADS = ((1, 2, 5, 4), (2, 3, 6, 5))
BEAM_IDS = ((101, 102, 103), (104, 105, 106))
PAIRS = tuple(zip(SHELL_IDS, (101, 102, 103, 104, 105, 106)))
SOURCES = {
    'j2': 'https://raw.githubusercontent.com/OpenSees/OpenSees/master/SRC/material/nD/J2Plasticity.cpp',
    'plate_fiber': 'https://raw.githubusercontent.com/OpenSees/OpenSees/master/SRC/material/nD/J2PlateFiber.cpp',
    'shell': 'https://raw.githubusercontent.com/OpenSees/OpenSees/master/SRC/element/shell/ShellMITC4.cpp',
    'section': 'https://raw.githubusercontent.com/OpenSees/OpenSees/master/SRC/material/section/MembranePlateFiberSection.cpp',
}


class UnsupportedQuantity(ValueError):
    """Required actual external response is absent or has unknown semantics."""


def rectangle():
    """Same centroidal rectangle in both domains; offset enters the bond once."""
    iy = B * HEIGHT**3 / 12
    iz = HEIGHT * B**3 / 12
    # Saint-Venant odd-series rectangle, long side B, short side HEIGHT.
    series = math.fsum(math.tanh(k * math.pi * B / (2 * HEIGHT)) / k**5
                       for k in range(1, 512, 2))
    torsion = B * HEIGHT**3 / 3 * (1 - 192 * HEIGHT / (math.pi**5 * B) * series)
    return iy, iz, torsion


def fixture_signature():
    """Frozen physical/history identity independent of comparator implementation."""
    fixture = dict(contract='RESTRICTED_MEMBRANE_SUPPLEMENTAL_V1',
        shell_ids=SHELL_IDS, shell_xyz=SHELL_XYZ, quads=QUADS, beam_ids=BEAM_IDS,
        pairs=PAIRS, length=LENGTH, width=WIDTH, thickness=THICKNESS,
        offset=OFFSET, rectangle=(B, HEIGHT), rectangle_properties=rectangle(),
        young=E, poisson=NU, force=FORCE, targets=TARGETS,
        curve=(SIG0, 2e6, 2e6, .01, .02, 2e6, .2), external_hardening=H,
        fixed_shell_components=(1, 2, 3, 4, 5), axial_fixed_shell_nodes=(1, 4),
        loaded_shell_nodes=(3, 6), shell_material_reference='CURVED_Q4_POINT_MATERIAL_REFERENCE_V1_DEV')
    return sha256(json.dumps(fixture, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def native_definition():
    import numpy as np
    from anymaterial import DNVC208MaterialCurve
    from anyfesolver import b3_ge, stiffened_shell as ss
    iy, iz, torsion = rectangle()
    section = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((E * AREA, 5 / 6 * G * AREA, 5 / 6 * G * AREA,
                 G * torsion, E * iy, E * iz)), np.eye(6), 1e20, 1e9)
    inertia = np.diag(7850 * np.array((AREA, AREA, AREA, iy + iz, iy, iz)))
    beams = tuple(b3_ge.define_beam('b3-ge', i + 1, ids,
        np.array([(x, i * WIDTH, OFFSET) for x in (0., .5, 1.)]),
        np.tile(np.eye(3), (3, 1, 1)), section, inertia)
        for i, ids in enumerate(BEAM_IDS))
    curve = DNVC208MaterialCurve(sigma_prop=SIG0, sigma_yield=2e6,
        sigma_yield_2=2e6, eps_p_y1=.01, eps_p_y2=.02, K=2e6, n=.2)
    shell = ss.ShellPanel(SHELL_IDS, SHELL_XYZ, QUADS, THICKNESS, E, NU,
                          7850., j2_curve=curve)
    fixed = tuple(ss.FixedCoordinate('shell', n, c)
                  for n in SHELL_IDS for c in (1, 2, 3, 4, 5)) + tuple(
                      ss.FixedCoordinate('shell', n, 0) for n in (1, 4))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1, shell,
        beams, tuple(ss.PointConnection(s, b) for s, b in PAIRS), fixed,
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        shell_reference_policy=ss.CURVED_Q4_POINT_MATERIAL_REFERENCE_V1,
        retained_scale_policy='native_force_moment_roles_v1')


def actual_response(ops, element, args, count):
    values = tuple(float(v) for v in ops.eleResponse(element, *args))
    if len(values) != count or not all(map(math.isfinite, values)):
        raise UnsupportedQuantity(f'element {element}, {args}: need {count} finite actual values')
    return values


def external_layers(ops):
    """Source-supported 4 shell points x 5 section fibers; no elastic proxy."""
    return [dict(element=e, point=p, fiber=f,
        strain=actual_response(ops, e, ('material', p, 'fiber', f, 'strain'), 5),
        stress=actual_response(ops, e, ('material', p, 'fiber', f, 'stress'), 5))
        for e in (1, 2) for p in range(1, 5) for f in range(1, 6)]


def plastic_from_actual(strain, stress):
    """Inferred plastic strain, explicitly distinct from engine history output."""
    ex, ey, gamma = strain[:3]
    sx, sy, tau = stress[:3]
    px = ex - (sx - NU * sy) / E
    py = ey - (sy - NU * sx) / E
    pg = gamma - tau / G
    return (px, py, -px - py, pg)


def equivalent_increment(left, right):
    dx, dy, dz, dg = (a - b for a, b in zip(right, left))
    return math.sqrt(2 / 3 * (dx * dx + dy * dy + dz * dz + .5 * dg * dg))


def encode(value):
    if is_dataclass(value):
        return {k: encode(v) for k, v in asdict(value).items()}
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def check_patch(displacements):
    """Freeze R=I pure extension admission before any response comparison."""
    eps = (displacements[3][0] + displacements[6][0]) / (2 * LENGTH)
    error = max(abs(displacements[n][0] - eps * SHELL_XYZ[n - 1][0])
                for n in SHELL_IDS)
    error = max(error, *(abs(v) for n in SHELL_IDS for v in displacements[n][1:]))
    if error > 1e-10:
        raise ValueError(f'restricted affine patch lost: {error:g}')
    return eps, error


def patch_record(factor, displacements, layers, beam_work):
    eps, error = check_patch(displacements)
    if not layers:
        raise UnsupportedQuantity('actual shell material layers required')
    stresses = [row['stress'][:3] for row in layers]
    strains = [row['strain'][:3] for row in layers]
    sx = math.fsum(s[0] for s in stresses) / len(stresses)
    spread = max(abs(s[j] - stresses[0][j]) for s in stresses for j in range(3))
    strain_error = max(abs(s[j] - (eps if j == 0 else 0.))
                       for s in strains for j in range(3))
    if strain_error > 1e-10 or spread > .01:
        raise ValueError('nonuniform material patch or engineering-strain mismatch')
    # Endpoint virtual work, NOT cumulative energy or dissipative work integral.
    shell_work = WIDTH * THICKNESS * LENGTH * sx * eps
    external_work = factor * FORCE * eps * LENGTH
    equilibrium = WIDTH * THICKNESS * sx + 2 * E * AREA * eps - factor * FORCE
    work_error = shell_work + beam_work - external_work
    if abs(equilibrium) > 1e-4 or abs(work_error) > 1e-8 * max(1., abs(external_work)):
        raise ValueError(f'axial balance/work failed: {equilibrium:g}, {work_error:g}')
    return dict(factor=factor, engineering_axial_strain=eps,
        patch_error_m=error, layer_stress_pa=stresses[0], actual_layers=layers,
        shell_axial_virtual_work_j=shell_work, beam_axial_virtual_work_j=beam_work,
        external_virtual_work_j=external_work, virtual_work_residual_j=work_error,
        axial_equilibrium_residual_n=equilibrium, shell_stress_spread_pa=spread)


def elastic_control(row):
    """First load is below yield: independently check both axial stiffnesses."""
    modulus = E / (1 - NU**2)
    expected = TARGETS[0] * FORCE / (WIDTH * THICKNESS * modulus + 2 * E * AREA)
    if abs(row['engineering_axial_strain'] - expected) > 1e-12 or max(
            abs(a - b) for a, b in zip(row['layer_stress_pa'],
            (modulus * expected, NU * modulus * expected, 0.))) > .01:
        raise ValueError('independent elastic axial/plane-stress patch control failed')
    row['elastic_control_strain'] = expected


def solve_native(output):
    import numpy as np
    from anyfesolver import stiffened_shell as ss
    definition = native_definition()
    owner = ss.create_analysis(definition=definition)
    records = []
    actions = tuple(ss.NodalAction('shell', n, (FORCE / 2, 0., 0., 0., 0., 0.)) for n in (3, 6))
    try:
        save(output / 'native-preflight.json', dict(preflight=owner.preflight(),
            geometry='flat', shell_reference_policy=definition.shell_reference_policy,
            retained_scale_policy=definition.retained_scale_policy))
        for factor in TARGETS:
            result = owner.solve(actions, targets=(factor,), max_iterations=24, max_backtracks=12)
            (output / 'native-last-accepted.checkpoint').write_bytes(owner.checkpoint())
            save(output / 'native-last-accepted-material.json', encode(result.shell_q4_material))
            if np.asarray(result.shell_displacement).shape != (36,):
                raise UnsupportedQuantity('native actual shell displacement must contain six complete six-coordinate nodes')
            displacements = dict(zip(SHELL_IDS, np.asarray(result.shell_displacement).reshape(6, 6)))
            if tuple(r.element_id for r in result.shell_q4_material) != (1, 2):
                raise UnsupportedQuantity('native actual shell element inventory must be exactly (1, 2)')
            for r in result.shell_q4_material:
                if (np.asarray(r.layer_strain).shape != (12, 3)
                        or np.asarray(r.layer_stress).shape != (12, 3)
                        or np.asarray(r.alpha).shape != (12,)
                        or not all(np.isfinite(np.asarray(v)).all()
                            for v in (r.layer_strain, r.layer_stress, r.alpha))):
                    raise UnsupportedQuantity(f'native element {r.element_id}: require complete finite 4x3 layer material inventory')
            layers = [dict(element=r.element_id, point=i,
                          strain=s.tolist(), stress=t.tolist(), alpha=float(a))
                for r in result.shell_q4_material
                for i, (s, t, a) in enumerate(zip(r.layer_strain, r.layer_stress, r.alpha, strict=True))]
            if not layers or not all(0 <= r['alpha'] < .01 for r in layers):
                raise ValueError('actual native alpha outside matched first branch')
            if result.free_equilibrium_ratio > 1e-8 or result.constraint_max > 1e-9:
                raise ValueError('native equilibrium/constraint gate failed')
            alpha = np.array([r['alpha'] for r in layers])
            if not np.array_equal(definition.shell.j2_curve.flow_stress(alpha), SIG0 + H * alpha):
                raise ValueError('effective native material first branch differs from external J2 law')
            if any(a != 0. for row in result.beam_station_accumulated for a in row):
                raise ValueError('beam must remain elastic')
            stations = [s for row in result.beam_generalized_material for s in row.stations]
            if not stations:
                raise UnsupportedQuantity('actual retained beam material stations required')
            beam_work = math.fsum(s.measure * math.fsum((
                float(s.local_strain[0]), float(s.local_strain_low[0]))) * math.fsum((
                float(s.local_resultants[0]), float(s.local_resultants_low[0]))) for s in stations)
            eps, _ = check_patch(displacements)
            if not np.allclose(result.beam_frames, np.eye(3), rtol=0, atol=1e-10) or not np.allclose(
                    result.cell_frames, np.eye(3), rtol=0, atol=1e-10):
                raise ValueError('retained physical/cell frames left identity subspace')
            for s in stations:
                strain = s.local_strain + s.local_strain_low
                force = s.local_resultants + s.local_resultants_low
                if abs(strain[0] - eps) > 1e-10 or max(abs(strain[1:])) > 1e-10 or abs(
                        force[0] - E * AREA * eps) > 1e-4 or max(abs(force[1:])) > 1e-4:
                    raise ValueError('retained beam axial station contract failed')
            for shell, beam in PAIRS:
                bi = beam - 101
                expected = np.asarray(SHELL_XYZ[shell - 1]) + displacements[shell][:3] + (0., 0., OFFSET)
                if np.max(np.abs(result.beam_positions[bi] - expected)) > 1e-10:
                    raise ValueError('exact restricted relative-pose bond failed')
            row = patch_record(factor, displacements, layers, beam_work)
            row.update(beam_material=encode(result.beam_generalized_material),
                       shell_material=encode(result.shell_q4_material),
                       native_free_equilibrium_ratio=result.free_equilibrium_ratio,
                       native_constraint_max=result.constraint_max)
            if not records:
                elastic_control(row)
            records.append(row)
            save(output / 'native.json', records)
    finally:
        owner.close()
    if max(r['alpha'] for row in records for r in row['actual_layers']) <= 0:
        raise ValueError('shell plasticity never activated')
    return records


def solve_external(ops, output):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    for n, xyz in zip(SHELL_IDS, SHELL_XYZ):
        ops.node(n, *xyz)
        ops.fix(n, int(n in (1, 4)), 1, 1, 1, 1, 1)
    for s, b in PAIRS:
        x, y, _ = SHELL_XYZ[s - 1]
        ops.node(b, x, y, OFFSET)
        ops.rigidLink('beam', s, b)
    ops.nDMaterial('J2Plasticity', 1, BULK, G, SIG0, SIG0, 0., H, 0.)
    ops.section('PlateFiber', 1, 1, THICKNESS)
    for e, q in enumerate(QUADS, 1):
        ops.element('ShellMITC4', e, *q, 1)
    ops.geomTransf('Linear', 1, 0., 0., 1.)
    iy, iz, torsion = rectangle()
    beams = [(3 + 2 * i + j, (ids[j], ids[j + 1]))
             for i, ids in enumerate(BEAM_IDS) for j in (0, 1)]
    for e, ids in beams:
        ops.element('ElasticTimoshenkoBeam', e, *ids, E, G, AREA,
                    torsion, iy, iz, 5 / 6 * AREA, 5 / 6 * AREA, 1)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for n in (3, 6):
        ops.load(n, FORCE / 2, 0., 0., 0., 0., 0.)
    # Scaling exact multiplier rows does not add stiffness or penalty energy.
    ops.constraints('Lagrange', 1e9, 1e9)
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormUnbalance', 1e-8, 30)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', TARGETS[0])
    ops.analysis('Static')
    records = []
    previous_factor = 0.
    inferred = [(0., 0., 0., 0.)] * 40
    inferred_alpha = [0.] * 40
    try:
        for factor in TARGETS:
            ops.integrator('LoadControl', factor - previous_factor)
            code = ops.analyze(1)
            if code:
                raise RuntimeError(f'OpenSees analyze returned {code}, target={factor}')
            displacements = {n: tuple(ops.nodeDisp(n)) for n in SHELL_IDS}
            for s, b in PAIRS:
                if max(abs(x - y) for x, y in zip(ops.nodeDisp(b), displacements[s])) > 1e-10:
                    raise ValueError('affine constraint departed exact R=I bond')
            layers = external_layers(ops)
            for i, row in enumerate(layers):
                plastic = plastic_from_actual(row['strain'], row['stress'])
                inferred_alpha[i] += equivalent_increment(inferred[i], plastic)
                inferred[i] = plastic
                row['inferred_plastic_strain'] = plastic
                row['inferred_alpha_from_actual_stress_strain'] = inferred_alpha[i]
            if not all(0 <= a < .01 for a in inferred_alpha):
                raise ValueError('inferred external alpha outside matched first branch')
            forces = {e: actual_response(ops, e, ('globalForce',), 12) for e, _ in beams}
            beam_work = math.fsum(forces[e][6 * j] * float(ops.nodeDisp(n, 1))
                                 for e, ids in beams for j, n in enumerate(ids))
            row = patch_record(factor, displacements, layers, beam_work)
            row['beam_actual_global_forces'] = forces
            if not records:
                elastic_control(row)
            records.append(row)
            save(output / 'external.json', records)
            previous_factor = factor
    finally:
        ops.wipe()
    if max(inferred_alpha) <= 1e-12:
        raise ValueError('external actual stress/strain never evidenced plastic activity')
    return records


def save(path, value):
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False), encoding='utf-8')


def compare(native, external):
    """Measured differences only: this supplemental screen sets no acceptance gate."""
    if len(native) != len(TARGETS) or len(external) != len(TARGETS):
        raise ValueError('complete matched loading/unloading history required')
    if tuple(a['factor'] for a in native) != TARGETS or tuple(b['factor'] for b in external) != TARGETS:
        raise ValueError('loading/unloading targets differ')
    for row in (*native, *external):
        if len(row['layer_stress_pa']) != 3 or not all(map(math.isfinite,
                (row['engineering_axial_strain'], *row['layer_stress_pa']))):
            raise UnsupportedQuantity('comparison requires finite strain and complete three-component actual stress')
    return [dict(factor=a['factor'], strain_difference=a['engineering_axial_strain'] - b['engineering_axial_strain'],
                 stress_difference_pa=[x - y for x, y in zip(a['layer_stress_pa'], b['layer_stress_pa'], strict=True)])
            for a, b in zip(native, external, strict=True)]


def runtime_identity(record, route, opensees_site=None):
    """Import only this route's dependencies; measure the actual loaded pools."""
    import_paths = {}
    record.update(import_paths=import_paths, numerical_environment={
        k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMBA_DISABLE_JIT')})
    if route == 'external':
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('installed OpenSeesPy binary requires Python 3.12')
        sys.path.append(str(opensees_site.resolve()))
    import numpy
    import threadpoolctl
    import_paths['numpy'] = numpy.__file__
    if route == 'native':
        import scipy
        import scipy.linalg
        import anyfesolver
        import anymaterial
        import numba
        import_paths.update(scipy=scipy.__file__, anyfesolver=anyfesolver.__file__,
                            anymaterial=anymaterial.__file__, numba=numba.__file__)
        record.update(numba_disable_jit=bool(numba.config.DISABLE_JIT),
                      numba_threads=numba.get_num_threads())
        if not numba.config.DISABLE_JIT or numba.get_num_threads() != 1:
            raise RuntimeError('effective native JIT-off and numba=1 required')
        ops = None
    else:
        # External mechanics uses only its independent engine and compatible NumPy.
        # Environment is checked; no unimported Numba runtime is claimed measured.
        record['numba_runtime'] = 'not imported in external route'
        if os.environ.get('NUMBA_DISABLE_JIT') != '1' or any(
                os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
            raise RuntimeError('external supervisor JIT-off and numerical environment=1 required')
        import openseespy.opensees as ops
        import_paths['opensees'] = ops.__file__
        binaries = [module for name, module in tuple(sys.modules.items())
            if name.startswith('opensees') and name.endswith('.opensees')
            and getattr(module, '__file__', None)
            and any(module.__file__.endswith(suffix) for suffix in EXTENSION_SUFFIXES)]
        if len(binaries) != 1:
            raise RuntimeError('one actual imported OpenSees extension binary required')
        binary = binaries[0]
        import_paths['opensees_binary'] = binary.__file__
        record.update(opensees_version=ops.version(),
            opensees_binary_sha256=sha256(Path(binary.__file__).read_bytes()).hexdigest())
    pools = threadpoolctl.threadpool_info()
    record['numerical_pools'] = pools
    record['import_source_sha256'] = {k: sha256(Path(p).read_bytes()).hexdigest()
                                    for k, p in import_paths.items()}
    if not pools or any(p['num_threads'] != 1 for p in pools):
        raise RuntimeError('effective numerical pools=1 required after route imports')
    return ops


def read_completed_input(path, route):
    status = json.loads((path.parent / 'status.json').read_text(encoding='utf-8'))
    if status.get('route') != route or status.get('status') != 'COMPLETED_SUPPLEMENTAL_' + route.upper():
        raise ValueError(f'{route} comparison input lacks completed route evidence')
    return json.loads(path.read_text(encoding='utf-8')), status


def matching_input_identity(native_status, external_status):
    """Compare originating runs, allowing a later comparator-only source edit."""
    source = native_status.get('script_sha256')
    if not isinstance(source, str) or not source or source != external_status.get('script_sha256'):
        raise ValueError('native/external originating script identities must match and be nonempty')
    # Matching originating source binds the immutable factory when both runs
    # predate explicit signatures. If either supplies one, both must match.
    fixture = native_status.get('fixture_signature_sha256')
    external_fixture = external_status.get('fixture_signature_sha256')
    if fixture != external_fixture or (fixture is not None and (
            not isinstance(fixture, str) or not fixture)):
        raise ValueError('native/external frozen fixture signatures must match')
    return dict(originating_script_sha256=source, fixture_signature_sha256=fixture)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('native', 'external', 'compare'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opensees-site', type=Path)
    parser.add_argument('--native-input', type=Path)
    parser.add_argument('--external-input', type=Path)
    args = parser.parse_args(argv)
    if args.route == 'external' and args.opensees_site is None:
        parser.error('--route external requires --opensees-site')
    if args.route == 'compare' and (args.native_input is None or args.external_input is None):
        parser.error('--route compare requires --native-input and --external-input')
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='IN_PROGRESS', claim='SUPPLEMENTAL_FEASIBILITY_ONLY',
        production_qualified=False, e04_external_row_closed=False,
        route=args.route, python_executable=sys.executable, python_version=sys.version,
        fixture_signature_sha256=fixture_signature(),
        sources=SOURCES, script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.output / 'status.json', record)
    try:
        if args.route == 'compare':
            native, native_status = read_completed_input(args.native_input, 'native')
            external, external_status = read_completed_input(args.external_input, 'external')
            record['comparison_input_identity'] = matching_input_identity(native_status, external_status)
            record['input_sha256'] = {str(p.resolve()): sha256(p.read_bytes()).hexdigest()
                                     for p in (args.native_input, args.external_input)}
            save(args.output / 'comparison.json', compare(native, external))
        else:
            ops = runtime_identity(record, args.route, args.opensees_site)
            save(args.output / 'status.json', record)
            if args.route == 'native':
                solve_native(args.output)
            else:
                solve_external(ops, args.output)
        record['status'] = 'COMPLETED_SUPPLEMENTAL_' + args.route.upper()
    except BaseException as exc:
        record.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save(args.output / 'status.json', record)


if __name__ == '__main__':
    main()
