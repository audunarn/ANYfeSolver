"""Fixed E01 diagnostic: physical finite-vertex B2 point law in OpenSees.

Run only after the caller freezes this source, runtime and inputs. Requires the
verified Python 3.12/OpenSees 3.8 runtime. No ANYfeSolver or NumPy dependency.
Run exactly one named case per fresh process; repeated-domain execution failed
in the retained diagnostic despite a successful identical fresh-process case.
This is a reference-linear, scalar elastic, two-Q4/two-B2 comparison, not GE
nodal pose/BL5, finite B2, public S3, mesh convergence or qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback


E = 210.e9
NU = .3
G = E / (2. * (1. + NU))
H = .07
A = .01
IY = IZ = 8.333333333333333e-6
J = 1.6666666666666667e-5
KY = KZ = .8
LREF = 2.
SHELL_NODES = {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (2., 0., 0.),
               4: (0., 1., 0.), 5: (1., 1., 0.), 6: (2., 1., 0.)}
CELLS = ((1, 2, 5, 4), (2, 3, 6, 5))
BEAMS = ((101, 102), (102, 103))
NODE_IDS = tuple(SHELL_NODES) + (101, 102, 103)
INDEX = {node: 6 * i for i, node in enumerate(NODE_IDS)}
LOADS = (('Fx', 0, 1000.), ('Fy', 1, 1000.), ('Fz', 2, 1000.),
         ('Mx', 3, 100.), ('My', 4, 100.), ('Mz', 5, 100.))
LAW = 'B2_FINITE_Q4_POINT_REFERENCE_LINEAR_V1_DEV'
POSE_SCALE = (LREF, LREF, LREF, 1., 1., 1.)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def mv(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return [list(row) for row in zip(*a, strict=True)]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def skew(v):
    x, y, z = v
    return [[0., -z, y], [z, 0., -x], [-y, x, 0.]]


def solve(a, b):
    """Row-scaled partial-pivot elimination for these fixed small systems."""
    n = len(b)
    if len(a) != n or any(len(row) != n for row in a):
        raise ValueError('square fixed reference system required')
    rows = []
    for row, value in zip(a, b, strict=True):
        scale = max(map(abs, row))
        if not math.isfinite(scale) or scale == 0.:
            raise ValueError('singular fixed reference system')
        rows.append([v / scale for v in row] + [value / scale])
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
        rows[k], rows[pivot] = rows[pivot], rows[k]
        if abs(rows[k][k]) <= 128. * sys.float_info.epsilon:
            raise ValueError('unresolved pivot in fixed reference system')
        for i in range(k + 1, n):
            factor = rows[i][k] / rows[k][k]
            rows[i][k] = 0.
            for j in range(k + 1, n + 1):
                rows[i][j] = math.fsum((rows[i][j], -factor * rows[k][j]))
    result = [0.] * n
    for i in range(n - 1, -1, -1):
        result[i] = (rows[i][n] - dot(rows[i][i+1:n], result[i+1:])) / rows[i][i]
    if not all(map(math.isfinite, result)):
        raise ValueError('nonfinite fixed reference solution')
    return result


def beam_pose(x, initial_pose, initial_resultants):
    """Scalar force-method ODE, local y=global y, local z=global z.

    N,Vy,Vz,T constant; My=My0+Vz*x; Mz=Mz0-Vy*x. No production
    interpolation, internal station nodes or Euler-Bernoulli substitution.
    """
    u, v, w, tx, ty, tz = initial_pose
    n, vy, vz, torque, my, mz = initial_resultants
    return [u+x*n/(E*A),
            v+x*tz+x*vy/(G*A*KY)+x*x*mz/(2.*E*IZ)-x**3*vy/(6.*E*IZ),
            w-x*ty+x*vz/(G*A*KZ)-x*x*my/(2.*E*IY)-x**3*vz/(6.*E*IY),
            tx+x*torque/(G*J),
            ty+x*my/(E*IY)+x*x*vz/(2.*E*IY),
            tz+x*mz/(E*IZ)-x*x*vy/(2.*E*IZ)]


def beam_port(length, station):
    zero = [0.] * 6
    compliance = transpose([beam_pose(length, zero,
                             [float(i == j) for i in range(6)]) for j in range(6)])
    columns = []
    for j in range(12):
        nodal = [float(i == j) for i in range(12)]
        baseline = beam_pose(length, nodal[:6], zero)
        initial = solve(compliance, [v-b for v, b in zip(nodal[6:], baseline, strict=True)])
        columns.append(beam_pose(station*length, nodal[:6], initial))
    return transpose(columns)


def shell_port(vertices):
    center = [math.fsum(x[j] for x in vertices)/4. for j in range(3)]
    radii = [[x[j]-center[j] for j in range(3)] for x in vertices]
    covariance = [[math.fsum(dot(r, r)*float(i == j)-r[i]*r[j] for r in radii)
                   for j in range(3)] for i in range(3)]
    port = [[0.] * 24 for _ in range(6)]
    for node, radius in enumerate(radii):
        s = skew(radius)
        for component in range(3):
            port[component][6*node+component] = .25
            rotation = solve(covariance, [s[i][component] for i in range(3)])
            for i in range(3):
                port[3+i][6*node+component] = rotation[i]
    return center, covariance, port


def panel_input(sign):
    coordinates = dict(SHELL_NODES)
    coordinates.update({node: (float(i), .5, sign*.1)
                        for i, node in enumerate((101, 102, 103))})
    fixed = {INDEX[node]+j for node in (1, 4, 101) for j in range(6)}
    fixed.update(INDEX[node]+5 for node in (2, 3, 5, 6))
    rows, ports, equations = [], [], []
    pb = beam_port(1., .5)
    for point, (cell, beam) in enumerate(zip(CELLS, BEAMS, strict=True)):
        center, covariance, ps = shell_port([coordinates[node] for node in cell])
        station = [(coordinates[beam[0]][j]+coordinates[beam[1]][j])/2. for j in range(3)]
        offset = [station[j]-center[j] for j in range(3)]
        lever = skew(offset)
        point_rows = [[0.] * 54 for _ in range(6)]
        for row in range(6):
            for node_index, node in enumerate(cell):
                for dof in range(6):
                    col = 6*node_index+dof
                    value = (-ps[row][col] + dot(lever[row], [ps[3+i][col] for i in range(3)])
                             if row < 3 else -ps[row][col])
                    point_rows[row][INDEX[node]+dof] = value
            for node_index, node in enumerate(beam):
                for dof in range(6):
                    point_rows[row][INDEX[node]+dof] = pb[row][6*node_index+dof]
            dependent = INDEX[beam[1]]+row
            coefficient = point_rows[row][dependent]
            if coefficient == 0.:
                raise ValueError('declared unique dependent coefficient is zero')
            terms = [(beam[1], row+1, 1.)]
            for node in NODE_IDS:
                for dof in range(6):
                    index = INDEX[node]+dof
                    value = point_rows[row][index]
                    if index != dependent and value != 0.:
                        terms.append((node, dof+1, value/coefficient))
            equations.append({'dependent': [beam[1], row+1],
                              'physical_row': 6*point+row,
                              'normalizing_coefficient': coefficient, 'terms': terms})
        rows.extend(point_rows)
        ports.append({'point': point, 'shell_nodes': cell, 'beam_nodes': beam,
                      'shell_centroid_m': center, 'beam_station_m': station,
                      'relative_offset_m': offset, 'covariance_m2': covariance,
                      'shell_port': ps, 'beam_port': pb})
    return {'sign': sign, 'law': LAW, 'node_order': NODE_IDS,
            'coordinates_m': coordinates, 'shell_cells': CELLS, 'beam_members': BEAMS,
            'fixed_coordinates': sorted(fixed), 'constraint_rows': rows,
            'point_ports': ports, 'equation_constraints': equations,
            'beam_section_offset_m': [0., 0., 0.]}


def normalized_gap(gap):
    return max(abs(value)/POSE_SCALE[i % 6] for i, value in enumerate(gap))


def rigid_guards(inputs):
    report = []
    for model in inputs:
        errors = []
        for direction in range(6):
            translation = [float(i == direction)*.01*LREF for i in range(3)]
            spin = [float(i+3 == direction)*.01 for i in range(3)]
            field = []
            for node in NODE_IDS:
                arm = cross(spin, model['coordinates_m'][node])
                field.extend([translation[i]+arm[i] for i in range(3)]+spin)
            errors.append(normalized_gap(mv(model['constraint_rows'], field)))
        report.append({'sign': model['sign'], 'six_rigid_errors': errors,
                       'limit': 1.e-9, 'passed': all(math.isfinite(v) and v <= 1.e-9 for v in errors)})
    return report


def finite_vector(values, size):
    result = list(map(float, values))
    if len(result) != size or not all(map(math.isfinite, result)):
        raise ValueError('finite complete external vector required')
    return result


def analyze(ops):
    ops.constraints('Lagrange', 1.e9, 1.e9)
    ops.numberer('Plain')
    ops.system('BandGeneral')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.)
    ops.analysis('Static')
    return int(ops.analyze(1))


def element_beam(ops, tag, nodes):
    ops.element('ElasticTimoshenkoBeam', tag, *nodes, E, G, A, J, IY, IZ,
                A*KY, A*KZ, 1)


def beam_control(ops, record, component, magnitude):
    ops.wipe()
    try:
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        ops.node(1, 0., 0., 0.); ops.node(2, LREF, 0., 0.)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.geomTransf('Linear', 1, 0., 0., 1.)
        element_beam(ops, 1, (1, 2))
        load = [0.] * 6; load[component] = magnitude
        ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1); ops.load(2, *load)
        record['solver_code'] = analyze(ops)
        if record['solver_code'] != 0:
            raise RuntimeError('external beam control solve failed')
        states = {node: list(map(float, ops.nodeDisp(node))) for node in (1, 2)}
        record['nodal_states'] = states
        observed = finite_vector(states[2], 6)
        expected = [0.] * 6
        if component == 0:
            expected[0] = magnitude*LREF/(E*A)
        elif component == 1:
            expected[1] = magnitude*(LREF**3/(3.*E*IZ)+LREF/(G*A*KY))
            expected[5] = magnitude*LREF**2/(2.*E*IZ)
        elif component == 2:
            expected[2] = magnitude*(LREF**3/(3.*E*IY)+LREF/(G*A*KZ))
            expected[4] = -magnitude*LREF**2/(2.*E*IY)
        elif component == 3:
            expected[3] = magnitude*LREF/(G*J)
        elif component == 4:
            expected[2] = -magnitude*LREF**2/(2.*E*IY)
            expected[4] = magnitude*LREF/(E*IY)
        else:
            expected[1] = magnitude*LREF**2/(2.*E*IZ)
            expected[5] = magnitude*LREF/(E*IZ)
        scale = max(abs(v)/s for v, s in zip(expected, POSE_SCALE, strict=True))
        error = max(abs(v-r)/s for v, r, s in zip(observed, expected, POSE_SCALE, strict=True))/scale
        ops.reactions()
        record.update(tip_pose=observed, analytic_tip_pose=expected,
                      raw_opensees_node_reactions={node: list(map(float, ops.nodeReaction(node))) for node in (1, 2)},
                      element_global_covector=finite_vector(ops.eleForce(1), 12),
                      metric=error, limit=1.e-10, passed=math.isfinite(error) and error <= 1.e-10)
    finally:
        ops.wipe()


def wrench(vector, coordinates):
    force, moment = [0.] * 3, [0.] * 3
    for node in NODE_IDS:
        base = INDEX[node]; f = vector[base:base+3]; m = vector[base+3:base+6]
        arm = cross(coordinates[node], f)
        for i in range(3):
            force[i] += f[i]; moment[i] += m[i]+arm[i]
    return force, moment


def point_covector(model, multipliers):
    """Independent shell vertex wrench pullback; no nodal drill actions."""
    result = [0.] * 54
    for point in model['point_ports']:
        index = 6*point['point']; lam = multipliers[index:index+6]
        master_force = [-v for v in lam[:3]]
        lever = cross(point['relative_offset_m'], lam[:3])
        master_moment = [-lever[i]-lam[3+i] for i in range(3)]
        couple = solve(point['covariance_m2'], master_moment)
        for node in point['shell_nodes']:
            radius = [model['coordinates_m'][node][i]-point['shell_centroid_m'][i] for i in range(3)]
            pair = cross(couple, radius)
            for i in range(3):
                result[INDEX[node]+i] += master_force[i]/4.+pair[i]
        beam_force = mv(transpose(point['beam_port']), lam)
        for j, node in enumerate(point['beam_nodes']):
            for i in range(6):
                result[INDEX[node]+i] += beam_force[6*j+i]
    return result


def panel_case(ops, record, model, component, magnitude):
    ops.wipe()
    try:
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        for node in NODE_IDS:
            ops.node(node, *model['coordinates_m'][node])
        fixed = set(model['fixed_coordinates'])
        for node in NODE_IDS:
            flags = [int(INDEX[node]+i in fixed) for i in range(6)]
            if any(flags): ops.fix(node, *flags)
        ops.section('ElasticMembranePlateSection', 1, E, NU, H, 0.)
        elements = [(i+1, cell) for i, cell in enumerate(CELLS)]
        for tag, cell in elements:
            ops.element('ASDShellQ4', tag, *cell, 1)
        ops.geomTransf('Linear', 1, 0., 0., 1.)
        for i, beam in enumerate(BEAMS):
            element_beam(ops, 101+i, beam); elements.append((101+i, beam))
        for equation in model['equation_constraints']:
            flat = [value for term in equation['terms'] for value in term]
            ops.equationConstraint(*flat)
        applied = [0.] * 54; applied[INDEX[103]+component] = magnitude
        load = [0.] * 6; load[component] = magnitude
        ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1); ops.load(103, *load)
        record['solver_code'] = analyze(ops)
        if record['solver_code'] != 0:
            raise RuntimeError('external panel solve failed')
        states = {node: list(map(float, ops.nodeDisp(node))) for node in NODE_IDS}
        record['nodal_states'] = states
        displacement = finite_vector([v for node in NODE_IDS for v in states[node]], 54)
        internal = [0.] * 54; covectors = []
        record['element_global_covectors'] = covectors
        for tag, nodes in elements:
            force = list(map(float, ops.eleForce(tag)))
            covectors.append({'element': tag, 'nodes': nodes, 'global_covector': force})
            force = finite_vector(force, 6*len(nodes))
            for j, node in enumerate(nodes):
                for i in range(6): internal[INDEX[node]+i] += force[6*j+i]
        c = model['constraint_rows']; free_beam = list(range(INDEX[102], 54))
        cb = [[row[index] for index in free_beam] for row in c]
        multipliers = solve(transpose(cb), [applied[i]-internal[i] for i in free_beam])
        interface = point_covector(model, multipliers)
        transposed = mv(transpose(c), multipliers)
        residual = [internal[i]+interface[i]-applied[i] for i in range(54)]
        ground = [residual[i] if i in fixed else 0. for i in range(54)]
        ops.reactions()
        record.update(applied_covector=applied, internal_covector=internal,
                      interface_covector=interface, physical_point_multipliers=multipliers,
                      actual_ground_covector=ground,
                      raw_opensees_node_reactions={node: list(map(float, ops.nodeReaction(node))) for node in NODE_IDS})
        force_scale = magnitude if component < 3 else magnitude/LREF
        moment_scale = force_scale*LREF
        scale = [force_scale if i % 6 < 3 else moment_scale for i in range(54)]
        metrics = {'free_equilibrium_scaled': max(abs(residual[i])/scale[i] for i in range(54) if i not in fixed),
                   'gap_scaled': normalized_gap(mv(c, displacement)),
                   'interface_covector_scaled': max(abs(interface[i]-transposed[i])/scale[i] for i in range(54))}
        for label, vector in (('internal', internal), ('interface', interface),
                              ('external_and_ground', [applied[i]+ground[i] for i in range(54)])):
            force, moment = wrench(vector, model['coordinates_m'])
            metrics[label+'_force_wrench_scaled'] = max(map(abs, force))/force_scale
            metrics[label+'_moment_wrench_scaled'] = max(map(abs, moment))/moment_scale
        energy2 = dot(internal, displacement); applied_work = dot(applied, displacement)
        interface_work = dot(interface, displacement); ground_work = dot(ground, displacement)
        floor = force_scale*LREF*1.e-30
        metrics['energy_work_relative'] = abs(energy2+interface_work-applied_work-ground_work)/max(abs(energy2), abs(applied_work), floor)
        variation = [0. if i in fixed else .01*POSE_SCALE[i % 6]*math.sin(.4+.17*i) for i in range(54)]
        work_i = dot(internal, variation); work_e = dot(applied, variation)
        work_c = dot(interface, variation); dual_work = dot(multipliers, mv(c, variation))
        work_scale = max(abs(work_i), abs(work_e), abs(work_c), floor)
        metrics['virtual_work_relative'] = abs(work_i+work_c-work_e)/work_scale
        metrics['interface_virtual_work_relative'] = abs(work_c-dual_work)/work_scale
        shell_variation = variation[:]; shell_variation[INDEX[102]:] = [0.]*12
        compatible_beam = solve(cb, [-v for v in mv(c, shell_variation)])
        compatible = shell_variation[:]; compatible[INDEX[102]:] = compatible_beam
        ci, ce = dot(internal, compatible), dot(applied, compatible)
        metrics['compatible_virtual_work_relative'] = abs(ci-ce)/max(abs(ci), abs(ce), floor)
        metrics['compatible_variation_gap_scaled'] = normalized_gap(mv(c, compatible))
        checks = {key: {'value': value, 'limit': 1.e-9 if 'gap' in key else 1.e-8,
                        'passed': math.isfinite(value) and value <= (1.e-9 if 'gap' in key else 1.e-8)}
                  for key, value in metrics.items()}
        record.update(scales={'force_n': force_scale, 'moment_nm': moment_scale, 'length_m': LREF},
                      internal_strain_energy=.5*energy2,
                      actual_work={'twice_internal_energy': energy2, 'applied': applied_work,
                                   'interface': interface_work, 'ground': ground_work},
                      arbitrary_supported_variation=variation, compatible_variation=compatible,
                      checks=checks, passed=all(item['passed'] for item in checks.values()))
    finally:
        ops.wipe()


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if isinstance(value, dict): return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    return value


def atomic_json(path, value):
    payload = json.dumps(safe_json(value), indent=2, allow_nan=False)+'\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.'+path.name+'.', delete=False) as stream:
            temporary = Path(stream.name); stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='exclusive new directory; an existing path is rejected')
    names = ['beam-'+name for name, _, _ in LOADS]
    names += [f'panel-{sign}-{name}' for sign in (-1, 1) for name, _, _ in LOADS]
    parser.add_argument('--case', choices=names, required=True,
                        help='one case in a fresh process; do not reuse a domain')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'FAILED', 'scope': __doc__, 'cases': [],
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'selected_case': args.case,
               'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}
    try:
        inputs = [panel_input(sign) for sign in (-1, 1)]
        atomic_json(output/'inputs.json', {'panels': inputs, 'beam_control_length_m': LREF,
                    'beam_control_nodes_m': {1: [0., 0., 0.], 2: [LREF, 0., 0.]},
                    'beam_control_fixed_node': 1, 'beam_control_loaded_node': 2,
                    'pose_order': ['ux_m', 'uy_m', 'uz_m', 'rx_rad', 'ry_rad', 'rz_rad'],
                    'covector_order': ['Fx_N', 'Fy_N', 'Fz_N', 'Mx_Nm', 'My_Nm', 'Mz_Nm'],
                    'global_wrench_origin_m': [0., 0., 0.], 'beam_local_y_global': [0., 1., 0.],
                    'loads': LOADS, 'section': {'E_pa': E, 'nu': NU, 'G_pa': G, 'area_m2': A,
                    'Iy_m4': IY, 'Iz_m4': IZ, 'J_m4': J, 'ky': KY, 'kz': KZ},
                    'shell': {'E_pa': E, 'nu': NU, 'thickness_m': H},
                    'external': {'shell': 'ASDShellQ4/ElasticMembranePlateSection',
                                 'beam': 'ElasticTimoshenkoBeam', 'geometric_transformation': 'Linear',
                                 'constraints': 'exact Lagrange, both scales 1e9',
                                 'system': 'BandGeneral', 'algorithm': 'Linear'},
                    'reaction_semantics': 'physical C multipliers recovered from free beam equilibrium; ground from full structural+interface-applied covectors; raw nodeReaction is separate',
                    'limitations': 'one fixed mesh, no convergence or 1% cross-product qualification; ASDShellQ4 and product shell formulations remain separately identified'})
        guards = rigid_guards(inputs); atomic_json(output/'rigid-guards.json', guards)
        if not all(item['passed'] for item in guards): raise RuntimeError('common rigid-field guard failed')
        if sys.version_info[:2] != (3, 12): raise RuntimeError('use the verified Python 3.12 runtime, not default Python 3.14')
        import openseespy.opensees as ops
        summary['opensees_version'] = ops.version()
        summary['openseespy_distribution'] = importlib.metadata.version('openseespy')
        summary['native_modules'] = {name: {'path': str(Path(module.__file__).resolve()),
                                    'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                                    for name, module in list(sys.modules.items())
                                    if 'opensees' in name and isinstance(getattr(module, '__file__', None), str)
                                    and module.__file__.endswith(('.pyd', '.so'))}
        if not str(summary['opensees_version']).startswith('3.8'): raise RuntimeError('OpenSees 3.8 runtime required')
        if not summary['native_modules']: raise RuntimeError('OpenSees native binary identity required')
        if not callable(getattr(ops, 'equationConstraint', None)): raise RuntimeError('verified equationConstraint API required')
        jobs = [('beam-'+name, None, component, magnitude) for name, component, magnitude in LOADS]
        jobs += [(f'panel-{model["sign"]}-{name}', model, component, magnitude)
                 for model in inputs for name, component, magnitude in LOADS]
        jobs = [job for job in jobs if job[0] == args.case]
        for name, model, component, magnitude in jobs:
            record = {'case': name, 'status': 'FAILED', 'component': component, 'magnitude': magnitude,
                      'panel_sign': None if model is None else model['sign']}
            try:
                if model is None: beam_control(ops, record, component, magnitude)
                else: panel_case(ops, record, model, component, magnitude)
                if not record['passed']: raise RuntimeError('frozen physical predicate failed: '+name)
                record['status'] = 'PASSED_PHYSICAL_CHECKS'
            except BaseException:
                record['traceback'] = traceback.format_exc(); raise
            finally:
                atomic_json(output/(name+'.json'), record); summary['cases'].append(name+'.json')
        summary['status'] = 'PASSED_FIXED_REFERENCE_DIAGNOSTIC_ONLY'
    except BaseException:
        summary['traceback'] = traceback.format_exc(); raise
    finally:
        atomic_json(output/'summary.json', summary)


if __name__ == '__main__':
    main()
