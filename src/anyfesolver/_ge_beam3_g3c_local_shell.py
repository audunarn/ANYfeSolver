"""Private matrix-owned shell trial/work diagnostics, NOT G3c qualification.

No public element/recovery edits, accepted graph state, commit or restart API.
The Q4 signed work channels are not a single finite physical recovery field.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from threading import Lock
import numpy as np

from ._ge_beam3_mixed_ad import Jet2, constant_matrix, matmul, matvec, transpose, so3_exp, so3_log
from ._ge_beam3_pose_joint import _exp_terms
from ._ge_beam3_variational_shell import rotation_jets

POLICY = 'GE_BEAM3_G3C_MATRIX_POSE_SHELL_PULLBACK_V1'

def owned(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite shell input or contraction')
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)

def array(value, shape):
    if type(value) is not np.ndarray or value.dtype != np.dtype('float64') or value.shape != shape:
        raise ValueError('exact binary64 shell array required')
    return owned(value)

def rotations(value, n):
    q = array(value, (n, 3, 3))
    for v in q:
        product = owned(v.T@v)
        determinant = float(np.linalg.det(v))
        if not math.isfinite(determinant) or np.linalg.norm(product-np.eye(3)) > 1e-11 or abs(determinant-1) > 1e-11:
            raise ValueError('proper shell rotation required')
    return q

def jsonable(value):
    if isinstance(value, np.ndarray):
        return owned(value).tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError('string diagnostic keys required')
        return {k: jsonable(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [jsonable(v) for v in value]
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('unsupported diagnostic data')
    if type(value) is float and not math.isfinite(value):
        raise ValueError('nonfinite shell diagnostic')
    return value

def canonical(value):
    return (json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'),
                       allow_nan=False)+'\n').encode('ascii')

def equal(a, b, label):
    a, b = owned(a), owned(b)
    if a.shape != b.shape:
        raise ValueError(label)
    norms = [float(np.linalg.norm(v)) for v in (owned(a-b), a, b)]
    if not all(math.isfinite(v) for v in norms) or norms[0] > 1e-11*max(1., *norms[1:]):
        raise ValueError(label)

@dataclass(frozen=True)
class Kinematics:
    deformation: np.ndarray
    differential: np.ndarray
    second: np.ndarray
    frame: np.ndarray
    rotations: np.ndarray
    reference_positions: np.ndarray
    current_positions: np.ndarray

def deformation(reference, displacement, accepted_rotations):
    n = len(reference)
    if n not in (3, 4):
        raise ValueError('registered shell topology required')
    ref = array(reference, (n, 3)); u = array(displacement, (6*n,))
    qa = rotations(accepted_rotations, n)
    if np.any(np.linalg.norm(u.reshape(n, 6)[:, 3:], axis=1) >= .9*np.pi):
        raise ValueError('trial shell increment requires cutback')
    rotation = rotation_jets(ref, ref+u.reshape(n, 6)[:, :3])
    rt = transpose(rotation)
    z = [[Jet2.variable(u[6*i+j], 6*i+j, 6*n) for j in range(6)] for i in range(n)]
    x = [[z[i][j]+ref[i, j] for j in range(3)] for i in range(n)]
    centre = [sum(row[j] for row in x)/n for j in range(3)]
    refc = ref-ref.mean(axis=0)
    q = [matmul(so3_exp(z[i][3:]), constant_matrix(qa[i], 6*n)) for i in range(n)]
    values = []
    for i in range(n):
        local = matvec(rt, [x[i][j]-centre[j] for j in range(3)])
        values.extend(local[j]-refc[i, j] for j in range(3))
        values.extend(so3_log(matmul(rt, q[i])))
    return Kinematics(owned([v.value for v in values]), owned([v.gradient for v in values]),
        owned([v.hessian for v in values]), owned([[v.value for v in row] for row in rotation]),
        owned([[[v.value for v in row] for row in qi] for qi in q]),
        ref, owned(ref+u.reshape(n,6)[:,:3]))

@dataclass(frozen=True)
class WorkChannel:
    name: str
    sign: int
    energy: float
    force: np.ndarray
    tangent: np.ndarray
    stations: bytes
    numerical: bool

def channel(name, sign, energy, force, tangent, stations=None, *, numerical=False):
    value = float(energy)
    if not math.isfinite(value):
        raise ValueError('nonfinite channel energy')
    return WorkChannel(name, sign, sign*value, owned(sign*np.asarray(force)),
                       owned(sign*np.asarray(tangent)), canonical(stations), numerical)

def tensor(values, frame, shear_scale=1.):
    out = []
    for a, b, c in values:
        local = np.array([[a, shear_scale*c, 0.], [shear_scale*c, b, 0.], [0., 0., 0.]])
        out.append(frame@local@frame.T)
    return owned(out)

def physical_fields(public, frame, pose):
    """Separate source fields in global components; never a composite resultant."""
    reference = {}
    for name, factor in [('membrane_strain', .5), ('curvature', .5),
                         ('membrane_resultants', 1.), ('bending_resultants', 1.),
                         ('compatible_membrane_strain', .5), ('compatible_curvature', .5)]:
        if name in public: reference[name] = tensor(public[name], frame, factor)
    for name in ('transverse_shear_strain', 'transverse_shear_resultants', 'compatible_transverse_shear_strain'):
        if name in public: reference[name] = owned(np.asarray(public[name])@frame[:,:2].T)
    current = {k: owned(pose@v@pose.T if v.ndim == 3 else v@pose.T) for k,v in reference.items()}
    return dict(reference_global_fields=reference, current_global_fields=current,
                reference_frame=owned(frame), current_frame=owned(pose@frame))

def membrane_stations(strain, resultant, differential, weights, frame, positions, current_positions, pose):
    strain_global = tensor(strain, frame, .5)
    resultant_global = tensor(resultant, frame)
    return dict(strain=owned(strain), resultants=owned(resultant),
        strain_differential=owned(differential), weights=owned(weights),
        reference_frame=owned(frame), current_frame=owned(pose@frame),
        reference_station_positions=owned(positions),
        current_station_positions=owned(current_positions),
        differential_coordinates='REFERENCE_GLOBAL_LOCAL_DEFORMATION_D',
        reference_global_strain=strain_global, reference_global_resultants=resultant_global,
        current_global_strain=owned(pose@strain_global@pose.T),
        current_global_resultants=owned(pose@resultant_global@pose.T),
        recovery_scope='SEPARATE_SOURCE_WORK_CHANNEL_NOT_COMPOSITE_PHYSICAL_RECOVERY')

def q4_channels(element, mesh, material, kin, components):
    from . import e4_pl_element as source
    d, pose = kin.deformation, kin.frame
    mixed = element._recover_planar_mixed_fields(mesh, d, material, source._GAUSS)
    frame = mixed['frame']; c = source._coefficients(mixed['local_nodes'])
    select = np.eye(24)[[6*i+j for i in range(4) for j in range(5)]]
    B = owned([source._compatible(mixed['local_nodes'], c, r, s)@select@source._global_transform(frame).T
               for r, s in source._GAUSS])
    weights = owned(mixed['jacobian_determinants'])
    weak_force = owned(np.einsum('g,gij,gi->j', weights, B, mixed['resultants']))
    equal(weak_force, components['physical']@d, 'mixed stationary weak work')
    public = element.compute_stresses(mesh, d, material, return_global=True)
    physical_frame = element._physical_director_context(frame)[0]
    interpolation = owned([element.compute_shape_functions(float(r),float(t))[0] for r,t in source._GAUSS])
    baseline = dict(source_mixed=mixed, source_physical=public,
        compatible_differential=B, weights=weights, current_pose=pose,
        differential_coordinates='NUMBERED_ENGINEERING_ROWS_REFERENCE_GLOBAL_LOCAL_D_COLUMNS',
        reference_station_positions=owned(interpolation@kin.reference_positions),
        current_station_positions=owned(interpolation@kin.current_positions),
        **physical_fields(public, physical_frame, pose),
        current_global_membrane_resultants=pose@public['global_membrane_resultant_tensors']@pose.T,
        current_global_bending_resultants=pose@public['global_bending_resultant_tensors']@pose.T,
        current_global_shear_resultants=public['global_transverse_shear_resultants']@pose.T,
        recovery_scope='QUALIFIED_LINEAR_MIXED_BASELINE_ONLY')
    records = []
    for name, key, numerical in [('QUALIFIED_MIXED_PHYSICAL', 'physical', False),
                                 ('NUMERICAL_PL', 'pl', True),
                                 ('NUMERICAL_HOURGLASS', 'hourglass', True)]:
        k = owned(components[key])
        records.append(channel(name, 1, .5*d@k@d, k@d, k, baseline if not numerical else None,
                               numerical=numerical))
    geometry = element._nonlinear_geometry(mesh)
    T0, centre_frame = owned(geometry['T0']), owned(geometry['R0'])
    y = T0@d
    C = material.elastic_modulus/(1-material.poisson_ratio**2)*np.array(
        [[1., material.poisson_ratio, 0.], [material.poisson_ratio, 1., 0.],
         [0., 0., (1-material.poisson_ratio)/2]])
    A = element.thickness*C
    coordinates = element.get_node_coordinates(mesh)
    interpolation = owned([element.compute_shape_functions(float(r), float(s))[0] for r,s in element.gauss_points])
    positions = interpolation@coordinates
    for nonlinear, name, sign in [(True, 'COMPATIBLE_VK_MEMBRANE', 1),
                                   (False, 'REMOVED_COMPATIBLE_LINEAR_MEMBRANE', -1)]:
        energy = 0.; force = np.zeros(24); stiffness = np.zeros((24, 24))
        strains = []; resultants = []; operators = []; source_weights = []
        for gp in geometry['gp']:
            bm, gw, w = owned(gp['B_m']), owned(gp['Gw']), float(gp['detw'])
            e = bm@y; p = gw@y; beff = bm.copy()
            if nonlinear:
                e = e+np.array([.5*p[0]**2, .5*p[1]**2, p[0]*p[1]])
                beff += np.array([p[0]*gw[0], p[1]*gw[1], p[0]*gw[1]+p[1]*gw[0]])
            N = A@e
            energy += .5*w*e@N
            force += w*beff.T@N
            stiffness += w*beff.T@A@beff
            if nonlinear:
                stiffness += w*gw.T@np.array([[N[0], N[2]], [N[2], N[1]]])@gw
            strains.append(e); resultants.append(N); operators.append(beff@T0); source_weights.append(w)
        stations = membrane_stations(strains, resultants, operators, source_weights,
                                     centre_frame, positions, interpolation@kin.current_positions, pose)
        records.append(channel(name, sign, energy, T0.T@force, T0.T@stiffness@T0, stations))
    # Single elastic law: the qualified Q4 adds the von Karman cross terms
    # dE_m' A eps_nl + B_nl' A dE_m u (energy sum w (dE_m u).A eps_nl) that the
    # mixed + compatible-VK - compatible-linear split above does not contain.
    from . import _q4_assumed_strain_plasticity as single
    Gw_all, detw = np.asarray(geometry['Gw_all'], float), np.asarray(geometry['detw_all'], float)
    dEm = np.asarray(single.offset_operator(element, mesh, material), float)[:, :3, :]
    theta = np.einsum('gki,i->gk', Gw_all, y)
    Bnl = np.stack([theta[:, 0, None]*Gw_all[:, 0], theta[:, 1, None]*Gw_all[:, 1],
                    theta[:, 0, None]*Gw_all[:, 1]+theta[:, 1, None]*Gw_all[:, 0]], axis=1)
    enl = np.stack([.5*theta[:, 0]**2, .5*theta[:, 1]**2, theta[:, 0]*theta[:, 1]], axis=1)
    offset_strain = np.einsum('gki,i->gk', dEm, y)
    N_offset = offset_strain@A.T
    cross_force = (np.einsum('g,gki,gk->i', detw, dEm, enl@A.T)
                   + np.einsum('g,gki,gk->i', detw, Bnl, N_offset))
    cross = np.einsum('gki,gkj->ij', dEm*detw[:, None, None], A@Bnl)
    cross_tangent = cross+cross.T+single._geometric(detw[None], Gw_all[None], N_offset[None])[0]
    cross_energy = float(np.einsum('g,gk,gk->', detw, offset_strain, enl@A.T))
    records.append(channel('VK_OFFSET_CROSS_TERM', 1, cross_energy, T0.T@cross_force,
                           T0.T@cross_tangent@T0, None, numerical=True))
    return tuple(records)

def s3_channels(element, mesh, material, kin, components, candidate):
    d = kin.deformation
    geometry = element._geometry(mesh)
    raw = element._elastic_baseline_constitutive(material)
    ops = element._effective_station_operators(element._operators(geometry, raw))
    transform = owned(geometry['local_from_external'])
    Binternal = np.array([np.vstack(rows)@transform for rows in zip(*ops)])
    order = geometry['internal_order']; B = np.empty_like(Binternal)
    for internal, external in enumerate(order): B[external] = Binternal[internal]
    resultants = owned(candidate['station_generalized_resultant'])
    weights = np.full(3, float(geometry['area'])/3)
    equal(np.einsum('g,gij,gi->j', weights, B, resultants), components['physical']@d,
          'S3 actual station weak work')
    public = element.compute_stresses(mesh, d, material, return_global=True)
    interpolation = owned(public['external_barycentric_coordinates'])
    physical = dict(source_physical=public, strain_differential=owned(B), weights=owned(weights),
        station_strain=owned(candidate['station_generalized_strain']), resultants=resultants,
        **physical_fields(public, geometry['frame'], kin.frame),
        differential_coordinates='SOURCE_ENGINEERING_ROWS_REFERENCE_GLOBAL_LOCAL_D_COLUMNS',
        reference_station_positions=owned(public['physical_station_coordinates']),
        current_station_positions=owned(interpolation@kin.current_positions),
        recovery_scope='SOURCE_S3_LOCAL_STATION_VALUES', current_pose=kin.frame)
    return tuple(channel(name, 1, .5*d@components[key]@d, components[key]@d, components[key],
                         None if numerical else physical, numerical=numerical)
                 for name, key, numerical in [('S3_PHYSICAL', 'physical', False), ('NUMERICAL_PL', 'pl', True)])

@dataclass(frozen=True)
class LocalShellTrial:
    kinematics: Kinematics
    local_force: np.ndarray
    local_tangent: np.ndarray
    chart_force: np.ndarray
    chart_hessian: np.ndarray
    spatial_force: np.ndarray
    spatial_row_chart_tangent: np.ndarray
    energy: float
    channels: tuple
    candidate: bytes
    definition_sha256: str
    production_qualified: bool = field(default=False, init=False)
    recovery_complete: bool = field(default=False, init=False)
    state_committed: bool = field(default=False, init=False)

def family_objects(description):
    from .fe_core import FEModel
    from .e4_pl_element import QualifiedE4PLShellElement
    from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    model = FEModel('owned G3c matrix shell virgin trial')
    model.add_material('scalar', 100., .25)
    for node, point in zip(description['node_ids'], description['coordinates']): model.add_node(node, *point)
    cls = QualifiedE4PLShellElement if description['family'] == 'Q4' else NativeParityE4PLS3V2DShellElement
    element = cls(1, description['node_ids'], 'scalar', thickness=.1,
        reference_normal=np.array(description['normal']), material_direction=np.array(description['material_direction']),
        director_polarity=description['director_polarity'])
    model.add_element(1, element)
    material = model.get_material('scalar')
    origin = (element.init_nonlinear_state(3) if description['family'] == 'Q4'
              else element.init_model_bound_nonlinear_state(model.mesh, material, 3))
    return model, element, material, origin

class LocalShell:
    """Closed scalar facade, owned snapshots, fresh family origin, no commit API."""
    __slots__ = ('_body', '_seal', '_lock')
    def __setattr__(self, name, value):
        if hasattr(self, name): raise AttributeError('write-once shell definition')
        object.__setattr__(self, name, value)
    def __delattr__(self, name):
        raise AttributeError('shell definition cannot be deleted')

    def __init__(self, family, node_ids, coordinates, normal, material_direction, *, director_polarity=1):
        if type(family) is not str or family not in ('Q4', 'S3-V2D'):
            raise ValueError('registered shell family required')
        n = 4 if family == 'Q4' else 3
        if (type(node_ids) is not tuple or len(node_ids) != n or len(set(node_ids)) != n
                or any(type(i) is not int or i <= 0 for i in node_ids)):
            raise ValueError('exact distinct physical node IDs required')
        ref = array(coordinates, (n, 3)); normal = array(normal, (3,)); direction = array(material_direction, (3,))
        if abs(np.linalg.norm(normal)-1) > 1e-11 or abs(np.linalg.norm(direction)-1) > 1e-11 or abs(normal@direction) > 1e-11:
            raise ValueError('orthonormal physical normal/material direction required')
        relative = owned(ref-ref[0]); size = float(np.linalg.norm(relative))
        if not math.isfinite(size) or size == 0 or np.max(abs(relative@normal)) > 1e-11*size:
            raise ValueError('finite nondegenerate planar shell reference required')
        if len(np.unique(ref, axis=0)) != n:
            raise ValueError('distinct shell reference nodes required')
        if type(director_polarity) is not int or director_polarity not in (-1, 1):
            raise ValueError('explicit director polarity required')
        self._body = canonical(dict(policy=POLICY, family=family, node_ids=node_ids, coordinates=ref,
            normal=normal, material_direction=direction, director_polarity=director_polarity,
            E=100., nu=.25, thickness=.1, layers=3, origin='OWNED_VIRGIN_ONLY'))
        self._seal = sha256(self._body).hexdigest(); self._lock = Lock()

    def descriptor(self):
        if type(self) is not LocalShell or type(self._body) is not bytes or sha256(self._body).hexdigest() != self._seal:
            raise ValueError('shell definition changed')
        return json.loads(self._body)

    def evaluate(self, displacement, accepted_rotations):
        lock = self._lock
        if not lock.acquire(blocking=False): raise RuntimeError('concurrent shell evaluation forbidden')
        try:
            body, seal = self._body, self._seal
            description = self.descriptor(); n = len(description['node_ids'])
            u = array(displacement, (6*n,)); qa = rotations(accepted_rotations, n)
            kin = deformation(np.array(description['coordinates']), u, qa)
            model, element, material, origin = family_objects(description)
            before = canonical(origin)
            f, k, candidate = element.compute_nonlinear_response(model.mesh, material, kin.deformation, origin, 3, True)
            # Actual kernels may return aliases in the virgin elastic shortcut.
            # Nothing is shared with a caller or persisted as an accepted state.
            if not before: raise ValueError('missing owned origin')
            f = array(np.asarray(f), (6*n,)); k = array(np.asarray(k), (6*n, 6*n))
            equal(k, k.T, 'nonconservative local tangent')
            candidate_bytes = canonical(candidate)
            components = element.compute_stiffness_components(model.mesh, material)
            records = (q4_channels(element, model.mesh, material, kin, components)
                       if description['family'] == 'Q4'
                       else s3_channels(element, model.mesh, material, kin, components, candidate))
            equal(owned(sum(c.force for c in records)), f, 'signed channel force differs from actual source')
            equal(owned(sum(c.tangent for c in records)), k, 'signed channel tangent differs from actual source')
            energy = sum(c.energy for c in records)
            if not math.isfinite(energy): raise ValueError('nonfinite total shell energy')
            D, S = kin.differential, kin.second
            g = owned(D.T@f); H = owned(D.T@k@D+np.einsum('i,ijk->jk', f, S))
            P = np.eye(6*n); terms = []
            for i, row in enumerate(u.reshape(n, 6)):
                _, J, dJ = _exp_terms(row[3:]); sl = slice(6*i+3, 6*i+6)
                P[sl, sl] = J; terms.append((sl, dJ))
            r = owned(np.linalg.solve(P.T, g)); correction = np.zeros_like(H)
            for sl, dJ in terms: correction[sl, sl] = np.einsum('ijk,i->jk', dJ, r[sl])
            J = owned(np.linalg.solve(P.T, H-correction))
            self.descriptor()
            if self._body != body or self._seal != seal or self._lock is not lock:
                raise ValueError('shell identity changed during trial')
            return LocalShellTrial(kin, f, k, g, H, r, J, energy, records, candidate_bytes, seal)
        finally:
            lock.release()
