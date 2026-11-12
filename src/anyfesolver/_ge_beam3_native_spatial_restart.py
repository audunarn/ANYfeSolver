"""Authenticated accepted chains for native line forces and spatial nodal couples.

Couples are spatial nonconservative external work, not material resultants or
an energy potential. The caller must authenticate the complete checkpoint hash;
a resealed self hash alone is not authority for a replaced valid history.
"""
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
import numpy as np
from ._native_rotation_state import rotation_exponential
from ._ge_beam3_spatial_nodal_moments import SpatialNodalMoments, POLICY
from ._ge_beam3_native_line_loading import nodal_force_vector
from ._ge_beam3_native_line_restart import LoadPoint as LineLoadPoint, _point as _line_point, _state, _model
from ._ge_beam3_native_fibre_restart import _keys, _array
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES

SCHEMA = 'GE_BEAM3_SUPPORTED_NATIVE_SPATIAL_COUPLE_RESTART_V1'


def capture_moments(value, model):
    if value is None:
        return None
    if type(value) is not SpatialNodalMoments:
        raise ValueError('exact native restart spatial moment pattern')
    made = SpatialNodalMoments(value.rows)
    if any(row[0] not in model.mesh.nodes for row in made.rows):
        raise ValueError('native restart moment node absent')
    return made


@dataclass(frozen=True)
class LoadPoint:
    line: LineLoadPoint
    constant_moments: object
    proportional_moments: object

    def require(self, model):
        if type(self.line) is not LineLoadPoint:
            raise ValueError('exact native spatial restart line point')
        self.line.require(model)
        capture_moments(self.constant_moments, model)
        capture_moments(self.proportional_moments, model)

    def effective_moments(self, model):
        self.require(model)
        values = {}
        for factor, pattern in ((1., self.constant_moments), (self.line.parameter, self.proportional_moments)):
            for node, *moment in (() if pattern is None else pattern.rows):
                with np.errstate(over='ignore', invalid='ignore'):
                    values[node] = values.get(node, np.zeros(3)) + factor * np.array(moment)
                if not np.isfinite(values[node]).all():
                    raise ValueError('native spatial restart effective moment range')
        rows = tuple((node, *map(float, value)) for node, value in sorted(values.items()) if np.any(value))
        return SpatialNodalMoments(rows) if rows else None


def _moments(value, model):
    if value is None:
        return None
    _keys(value, ('rows',))
    if type(value['rows']) is not list or any(type(row) is not list for row in value['rows']):
        raise ValueError('native spatial restart moment rows')
    made = capture_moments(SpatialNodalMoments(tuple(tuple(row) for row in value['rows'])), model)
    if canonical(made) != canonical(value):
        raise ValueError('native spatial restart moment round trip')
    return made


def _point(value, model):
    _keys(value, ('line', 'constant_moments', 'proportional_moments'))
    made = LoadPoint(_line_point(value['line'], model), _moments(value['constant_moments'], model),
                     _moments(value['proportional_moments'], model))
    made.require(model)
    if canonical(made) != canonical(value):
        raise ValueError('native spatial restart load point round trip')
    return made


def _wire(snapshot, elements):
    _keys(snapshot, ('load_point', 'displacements', 'states'))
    if (type(snapshot['load_point']) is not LoadPoint or type(snapshot['states']) is not dict
            or set(snapshot['states']) != {i for i, _ in elements}):
        raise ValueError('complete typed native spatial restart snapshot')
    return dict(load_point=snapshot['load_point'], displacements=snapshot['displacements'],
                states=[dict(element_id=i, state=snapshot['states'][i]) for i, _ in elements])


def validate_chain(model, snapshots):
    started = monotonic()
    elements, n, free, fixed, identity = _model(model)
    if type(snapshots) is not tuple or not 1 <= len(snapshots) <= 65:
        raise ValueError('bounded complete native spatial restart chain')
    previous = None
    for index, snapshot in enumerate(snapshots):
        _wire(snapshot, elements)
        point = snapshot['load_point']
        point.require(model)
        pattern = point.line.effective(model)
        moments = point.effective_moments(model)
        u = np.asarray(snapshot['displacements'])
        if u.shape != (n,) or u.dtype != np.float64 or not np.isfinite(u).all() or np.any(u[list(fixed)]):
            raise ValueError('native spatial restart displacement/support mismatch')
        if index == 0 and (point.line.parameter != 0. or pattern.rows or moments is not None or np.any(u)):
            raise ValueError('native spatial restart stress-free genesis required')
        net = np.zeros(n)
        shared = {}
        for eid, element in elements:
            if monotonic() - started > 60.:
                raise RuntimeError('native spatial restart validation deadline')
            state = snapshot['states'][eid]
            mapping = list(element.get_dof_mapping(model.mesh))
            element._validate(model.mesh, state, u[mapping])
            if state['epoch'] != index or state['load_pattern'] != pattern:
                raise ValueError('native spatial restart epoch/effective line mismatch')
            net[mapping] += state['response'].residual
            for local, node in enumerate(element.node_ids):
                rotation = state['committed_nodal_rotation_matrices'][local]
                if node in shared and not np.array_equal(shared[node], rotation):
                    raise ValueError('native spatial restart shared rotation mismatch')
                shared[node] = rotation
            if previous is not None:
                old = previous['states'][eid]
                if (state['previous_state_sha256'] != old['state_sha256']
                        or canonical(state['origins']) != canonical(old['response'].history)):
                    raise ValueError('native spatial restart predecessor/history mismatch')
                if (not np.array_equal(state['seed_rotations'], old['response'].rotations)
                        or not np.array_equal(state['seed_resultants'], old['response'].resultants)):
                    raise ValueError('native spatial restart internal seed mismatch')
                increments = (state['committed_total_u'] - old['committed_total_u']).reshape(3, 6)[:, 3:]
                expected = np.array([rotation_exponential(d) @ r for d, r in zip(increments, old['committed_nodal_rotation_matrices'])])
                if not np.array_equal(expected, state['committed_nodal_rotation_matrices']):
                    raise ValueError('native spatial restart multiplicative rotation mismatch')
        external = nodal_force_vector(model, pattern)
        # Nodal work is counted once globally, including shared junction nodes.
        for node, *moment in (() if moments is None else moments.rows):
            dofs = list(model.mesh.dof_manager.get_node_dofs(node)[3:])
            net[dofs] -= moment
            external[dofs] += moment
        with np.errstate(over='ignore', invalid='ignore'):
            force_norm = float(np.linalg.norm(external))
            net_norm = float(np.linalg.norm(net[list(free)]))
        if not np.isfinite(external).all() or not np.isfinite(force_norm) or not np.isfinite(net_norm):
            raise ValueError('native spatial restart external/residual norm range')
        if net_norm / max(1., force_norm) > 1e-11:
            raise ValueError('native spatial restart accepted snapshot not equilibrated')
        previous = snapshot
    if _model(model)[-1] != identity:
        raise ValueError('native spatial restart model changed')
    return identity


def encode_checkpoint(model, snapshots):
    if type(snapshots) is not tuple or not 1 <= len(snapshots) <= 65:
        raise ValueError('bounded complete native spatial restart chain')
    elements = _model(model)[0]
    observed = canonical([_wire(s, elements) for s in snapshots])
    if len(observed) > MAX_BYTES:
        raise ValueError('native spatial restart byte limit')
    identity = validate_chain(model, snapshots)
    records = [_wire(s, elements) for s in snapshots]
    if canonical(records) != observed:
        raise ValueError('native spatial restart input changed during validation')
    body = dict(schema=SCHEMA, load_policy=POLICY, model_sha256=identity, snapshots=records,
                conservative_spectral_authority=False, production_qualified=False)
    raw = canonical({**body, 'checkpoint_sha256': sha(body)})
    if len(raw) > MAX_BYTES:
        raise ValueError('native spatial restart byte limit')
    return raw


def decode_checkpoint(model, raw, *, expected_sha256):
    if type(raw) is not bytes or type(expected_sha256) is not str or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('native spatial restart external SHA-256 mismatch')
    value = _load(raw.decode('ascii'))
    _keys(value, ('schema', 'load_policy', 'model_sha256', 'snapshots', 'conservative_spectral_authority',
                  'production_qualified', 'checkpoint_sha256'))
    elements, n, _, _, identity = _model(model)
    body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
    if (value['schema'] != SCHEMA or value['load_policy'] != POLICY or value['production_qualified'] is not False
            or value['conservative_spectral_authority'] is not False or value['model_sha256'] != identity
            or value['checkpoint_sha256'] != sha(body)):
        raise ValueError('native spatial restart model/schema/policy/hash mismatch')
    if type(value['snapshots']) is not list or not 1 <= len(value['snapshots']) <= 65:
        raise ValueError('native spatial restart snapshot count')
    snapshots = []
    for record in value['snapshots']:
        _keys(record, ('load_point', 'displacements', 'states'))
        rows = record['states']
        if type(rows) is not list or len(rows) != len(elements):
            raise ValueError('native spatial restart complete states')
        states = {}
        for row, (eid, element) in zip(rows, elements):
            _keys(row, ('element_id', 'state'))
            if type(row['element_id']) is not int or row['element_id'] != eid:
                raise ValueError('native spatial restart ordered element identity')
            states[eid] = _state(row['state'], element, model)
        snapshots.append(dict(load_point=_point(record['load_point'], model),
                              displacements=_array(record['displacements'], (n,)), states=states))
    snapshots = tuple(snapshots)
    if encode_checkpoint(model, snapshots) != raw:
        raise ValueError('native spatial restart canonical typed round trip')
    return snapshots
