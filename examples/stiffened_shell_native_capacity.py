"""SOURCE_ONLY_UNRUN: bounded experimental native BL5 panel investigation.

Execution requires separate numerical authority. This is a candidate physical
experiment, not an accepted capacity test, mesh study, or production workflow.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
from math import pi
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import SolveCancelled


@dataclass(frozen=True)
class Specimen:
    length_m: float = 2.
    width_m: float = .2
    skin_m: float = .002
    stiffener_width_m: float = .006
    stiffener_height_m: float = .04
    young_pa: float = 210e9
    poisson: float = .3
    density_kg_m3: float = 7850.
    fibre_yield_pa: float = 1e20
    generalized_yield_force_n: float = 1e16
    # Matching 8 shell segments and 8 native half-cells on each edge.
    nx: int = 8
    ny: int = 2

    def __post_init__(self):
        if self.nx != 8 or self.ny != 2:
            raise ValueError('this candidate freezes the matching 8x2 mesh')
        values = (self.length_m, self.width_m, self.skin_m,
                  self.stiffener_width_m, self.stiffener_height_m,
                  self.young_pa, self.density_kg_m3, self.fibre_yield_pa,
                  self.generalized_yield_force_n)
        if any(not np.isfinite(v) or v <= 0. for v in values):
            raise ValueError('positive finite SI specimen parameters required')
        if not np.isfinite(self.poisson) or not -1. < self.poisson < .5:
            raise ValueError('admitted isotropic Poisson ratio required')

    @property
    def eccentricity_m(self):
        return (self.skin_m + self.stiffener_height_m)/2.

    @property
    def beam_area_m2(self):
        return self.stiffener_width_m*self.stiffener_height_m

    def estimate(self):
        """Explanatory clamped-free composite Euler estimate, not an oracle."""
        e, a, z = self.young_pa, self.beam_area_m2, self.eccentricity_m
        ea_skin, ea_beam = e*self.width_m*self.skin_m, e*a
        ea = ea_skin + 2*ea_beam
        ei_skin = e*self.width_m*self.skin_m**3/12.
        ei_beam = e*self.stiffener_width_m*self.stiffener_height_m**3/12.
        first = 2*ea_beam*z
        ei = ei_skin + 2*(ei_beam+ea_beam*z*z) - first*first/ea
        return dict(ea_skin_n=ea_skin, ea_beam_n=ea_beam, ea_total_n=ea,
                    centroid_z_m=first/ea, ei_effective_nm2=ei,
                    euler_clamped_free_n=pi*pi*ei/(4*self.length_m**2))


def reference_nodes(specimen):
    s = specimen
    shell = {1+j*(s.nx+1)+i: (s.length_m*i/s.nx, s.width_m*j/s.ny, 0.)
             for j in range(s.ny+1) for i in range(s.nx+1)}
    beam = {101+100*line+i: (s.length_m*i/s.nx, y, s.eccentricity_m)
            for line, y in enumerate((0., s.width_m)) for i in range(s.nx+1)}
    return shell, beam


def definition(specimen=Specimen(), beams='mixed'):
    """Existing Q4/native material BL5; no point attachments or condensation."""
    if beams not in ('fibre', 'mixed'):
        raise ValueError('native material BL5 requires at least one fibre member')
    s = specimen
    e, g, a = s.young_pa, s.young_pa/(2*(1+s.poisson)), s.beam_area_m2
    iy = s.stiffener_width_m*s.stiffener_height_m**3/12.
    iz = s.stiffener_height_m*s.stiffener_width_m**3/12.
    # Explicit thin-rectangle torsional approximation and 5/6 shear area.
    j = s.stiffener_height_m*s.stiffener_width_m**3/3.
    stiffness = np.array((e*a, g*5*a/6, g*5*a/6, g*j, e*iy, e*iz))
    # q=sqrt(F.F + M.M/h^2) has force units. This independent resultant
    # comparator does not represent the fibres' yield stress or physical J2.
    resultant_metric = np.diag((1.,1.,1.,*(1./s.stiffener_height_m**2,)*3))
    generalized = b3_ge.EllipsoidalGeneralizedSection(
        np.diag(stiffness), resultant_metric, s.generalized_yield_force_n, 1e9)
    fibres = tuple(b3_ge.Fibre(str(i), y*np.sqrt(iz/a), z*np.sqrt(iy/a),
        a/4, e, b3_ge.FlowCurve((0.,), (s.fibre_yield_pa,), 1e9))
        for i, (y, z) in enumerate(((-1,-1), (-1,1), (1,-1), (1,1))))
    background = stiffness.copy(); background[[0,4,5]] = 0.
    physical = b3_ge.PhysicalFibreSection(fibres, np.diag(np.sqrt(background)))
    shell_nodes, beam_nodes = reference_nodes(s)
    members = []
    for line in range(2):
        for member in range(4):
            ids = tuple(101+100*line+2*member+i for i in range(3))
            section = generalized if beams == 'mixed' and member % 2 == 0 else physical
            inertia = s.density_kg_m3*np.diag((a,a,a,iy+iz,iy,iz))
            members.append(b3_ge.define_beam('b3-ge', 1+4*line+member, ids,
                np.array([beam_nodes[n] for n in ids]), np.tile(np.eye(3), (3,1,1)),
                section, inertia))
    cells = tuple((1+j*9+i, 2+j*9+i, 11+j*9+i, 10+j*9+i)
                  for j in range(2) for i in range(8))
    panel = ss.ShellPanel(tuple(shell_nodes), tuple(shell_nodes.values()), cells,
                         s.skin_m, e, s.poisson, s.density_kg_m3)
    lines = tuple(ss.ExperimentalBL5Line(tuple(1+18*line+i for i in range(9)),
        tuple((4*line+member, half) for member in range(4) for half in range(2)),
        expected_beam_arm=(0.,0.,-s.stiffener_height_m/2), basis='continuous_p1_coarse')
        for line in range(2))
    fixed = tuple(ss.FixedCoordinate('shell', n, axis)
                  for n in (1,10,19) for axis in range(6))
    return ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1,
        panel, tuple(members), (), fixed, lines,
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        bl5_material_policy=ss.BL5_NATIVE_FIBRE_MATERIAL_V1)


def load_patterns(specimen, compression_n, probe_n):
    """Dead compression through the reference composite centroid, tip z probe."""
    if any(not np.isfinite(v) or v <= 0. for v in (compression_n, probe_n)):
        raise ValueError('positive finite compression and probe magnitudes required')
    weights = specimen.estimate()
    skin = compression_n*weights['ea_skin_n']/weights['ea_total_n']
    beam = compression_n*weights['ea_beam_n']/weights['ea_total_n']
    # Skin end-face width quadrature: endpoint half shares, middle full share.
    preload = tuple(ss.NodalAction('shell', n, (-skin*w,0.,0.,0.,0.,0.))
                    for n, w in zip((9,18,27), (.25,.5,.25))) + tuple(
        ss.NodalAction('beam', n, (-beam,0.,0.,0.,0.,0.)) for n in (109,209))
    probe = tuple(ss.NodalAction('shell', n, (0.,0.,probe_n*w,0.,0.,0.))
                  for n, w in zip((9,18,27), (.25,.5,.25)))
    return preload, probe


def combine(*patterns):
    rows = {}
    for factor, actions in patterns:
        for a in actions:
            key = a.domain, a.node
            rows[key] = rows.get(key, np.zeros(6)) + factor*np.asarray(a.wrench)
    return tuple(ss.NodalAction(domain, node, tuple(values))
                 for (domain,node), values in rows.items())


def metric_for(order, specimen):
    """Half skin-area and half total beam-arclength nodal quadrature shares."""
    shell, beam = reference_nodes(specimen)
    measures = {}
    for n, (x,y,_) in shell.items():
        wx = .5 if x in (0., specimen.length_m) else 1.
        wy = .5 if y in (0., specimen.width_m) else 1.
        measures['shell',n] = .5*wx*wy/(specimen.nx*specimen.ny)
    for n, (x,_,_) in beam.items():
        wx = .5 if x in (0., specimen.length_m) else 1.
        measures['beam',n] = .5*wx/(2*specimen.nx)
    expected = set(measures)
    if len(order) != len(expected) or set(order) != expected:
        raise ValueError('complete unique compiled physical node order required')
    shell_count = len(shell)
    if any(domain != 'shell' for domain,_ in order[:shell_count]):
        raise ValueError('shell nodes must precede beam nodes')
    return ss.NativeDirectorChordMetric(np.array([measures[row] for row in order]),
                                       specimen.length_m, shell_count)


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if is_dataclass(value): return {f.name: _plain(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value, dict): return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    return value


def common_origin_wrench(positions, rows):
    """Only physical spatial force/moment rows may enter this aggregation."""
    positions, rows = np.asarray(positions), np.asarray(rows)
    if positions.shape != (len(rows),3) or rows.shape[1:] != (6,):
        raise ValueError('complete physical positions and six-component rows required')
    return np.r_[rows[:,:3].sum(axis=0),
                 (rows[:,3:] + np.cross(positions,rows[:,:3])).sum(axis=0)]


def declared_load_vector(actions, order, size):
    """Exact public nodal-dead-load mapping; retained rows remain zero."""
    index = {row:i for i,row in enumerate(order)}
    applied = np.zeros(size)
    for action in actions:
        i = index[action.domain,action.node]
        applied[6*i:6*i+6] += action.wrench
    return applied


def material_activity(result):
    """Exact nonzero public history/update flags; no elastic tolerance tuning."""
    fibre = _plain(result.beam_fibre_material)
    generalized = _plain(result.beam_generalized_material)
    physical_ids = {row['element_id'] for row in fibre}
    generalized_ids = {row['element_id'] for row in generalized}
    histories = result.beam_station_accumulated
    if len(histories) != 8 or physical_ids | generalized_ids != set(range(1,9)):
        raise ValueError('complete definition-ordered native material inventory required')
    physical_history = [dict(element_id=i,station_accumulated=list(values))
                        for i,values in enumerate(histories,1) if i in physical_ids]
    fibre_accumulated = any(v != 0. for row in physical_history for v in row['station_accumulated'])
    fibre_update = any(v != 0. for row in fibre for field in row['fibres']
                       for v in field['plastic_increment'])
    generalized_accumulated = any(v != 0. for row in generalized
                                 for station in row['stations'] for v in station['accumulated'])
    generalized_update = any(v != 0. for row in generalized for station in row['stations']
                             for key in ('plastic_increment','plastic_strain','plastic_strain_low')
                             for v in station[key])
    boundary = any(field['branch'] != 'ELASTIC' for row in fibre for field in row['fibres']) or any(
        station['branch'] != 'ELASTIC' for row in generalized for station in row['stations'])
    active = fibre_accumulated or fibre_update or generalized_accumulated or generalized_update
    return dict(physical_fibre_members=physical_history,
        fibre_history_accumulated=fibre_accumulated, fibre_update_nonzero=fibre_update,
        generalized_history_accumulated=generalized_accumulated,
        generalized_update_nonzero=generalized_update, nonelastic_branch_observed=boundary,
        any_material_history_or_update=active,
        elastic_fold_interpretation_eligible=not (active or boundary))


def observe(result, specimen, order, previous=None, *, seed_actions, arc_direction,
            arc_base_actions=None):
    """Copy all public state, plus physical load/reaction/work observations."""
    shell, _ = reference_nodes(specimen)
    shell_ids = [n for domain,n in order if domain == 'shell']
    root = [shell_ids.index(n) for n in (1,10,19)]
    displacement = np.asarray(result.shell_displacement).reshape(-1,6)
    if result.shell_chart_coordinates is None:
        # The ordinary unre-based force result omits explicit chart metadata.
        # Its public rotation coordinates use the identity-anchor exponential.
        if hasattr(result,'amplitude') or result.shell_rotation_anchors is not None:
            raise ValueError('unexpected incomplete public shell chart recovery')
        coordinates = displacement.copy()
        anchors = np.tile(np.eye(3),(len(shell),1,1))
        shell_frames = Rotation.from_rotvec(coordinates[:,3:]).as_matrix()
        chart_provenance = 'ordinary_unrebased_public_coordinates_implicit_identity_anchors'
    else:
        coordinates = np.asarray(result.shell_chart_coordinates).reshape(-1,6)
        anchors = np.asarray(result.shell_rotation_anchors)
        shell_frames = np.asarray(result.shell_frames)
        chart_provenance = 'actual_public_arc_shell_chart_and_frames'
    # Only these fixed, unre-based root rows are spatial reaction moments.
    if (not np.array_equal(coordinates[root], np.zeros((3,6))) or
        not np.array_equal(anchors[root], np.tile(np.eye(3),(3,1,1)))):
        raise ValueError('fixed identity root chart required for spatial root reactions')
    positions = np.array([shell[n] for n in shell_ids]) + displacement[:,:3]
    physical_positions = np.concatenate((positions, np.asarray(result.beam_positions)))
    count = len(order)
    reactions = np.asarray(result.support_reactions)
    seed_load = declared_load_vector(seed_actions,order,len(reactions))
    arc_base = (seed_load if arc_base_actions is None else
                declared_load_vector(arc_base_actions,order,len(reactions)))
    direction = declared_load_vector(arc_direction,order,len(reactions))
    if result.applied_dead_load is None:
        # Ordinary force recovery does not expose applied_dead_load. This exact
        # reconstruction uses the accepted public factor and owned action law.
        if hasattr(result,'amplitude'):
            raise ValueError('arc recovery must expose actual applied_dead_load')
        applied = result.load_factor*seed_load
        provenance = 'declared_seed_actions_times_accepted_public_load_factor'
    else:
        applied = np.asarray(result.applied_dead_load)
        if not np.array_equal(applied,arc_base+result.amplitude*direction):
            raise ValueError('actual arc load differs from fixed seed plus probe direction')
        provenance = 'actual_public_arc_applied_dead_load_verified_against_affine_law'
    if np.any(applied[6*count:] != 0.):
        raise ValueError('retained coordinates must not carry applied dead loads')
    nodal = applied[:6*count].reshape(count,6)
    if np.any(nodal[:,3:] != 0.):
        raise ValueError('this experiment admits dead spatial translation forces only')
    root_rows = reactions[:6*len(shell)].reshape(len(shell),6)[root]
    load_wrench = common_origin_wrench(physical_positions,nodal)
    reaction_wrench = common_origin_wrench(positions[root],root_rows)
    work = None
    if previous is not None:
        old_positions = np.asarray(previous['physical_positions_m'])
        old_forces = np.asarray(previous['physical_applied_rows'])[:,:3]
        work = float(np.sum(.5*(old_forces+nodal[:,:3])*(physical_positions-old_positions)))
    state = _plain(result)
    return dict(generation=result.generation, amplitude=getattr(result,'amplitude',None),
        material_activity=material_activity(result),
        tip_z_m=float(displacement[shell_ids.index(18),2]),
        physical_positions_m=physical_positions.tolist(), physical_applied_rows=nodal.tolist(),
        applied_load_provenance=provenance,
        shell_chart_provenance=chart_provenance, shell_physical_frames=shell_frames.tolist(),
        shell_chart_coordinates=coordinates.tolist(), shell_rotation_anchors=anchors.tolist(),
        applied_common_origin_wrench=load_wrench.tolist(),
        root_common_origin_reaction_wrench=reaction_wrench.tolist(),
        global_wrench_residual=(load_wrench+reaction_wrench).tolist(),
        external_work_trapezoid_j=work, external_work_is_endpoint_quadrature=True,
        state=state)


def turning_observations(observations):
    """Sign reversal in load amplitude is a bracket candidate, never capacity."""
    arc = []
    for observation in observations:
        amplitude = observation['amplitude']
        if amplitude is None:
            diagnostic = observation.get('affine_origin_diagnosis',{})
            state = observation.get('state',{})
            if (diagnostic.get('provenance') != 'completed_factor_one_declared_seed' or
                diagnostic.get('amplitude') != 0. or state.get('load_factor') != 1. or
                state.get('status') != 'completed'):
                continue
            amplitude = diagnostic['amplitude']
        arc.append((observation,amplitude))
    candidates = []
    for a,b,c in zip(arc,arc[1:],arc[2:]):
        left, right = b[1]-a[1], c[1]-b[1]
        if left*right < 0.:
            candidates.append(dict(generations=[o[0]['generation'] for o in (a,b,c)],
                                   kind='amplitude_direction_change_unvalidated'))
    return candidates


def _record_observation(observations, observation, observer):
    """Retain an owned witness before an optional independently owned delivery."""
    observations.append(deepcopy(observation))
    if observer is not None:
        observer(deepcopy(observations[-1]))


def _bind_seed_origin(observation, phase):
    """Tag only the committed full seed, before retained/stream copies split."""
    state = observation['state']
    if (phase == 'seed' and observation['amplitude'] is None and
        state.get('status') == 'completed' and state.get('load_factor') == 1. and
        observation['applied_load_provenance'] ==
            'declared_seed_actions_times_accepted_public_load_factor'):
        observation['affine_origin_diagnosis'] = dict(
            amplitude=0.,provenance='completed_factor_one_declared_seed')


def initial_elastic_screen(observations):
    return dict(any_material_history_or_update=any(
        o['material_activity']['any_material_history_or_update'] for o in observations),
        elastic_fold_interpretation_eligible=bool(observations) and all(
            o['material_activity']['elastic_fold_interpretation_eligible'] for o in observations),
        capacity_or_stability_proved=False)


def workflow(specimen=Specimen(), *, compression_n=None, probe_n=None,
             arc_steps=24, radius=.01, cancellation_token=None, on_accepted=None,
             on_observation=None):
    """One frozen mesh/load/control route; no automatic retries or sweeps."""
    if type(arc_steps) is not int or not 2 <= arc_steps <= 64:
        raise ValueError('bounded 2..64 accepted arc steps required')
    if on_accepted is not None and not callable(on_accepted):
        raise ValueError('accepted observer must be callable')
    if on_observation is not None and not callable(on_observation):
        raise ValueError('physical observation observer must be callable')
    estimate = specimen.estimate()
    compression = 1.1*estimate['euler_clamped_free_n'] if compression_n is None else compression_n
    probe = .02*estimate['euler_clamped_free_n'] if probe_n is None else probe_n
    preload, positive = load_patterns(specimen,compression,probe)
    seed_actions = combine((1.,preload),(1.,positive))
    direction = combine((-1.,positive))
    report = dict(status='source_candidate', production_qualified=False, capacity_claim=False,
        specimen=asdict(specimen), explanatory_estimate=estimate,
        seed_actions=_plain(seed_actions), arc_direction=_plain(direction),
        physical_load_law='F(alpha)=F_compression+(1-alpha)*F_positive_probe',
        geometric_imperfection=False, observations=[], checkpoint_inventory=[],
        restart_checks=[], turning_candidates=[], failure=None)
    phase = 'construction'
    with ExitStack() as stack:
        try:
            model = definition(specimen)
            owner = ss.create_analysis(definition=model); stack.callback(owner.close)
            phase = 'preflight'
            preflight = owner.preflight(); report['preflight'] = _plain(preflight)
            order = tuple(preflight['arc_metric_node_order'])
            metric = metric_for(order,specimen)
            control = ss.ArcLengthControl('shell',18,2,metric,radius,orientation=-1)
            report['arc_control'] = _plain(control)
            phase = 'seed'
            def record(result):
                nonlocal phase
                accepted_phase = phase
                observations = report['observations']
                previous = observations[-1] if observations else None
                observation = observe(result,specimen,order,previous,
                    seed_actions=seed_actions,arc_direction=direction)
                _bind_seed_origin(observation,accepted_phase)
                phase = 'observation_delivery'
                _record_observation(observations,observation,on_observation)
                if accepted_phase == 'arc' and on_accepted is not None:
                    phase = 'accepted_callback'
                    on_accepted(result)
                phase = accepted_phase
            seed = owner.solve(seed_actions,(.1,.25,.5,.75,1.),
                               cancellation_token=cancellation_token,on_accepted=record)
            if seed.status != 'completed':
                report['status'] = 'seed_failed'; report['seed_status'] = seed.status
                raw = owner.checkpoint()
                report['checkpoint_inventory'].append(dict(generation=seed.generation,
                    sha256=sha256(raw).hexdigest(),size_bytes=len(raw),checkpoint_bytes=raw))
                report['initial_elastic_screen'] = initial_elastic_screen(report['observations'])
                return report
            if seed.load_factor != 1. or report['observations'][-1]['generation'] != seed.generation:
                raise ValueError('completed full-force seed must bind the affine origin')
            # Validate the already retained/delivered diagnostic binding. The
            # committed full seed is authoritative even if a later writer fails.
            if report['observations'][-1].get('affine_origin_diagnosis') != dict(
                amplitude=0.,provenance='completed_factor_one_declared_seed'):
                raise ValueError('committed full-force seed lacks affine-origin diagnosis')
            phase = 'arc'
            for step in range(arc_steps):
                owner.continue_arc_length(direction,control,steps=1,
                    cancellation_token=cancellation_token,on_accepted=record)
                if step == 0:
                    phase = 'checkpoint_restore'
                    raw = owner.checkpoint()
                    report['checkpoint_inventory'].append(dict(generation=owner.recover().generation,
                        sha256=sha256(raw).hexdigest(), size_bytes=len(raw),checkpoint_bytes=raw))
                    fresh = ss.create_analysis(definition=model); stack.callback(fresh.close)
                    restored = fresh.restore(raw)
                    if restored.status != 'accepted_prefix' or fresh.checkpoint() != raw:
                        raise RuntimeError('fresh-owner prefix restore differs')
                    if _plain(fresh.recover()) != _plain(owner.recover()):
                        raise RuntimeError('fresh-owner prefix public recovery differs')
                    report['restart_checks'].append(dict(prefix_bytes_equal=True,prefix_recovery_equal=True))
                    phase = 'arc'
                elif step == 1:
                    phase = 'restart_continue'
                    fresh.continue_arc_length(direction,control,steps=1,
                                               cancellation_token=cancellation_token)
                    if fresh.checkpoint() != owner.checkpoint():
                        raise RuntimeError('fresh-owner continued endpoint differs')
                    if _plain(fresh.recover()) != _plain(owner.recover()):
                        raise RuntimeError('fresh-owner continued public recovery differs')
                    report['restart_checks'][-1]['continued_bytes_equal'] = True
                    report['restart_checks'][-1]['continued_recovery_equal'] = True
                    phase = 'arc'
            raw = owner.checkpoint()
            report['checkpoint_inventory'].append(dict(generation=owner.recover().generation,
                sha256=sha256(raw).hexdigest(), size_bytes=len(raw),checkpoint_bytes=raw))
            report['turning_candidates'] = turning_observations(report['observations'])
            report['status'] = ('turning_candidate_unvalidated' if report['turning_candidates']
                                else 'bounded_no_turning_observed')
        except (ss.FiniteIncrementFailure, ss.ArcSelectionFailure, SolveCancelled) as exc:
            if phase in ('observation_delivery','accepted_callback'):
                raise
            report['status'] = 'cancelled' if isinstance(exc,SolveCancelled) else 'typed_failure'
            report['failure'] = dict(phase=phase, type=type(exc).__name__, message=str(exc))
            for name in ('attempt','attempts','reason','stage'):
                if hasattr(exc,name): report['failure'][name] = _plain(getattr(exc,name))
            # Capture the actual committed prefix; never manufacture an endpoint.
            if report['observations']:
                recovered = owner.recover()
                if report['observations'][-1]['generation'] != recovered.generation:
                    observation = observe(recovered,specimen,order,
                        report['observations'][-1],seed_actions=seed_actions,arc_direction=direction)
                    _record_observation(report['observations'],observation,on_observation)
                raw = owner.checkpoint()
                report['checkpoint_inventory'].append(dict(generation=recovered.generation,
                    sha256=sha256(raw).hexdigest(),size_bytes=len(raw),checkpoint_bytes=raw))
            report['turning_candidates'] = turning_observations(report['observations'])
    report['initial_elastic_screen'] = initial_elastic_screen(report['observations'])
    return report


VERIFIED_FORCE_PREFIX_SHA256 = '4e7adf0619e8708102e564949922d0ab73fcc41fc75ddd5699b5f62ea64925ec'
VERIFIED_FORCE_PREFIX_BYTES = 27627
VERIFIED_FORCE_PREFIX_MODEL = 'ec405c7b2085629ba1e4177a5d517e1ec630dd25db61942c11bfa1b3142b0c43'
# Exact recorded actions from installed-v1/screen-result.json, not a force law
# inferred from a checksum or common-origin wrench alone.
VERIFIED_FORCE_PREFIX_ACTIONS = (
    ss.NodalAction('shell',9,(-2596.463001005903,0.,103.85852004023613,0.,0.,0.)),
    ss.NodalAction('shell',18,(-5192.926002011806,0.,207.71704008047226,0.,0.,0.)),
    ss.NodalAction('shell',27,(-2596.463001005903,0.,103.85852004023613,0.,0.,0.)),
    ss.NodalAction('beam',109,(-6231.511202414168,0.,0.,0.,0.,0.)),
    ss.NodalAction('beam',209,(-6231.511202414168,0.,0.,0.,0.,0.)))


def read_verified_force_prefix(source):
    """Only the retained, individually verified ordinary-force prefix is bound."""
    if type(source) is bytes:
        raw = source
    elif isinstance(source,(str,Path)):
        raw = Path(source).read_bytes()  # Path/open errors intentionally propagate.
    else:
        raise TypeError('checkpoint bytes or an explicit checkpoint path required')
    if len(raw) != VERIFIED_FORCE_PREFIX_BYTES or sha256(raw).hexdigest() != VERIFIED_FORCE_PREFIX_SHA256:
        raise ValueError('checkpoint differs from the verified generation-three force prefix')
    return raw


def validate_proportional_origin(result, preflight):
    """Bind actual restored public state; checksums alone do not admit mechanics."""
    if (result.status != 'completed' or result.generation != 3 or
        not np.isfinite(result.load_factor) or result.load_factor != .5 or
        getattr(result,'amplitude',None) is not None or
        tuple(result.accepted_factors) != (.1,.25,.5) or
        any(getattr(result,name,None) is not None for name in (
            'load_programme_identity','body_load','pressure_load','distributed_load')) or
        preflight['identity'] != VERIFIED_FORCE_PREFIX_MODEL):
        raise ValueError('verified ordinary-force origin/model/generation/factor required')
    return float(result.load_factor)


def proportional_turning_observations(observations):
    """Actual proportional-factor sign-change brackets, including restored origin."""
    candidates = []
    for a,b,c in zip(observations,observations[1:],observations[2:]):
        left = b['proportional_factor']-a['proportional_factor']
        right = c['proportional_factor']-b['proportional_factor']
        if left*right < 0.:
            candidates.append(dict(generations=[o['generation'] for o in (a,b,c)],
                kind='proportional_factor_direction_change_unvalidated'))
    return candidates


def _checkpointed_packet(owner, result, observation):
    """Read the committed public endpoint inside its post-publication observer."""
    recovered = owner.recover()
    if recovered.generation != result.generation or _plain(recovered) != _plain(result):
        raise RuntimeError('accepted callback differs from committed public recovery')
    raw = owner.checkpoint()
    if type(raw) is not bytes:
        raise TypeError('public checkpoint must return owned immutable bytes')
    return dict(observation=deepcopy(observation), checkpoint_bytes=raw,
        checkpoint_sha256=sha256(raw).hexdigest(),checkpoint_size_bytes=len(raw),
        generation=result.generation, accepted_status=recovered.status)


def _retain_checkpointed_packet(report, packet, observer):
    """No borrowed public arrays or user-mutated callback packet is retained."""
    report['observations'].append(deepcopy(packet['observation']))
    report['checkpoint_inventory'].append(dict(generation=packet['generation'],
        sha256=packet['checkpoint_sha256'],size_bytes=packet['checkpoint_size_bytes'],
        checkpoint_bytes=packet['checkpoint_bytes'],accepted_status=packet['accepted_status']))
    if observer is not None:
        observer(deepcopy(packet))


def proportional_workflow(prefix_checkpoint, *, arc_steps=24, cancellation_token=None,
                          on_state=None, on_accepted=None):
    """SOURCE_ONLY_UNRUN explicit new route: F=(restored_factor+alpha)*F_full.

    Restores the named verified prefix. It never retries the historical failed
    force seed. on_state receives complete owned observation and checkpoint bytes
    for the restored origin and every subsequent committed accepted arc endpoint.
    It may stream durable evidence under separately granted output authority.
    """
    if type(arc_steps) is not int or not 1 <= arc_steps <= 64:
        raise ValueError('bounded 1..64 accepted proportional arc steps required')
    if any(observer is not None and not callable(observer) for observer in (on_state,on_accepted)):
        raise ValueError('callable checkpoint/accepted observers required')
    raw = read_verified_force_prefix(prefix_checkpoint)
    specimen = Specimen()
    estimate = specimen.estimate()
    preload, probe = load_patterns(specimen,1.1*estimate['euler_clamped_free_n'],
                                   .02*estimate['euler_clamped_free_n'])
    full_actions = combine((1.,preload),(1.,probe))
    if _plain(full_actions) != _plain(VERIFIED_FORCE_PREFIX_ACTIONS):
        raise ValueError('full actions differ from the retained verified seed source record')
    report = dict(status='source_candidate',production_qualified=False,capacity_claim=False,
        route='proportional_full_load_from_verified_force_prefix',specimen=asdict(specimen),
        full_actions=_plain(full_actions), physical_load_law='F(alpha)=(restored_factor+alpha)*F_full',
        observations=[],checkpoint_inventory=[],turning_candidates=[],failure=None,
        requested_arc_steps=arc_steps,accepted_arc_steps=0,
        input_prefix=dict(sha256=sha256(raw).hexdigest(),size_bytes=len(raw)))
    with ExitStack() as stack:
        # Construction/restore/checkpoint/model and writer errors propagate.
        owner = ss.create_analysis(definition=definition(specimen)); stack.callback(owner.close)
        preflight = owner.preflight(); report['preflight'] = _plain(preflight)
        origin = owner.restore(raw)
        base_factor = validate_proportional_origin(origin,preflight)
        if owner.checkpoint() != raw:
            raise RuntimeError('restored ordinary-force prefix checkpoint differs')
        order = tuple(preflight['arc_metric_node_order'])
        base_actions = combine((base_factor,full_actions))
        report['origin_factor'] = base_factor
        report['origin_generation'] = origin.generation
        report['base_actions'] = _plain(base_actions)
        control = ss.ArcLengthControl('shell',18,2,metric_for(order,specimen),.01,orientation=1)
        report['arc_control'] = _plain(control)
        phase = 'origin_observation'
        def record(result, *, restored_origin=False):
            nonlocal phase
            phase = 'observation_capture'
            previous = report['observations'][-1] if report['observations'] else None
            observation = observe(result,specimen,order,previous,seed_actions=full_actions,
                                  arc_direction=full_actions,arc_base_actions=base_actions)
            observation['proportional_factor'] = (base_factor if restored_origin else
                                                  base_factor+result.amplitude)
            observation['proportional_factor_provenance'] = (
                'actual_restored_public_force_factor' if restored_origin else
                'actual_restored_factor_plus_public_arc_amplitude')
            if restored_origin:
                observation['affine_origin_diagnosis'] = dict(amplitude=0.,
                    provenance='verified_restored_ordinary_force_prefix')
            phase = 'checkpoint_capture'
            packet = _checkpointed_packet(owner,result,observation)
            phase = 'checkpoint_writer'
            _retain_checkpointed_packet(report,packet,on_state)
            if not restored_origin:
                report['accepted_arc_steps'] += 1
                if on_accepted is not None:
                    phase = 'accepted_callback'; on_accepted(deepcopy(result))
            phase = 'arc'
        record(origin,restored_origin=True)
        try:
            for _ in range(arc_steps):
                before = report['accepted_arc_steps']
                returned = owner.continue_arc_length(full_actions,control,steps=1,
                    cancellation_token=cancellation_token,on_accepted=record)
                if returned.status != 'completed':
                    report['status'] = 'incomplete_arc_return'
                    report['returned_status'] = returned.status
                    break
                if report['accepted_arc_steps'] != before+1:
                    raise RuntimeError('completed public arc call did not deliver one accepted endpoint')
            else:
                report['status'] = 'bounded_completed'
        except (ss.ArcSelectionFailure,ss.FiniteIncrementFailure,SolveCancelled) as exc:
            if phase in ('observation_capture','checkpoint_capture','checkpoint_writer','accepted_callback'):
                raise
            report['status'] = 'cancelled' if isinstance(exc,SolveCancelled) else 'typed_failure'
            report['failure'] = dict(phase=phase,type=type(exc).__name__,message=str(exc))
            for name in ('attempt','attempts','reason','stage'):
                if hasattr(exc,name): report['failure'][name] = _plain(getattr(exc,name))
        recovered = owner.recover()
        report['recovered_status'] = recovered.status
        report['recovered_generation'] = recovered.generation
        if report['observations'][-1]['generation'] != recovered.generation:
            raise RuntimeError('committed endpoint was not delivered by public accepted observer')
        if owner.checkpoint() != report['checkpoint_inventory'][-1]['checkpoint_bytes']:
            raise RuntimeError('retained checkpoint differs from actual committed endpoint')
        report['turning_candidates'] = proportional_turning_observations(report['observations'])
        if report['status'] == 'bounded_completed':
            report['status'] = ('turning_candidate_unvalidated' if report['turning_candidates']
                                else 'bounded_no_turning_observed')
        report['initial_elastic_screen'] = initial_elastic_screen(report['observations'])
    return report


if __name__ == '__main__':
    raise SystemExit('SOURCE_ONLY_UNRUN: obtain separate numerical execution authority first')
