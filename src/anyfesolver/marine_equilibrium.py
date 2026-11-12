"""Fixed-fairlead mooring statics using the public line law (development).

No dynamic relaxation, radiation history or body equilibrium is implied. Static
residual is minus applied nodal force; public K is its position derivative.
Only accepted stages are exposed on cancellation/failure. Original model and
states are immutable. Continuous catenary is an initializer/reference geometry,
not a replacement for the discrete line law or W6 qualification.
"""
from dataclasses import dataclass, replace
import math
import warnings
from types import MappingProxyType
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve, lsmr, MatrixRankWarning
from .marine_mooring import (MooringLineAssembler, MooringNodeState,
    FixedEndpoint, RigidBodyAttachmentEndpoint, DynamicLineNodeEndpoint,
    MooringLineError, _array)
from .marine_rigid import RigidBodyState

API_VERSION = 'marine_fixed_fairlead_equilibrium_v1'
__all__ = ['API_VERSION','MooringEquilibriumError','ElasticCatenary',
           'elastic_catenary','initialize_mooring_catenaries','MooringStaticSpec',
           'MooringStaticStage','MooringStaticResult','solve_mooring_equilibrium']


class MooringEquilibriumError(ValueError):
    """Malformed or unsupported static intent, not a convergence result."""


def _real(value, name, *, positive=False, nonnegative=False):
    try:
        result = float(_array([value], (1,), name, positive, nonnegative)[0])
    except MooringLineError as error:
        raise MooringEquilibriumError(str(error)) from error
    return result


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise MooringEquilibriumError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class ElasticCatenary:
    positions: np.ndarray
    tensions: np.ndarray
    first_force: np.ndarray
    last_force: np.ndarray
    endpoint_error: float

    def __post_init__(self):
        n = len(self.positions)
        for name, shape in (('positions', (n,3)), ('tensions',(n,)),
                            ('first_force',(3,)), ('last_force',(3,))):
            object.__setattr__(self, name, _array(getattr(self,name),shape,name))


def _terminals(first, last):
    first = _array(first,(3,), 'first terminal')
    last = _array(last,(3,), 'last terminal')
    delta = last-first
    horizontal = float(np.linalg.norm(delta[:2]))
    if horizontal <= 1e-12*max(1.,float(np.linalg.norm(delta))):
        raise MooringEquilibriumError('vertical/coincident catenary terminals require an explicit node guess')
    direction = np.array([*delta[:2]/horizontal,0.])
    return first, last, horizontal, direction


def _shoot(evaluate, horizontal, height, length, ea, weight, max_evaluations):
    """Two force unknowns, no stiffness/compression regularization."""
    scale = max(abs(weight)*length, ea*max(math.hypot(horizontal,height)/length-1.,0.),1.)
    h0 = max(.1*scale, ea*max(horizontal/length-1.,0.))
    v0 = scale*height/max(horizontal, length*.1)-.5*weight*length
    def values(x):
        h = scale*math.exp(x[0]); v = scale*x[1]
        geometry,jac = evaluate(h,v)
        return (geometry-np.array([horizontal,height]))/length, jac@np.diag([h,scale])/length
    fit = least_squares(lambda x:values(x)[0], [math.log(h0/scale),v0/scale],
                        jac=lambda x:values(x)[1], bounds=([-40.,-np.inf],[40.,np.inf]),
                        max_nfev=max_evaluations, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    h,v = scale*math.exp(fit.x[0]),scale*fit.x[1]
    error = float(np.linalg.norm(values(fit.x)[0],ord=np.inf)*length)
    if not fit.success or not math.isfinite(error) or error > 1e-9*max(1.,length):
        raise MooringEquilibriumError(f'catenary shooting failed: endpoint error {error:g} m')
    return h,v,error


def elastic_catenary(first, last, rest_length, axial_rigidity, submerged_weight,
                     material_coordinates, *, max_evaluations=200):
    """Homogeneous suspended elastic cable; weight N/m of reference length.

    Positive submerged_weight acts downward (z-up); negative is buoyant. Nodes
    sample reference material arclength, not stretched arclength. Both terminal
    forces act ON their supports; their sum equals downward distributed weight.
    Horizontal separation is required; no floor/wetting is inferred here.
    """
    first,last,x,direction = _terminals(first,last)
    length = _real(rest_length,'rest length',positive=True)
    ea = _real(axial_rigidity,'axial rigidity',positive=True)
    w = _real(submerged_weight,'submerged weight')
    _integer(max_evaluations,'max evaluations')
    shape = np.shape(material_coordinates)
    if len(shape)!=1 or shape[0]<2:
        raise MooringEquilibriumError('at least two material coordinates required')
    s = _array(material_coordinates,shape,'material coordinates')
    if s[0]!=0. or s[-1]!=length or np.any(np.diff(s)<=0.):
        raise MooringEquilibriumError('material coordinates must increase from zero to rest length')
    height = float(last[2]-first[2])
    if w == 0.:
        distance = float(np.linalg.norm(last-first))
        if distance <= length:
            raise MooringEquilibriumError('weightless slack cable has no unique catenary')
        tension = ea*(distance/length-1.)
        force = tension*(last-first)/distance
        return ElasticCatenary(first+s[:,None]/length*(last-first),np.full(len(s),tension),force,-force,0.)
    def geometry(h,v,at):
        a = v; d = w*at; b = v+d
        ta = np.hypot(h,a); tb = np.hypot(h,b)
        # Rationalize both differences. Using d=w*s (rather than b-a) retains
        # the exact weightless limit when b rounds to a. Q*d is sinh of the
        # asinh difference; positive ab uses a second rationalization.
        ab = a*b
        same = ab>=0.
        q = np.empty_like(np.asarray(b,dtype=float))
        numerator = np.ones_like(q)
        np.divide(h*h+a*a+b*b,ta*tb+ab,out=numerator,where=same)
        np.divide(1.+numerator,ta+tb,out=q,where=same)
        np.divide((h*h+ta*tb-ab)/(h*h),ta+tb,out=q,where=~same)
        z = d*q
        ratio = np.ones_like(z)
        np.divide(np.arcsinh(z),z,out=ratio,where=z!=0.)
        integral = at*q*ratio
        dx = h*integral+h*at/ea
        dz = at*(2.*v+d)/(ta+tb)+(v*at+.5*w*at**2)/ea
        return dx,dz,tb,integral,q
    def evaluate(h,v):
        dx,dz,tb,integral,q = geometry(h,v,length); ta = math.hypot(h,v)
        vertical = length*q*h*h/(ta*tb)
        cross = -h*length*(2.*v+w*length)/((ta+tb)*ta*tb)
        jac = np.array([[integral-vertical+length/ea,cross],
                        [cross,vertical+length/ea]])
        return np.array([dx,dz]),jac
    h,v,error = _shoot(evaluate,x,height,length,ea,w,max_evaluations)
    dx,dz,tensions,_,_ = geometry(h,v,s)
    positions = first+dx[:,None]*direction
    positions[:,2] += dz
    return ElasticCatenary(positions,tensions,h*direction+[0.,0.,v],
                          -h*direction-[0.,0.,v+w*length],error)


def _fixed_states(model, states):
    if type(model) is not MooringLineAssembler:
        raise MooringEquilibriumError('public MooringLineAssembler required')
    try:states = dict(states)
    except (TypeError,ValueError) as error:
        raise MooringEquilibriumError('named fixed body states required') from error
    if set(states)!={b.body_id for b in model.bodies}:
        raise MooringEquilibriumError('exact fixed body state inventory required')
    for state in states.values():
        if type(state) is not RigidBodyState:
            raise MooringEquilibriumError('public RigidBodyState required')
        if np.any(state.velocity) or np.any(state.acceleration):
            raise MooringEquilibriumError('fixed fairleads require zero velocity and acceleration')
    if tuple(model.gravity_world[:2]) != (0.,0.) or model.gravity_world[2]>0.:
        raise MooringEquilibriumError('catenary statics require downward z-up gravity')
    if model.environment is not None and any(c.amplitude!=0. for c in model.environment.components):
        raise MooringEquilibriumError('active waves are not static loads; omit waves from equilibrium intent')
    return states


def initialize_mooring_catenaries(model, body_states, *, max_evaluations=200):
    """Exact discrete suspended-chain gravity initialization at fixed fairleads.

    Homogeneous EA, mass and displaced area per line are required. Actual
    segment reference lengths may differ. Nodal half-segment weight and the
    public tension-only law determine the polygon; no continuum curve is
    silently treated as exact discrete equilibrium. Contact/current are solved
    by subsequent stages. Emerged/invalid guesses are refused by the assembler.
    """
    states = _fixed_states(model,body_states)
    _integer(max_evaluations,'max evaluations')
    bodies = {b.body_id:b for b in model.bodies}
    def terminal(ep):
        if type(ep) is FixedEndpoint:return np.array(ep.position)
        return bodies[ep.body_id].point_kinematics(states[ep.body_id],ep.point_body).position
    nodes = np.zeros((model.node_count,3))
    for line in model.lines:
        for name in ('axial_rigidity','mass_per_length','displaced_area'):
            a = getattr(line,name)
            if not np.all(a==a[0]):
                raise MooringEquilibriumError('heterogeneous lines require an explicit node guess')
        first,last,x,direction = _terminals(terminal(line.endpoints[0]),terminal(line.endpoints[-1]))
        lengths = line.rest_lengths; length = float(sum(lengths)); ea = float(line.axial_rigidity[0])
        w = float((line.mass_per_length[0]-model.fluid_density*line.displaced_area[0])*(-model.gravity_world[2]))
        mid = np.cumsum(lengths)-.5*lengths
        if w==0.:
            curve = elastic_catenary(first,last,length,ea,w,np.r_[0.,np.cumsum(lengths)],max_evaluations=max_evaluations)
            positions = curve.positions
        else:
            def evaluate(h,v):
                vertical = v+w*mid; t = np.hypot(h,vertical)
                dx = lengths*(h/t+h/ea); dz = lengths*(vertical/t+vertical/ea)
                cross = -np.sum(lengths*h*vertical/t**3)
                jac = np.array([[np.sum(lengths*(vertical**2/t**3+1./ea)),cross],
                                [cross,np.sum(lengths*(h**2/t**3+1./ea))]])
                return np.array([sum(dx),sum(dz)]),jac
            h,v,_ = _shoot(evaluate,x,float(last[2]-first[2]),length,ea,w,max_evaluations)
            vertical = v+w*mid; t = np.hypot(h,vertical)
            increments = (lengths*(h/t+h/ea))[:,None]*direction
            increments[:,2] += lengths*(vertical/t+vertical/ea)
            positions = first+np.vstack([np.zeros(3),np.cumsum(increments,axis=0)])
        for ep,p in zip(line.endpoints[1:-1],positions[1:-1]):
            nodes[ep.node_index] = p
    result = MooringNodeState(nodes,np.zeros_like(nodes),np.zeros_like(nodes))
    # Full physics may refuse an initializer below the environmental floor.
    model.force(states,result)
    return result


@dataclass(frozen=True)
class MooringStaticSpec:
    force_scale: float = 1.                 # explicit N normalization
    force_atol: float = 1e-8                # N
    force_rtol: float = 1e-8
    load_fractions: tuple = (.25,.5,.75,1.)
    max_iterations: int = 40
    max_line_search: int = 20

    def __post_init__(self):
        for name in ('force_scale','force_atol','force_rtol'):
            object.__setattr__(self,name,_real(getattr(self,name),name,positive=name=='force_scale',nonnegative=name!='force_scale'))
        if not math.isfinite(self.force_limit) or self.force_limit<=0.:
            raise MooringEquilibriumError('positive finite derived force tolerance required')
        f = tuple(_real(x,'load fraction',positive=True) for x in self.load_fractions)
        if not f or f[-1]!=1. or any(b<=a for a,b in zip(f,f[1:])):
            raise MooringEquilibriumError('load fractions must increase to exactly one')
        object.__setattr__(self,'load_fractions',f)
        for name in ('max_iterations','max_line_search'):_integer(getattr(self,name),name)

    @property
    def force_limit(self):return self.force_atol+self.force_rtol*self.force_scale


@dataclass(frozen=True)
class MooringStaticStage:
    name: str
    fraction: float
    nodes: MooringNodeState
    residual_norm: float
    normalized_residual: float
    iterations: int
    least_norm_steps: int = 0


@dataclass(frozen=True)
class MooringStaticResult:
    status: str
    stages: tuple
    message: str
    audit: object = None

    @property
    def nodes(self):return self.stages[-1].nodes if self.stages else None


def _stage_model(model, contact, current):
    lines = model.lines if contact else tuple(replace(l,contact_diameters=None) for l in model.lines)
    env = model.environment
    if env is not None:
        profile = replace(env.current,velocities=tuple(tuple(current*np.array(v)) for v in env.current.velocities))
        env = replace(env,current=profile)
    return replace(model,lines=lines,seabed_contact=model.seabed_contact if contact else None,environment=env)


def _contact_potential(nodes, out, offset, weight, reference_positions):
    """Conservative potential with a fixed, local gravitational datum.

    Summation resolution depends on the absolute contributing terms, never the
    potentially cancelling total. The arbitrary WORLD origin does not contribute
    a large constant to the merit. This changes neither force nor its tangent.
    """
    loads = out.force_components['weight'][offset:].reshape(-1,3)
    with np.errstate(over='ignore',invalid='ignore'):
        work = -weight*loads*(nodes.positions-reference_positions)
    terms = [out.elastic_energy,out.contact_diagnostics['potential_energy'],*work.ravel()]
    try:
        return math.fsum(terms), math.fsum(abs(float(x)) for x in terms)
    except (OverflowError,ValueError):
        # An unrepresentable merit is not a reason to change a finite force
        # criterion. The caller uses the original residual globalization.
        return float('nan'),float('inf')


def solve_mooring_equilibrium(model, body_states, initial_nodes=None, spec=None, *, cancel=None):
    """Ramped fixed-fairlead statics; full successful audit is required.

    Gravity/buoyancy ramp together as net submerged weight. Then steady current
    ramps with full weight, then contact activates with both at full load.
    Conservative contact without an environment uses total-potential Armijo
    globalization. Near its floating-point energy resolution, residual decrease
    takes over. Current-loaded cases retain residual globalization; no potential
    is assigned to nonconservative drag. Final residual criteria are unchanged.
    Bodies remain fixed and their reactions are retained for a later coupled
    solve. Failure/cancellation returns only completed stages, never a trial.
    """
    states = _fixed_states(model,body_states)
    spec = MooringStaticSpec() if spec is None else spec
    if type(spec) is not MooringStaticSpec:raise MooringEquilibriumError('named static spec required')
    if initial_nodes is not None and (type(initial_nodes) is not MooringNodeState or
            initial_nodes.positions.shape!=(model.node_count,3) or np.any(initial_nodes.velocities) or np.any(initial_nodes.accelerations)):
        raise MooringEquilibriumError('stationary exact node inventory required')
    accepted = []; offset = 6*len(model.bodies)
    stopped = lambda:bool(cancel is not None and cancel())
    if stopped():return MooringStaticResult('cancelled',(),'cancelled before initialization')
    try:
        nodes = initialize_mooring_catenaries(model,states) if initial_nodes is None else initial_nodes
    except (MooringLineError,MooringEquilibriumError,ValueError) as error:
        return MooringStaticResult('failed',(),str(error))
    stages = [('weight',f,f,0.,False) for f in spec.load_fractions]
    if model.environment is not None:
        stages += [('current',f,1.,f,False) for f in spec.load_fractions]
    if model.seabed_contact is not None:stages.append(('contact',1.,1.,1.,True))
    if not stages:raise AssertionError('missing full load stage')
    for name,fraction,weight,current,contact in stages:
        stage_model = _stage_model(model,contact,current)
        least_norm_steps = 0
        def evaluate(n):
            out = stage_model.evaluate(states,n)
            force = out.force[offset:]-(1.-weight)*out.force_components['weight'][offset:]
            norm = float(np.max(np.abs(force),initial=0.))
            return out,force,norm
        conservative_contact = contact and stage_model.environment is None
        reference_positions = nodes.positions
        def potential(n, out):
            return _contact_potential(n,out,offset,weight,reference_positions)
        try:
            for iteration in range(spec.max_iterations+1):
                if stopped():return MooringStaticResult('cancelled',tuple(accepted),f'cancelled in {name}')
                out,force,norm = evaluate(nodes)
                if norm<=spec.force_limit:
                    normalized=norm/spec.force_scale
                    if not math.isfinite(normalized):
                        raise MooringEquilibriumError('nonfinite normalized residual; choose a representable force scale')
                    accepted.append(MooringStaticStage(name,fraction,nodes,norm,normalized,iteration,least_norm_steps))
                    break
                if iteration==spec.max_iterations:
                    raise MooringEquilibriumError(f'{name} iteration limit; residual {norm:g} N')
                tangent = out.stiffness[offset:,offset:]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',MatrixRankWarning)
                        delta = spsolve(tangent,force)
                except MatrixRankWarning:
                    # Slack grounded segments can have zero horizontal K and
                    # zero horizontal residual: a compatible neutral mechanism.
                    # Solve on its range without adding a physical penalty.
                    delta = lsmr(tangent,force,atol=1e-12,btol=1e-12,
                                 maxiter=min(500,max(10,3*len(force))))[0]
                    least_norm_steps += 1
                if not np.isfinite(delta).all():raise MooringEquilibriumError('nonfinite static Newton direction')
                linear_error = float(np.max(np.abs(tangent@delta-force),initial=0.))
                if linear_error>max(spec.force_limit,.1*norm):
                    raise MooringEquilibriumError('static tangent cannot balance residual on its range')
                energy,energy_scale = potential(nodes,out) if conservative_contact else (None,None)
                slope = float(force@delta) if conservative_contact else 0.
                for search in range(spec.max_line_search):
                    if stopped():return MooringStaticResult('cancelled',tuple(accepted),f'cancelled in {name} line search')
                    trial = replace(nodes,positions=nodes.positions+2.**(-search)*delta.reshape(-1,3))
                    try:
                        trial_out,_,trial_norm = evaluate(trial)
                    except (MooringLineError,ValueError,OverflowError):continue
                    decrease = 1e-4*2.**(-search)*slope
                    trial_energy,trial_scale = potential(trial,trial_out) if conservative_contact else (None,None)
                    energy_resolved = (conservative_contact and math.isfinite(slope) and slope>0.
                                       and math.isfinite(energy) and math.isfinite(trial_energy)
                                       and math.isfinite(energy_scale) and math.isfinite(trial_scale)
                                       and decrease>32*math.ulp(max(energy_scale,trial_scale,1.)))
                    accepted_trial = (trial_energy<=energy-decrease if energy_resolved else
                                      trial_norm<(1.-1e-4*2.**(-search))*norm)
                    if accepted_trial:
                        nodes=trial;break
                else:raise MooringEquilibriumError(f'{name} admissible line search exhausted')
        except (MooringLineError,MooringEquilibriumError,MatrixRankWarning,ValueError,RuntimeError,OverflowError) as error:
            return MooringStaticResult('failed',tuple(accepted),str(error))
    if stopped():return MooringStaticResult('cancelled',tuple(accepted),'cancelled before final audit')
    try:
        out = model.evaluate(states,nodes)
        norm = float(np.max(np.abs(out.force[offset:]),initial=0.))
        if norm>spec.force_limit or np.any(out.tensions<0.):
            raise MooringEquilibriumError('final full-load residual/tension audit failed')
        normalized=norm/spec.force_scale
        if not math.isfinite(normalized):raise MooringEquilibriumError('nonfinite final normalized residual')
        component_error = float(np.max(np.abs(sum(out.force_components.values())-out.force),initial=0.))
        if component_error>spec.force_limit:raise MooringEquilibriumError('force decomposition audit failed')
        audit = {'residual_norm':norm,'normalized_residual':normalized,
                 'force_limit':spec.force_limit,'component_error':component_error,
                 'min_tension':float(np.min(out.tensions)),
                 'max_penetration':0. if out.contact_diagnostics is None else out.contact_diagnostics['max_penetration'],
                 'anchor_reactions':_array(out.anchor_reactions,out.anchor_reactions.shape,'anchor reactions'),
                 'fixed_body_generalized_forces':_array(out.force[:offset],(offset,),'body generalized forces')}
    except (MooringLineError,MooringEquilibriumError,ValueError,RuntimeError,OverflowError) as error:
        return MooringStaticResult('failed',tuple(accepted),str(error))
    return MooringStaticResult('converged',tuple(accepted),'fixed-fairlead full-load residual audited; development only',MappingProxyType(audit))
