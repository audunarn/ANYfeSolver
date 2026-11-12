"""Coupled fixed-load body/link/line equilibrium, development not qualification."""
from dataclasses import dataclass,replace
from types import MappingProxyType
import math
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve,lsmr,MatrixRankWarning
from .marine_mechanical import MarineMechanicalModel
from .marine_rigid import RigidBodyState
from .marine_mooring import MooringNodeState,_array
from .marine_equilibrium import (MooringEquilibriumError,MooringStaticSpec,
    solve_mooring_equilibrium,_real,_integer)
from .marine_contributions import (ContributionSet,CommittedHistory,DryGravityContribution,
    MechanicalLinkSetContribution,WorldWrenchContribution,freeze_value)
from .marine_potential_flow import PotentialFlowDevelopmentContribution,equilibrium_geometry_hash
from .marine_morison import MorisonMemberSetContribution

API_VERSION='marine_coupled_equilibrium_development_v1'
__all__=['API_VERSION','CoupledStaticSpec','CoupledStaticAssembly','CoupledStaticStage',
         'CoupledStaticResult','evaluate_marine_static','solve_coupled_marine_equilibrium']


def _admit(model,states,nodes):
    if type(model) is not MarineMechanicalModel:
        raise MooringEquilibriumError('public MarineMechanicalModel required')
    try:states=dict(states)
    except (ValueError,TypeError) as error:
        raise MooringEquilibriumError('exact stationary body state mapping required') from error
    if set(states)!=set(model.body_ids) or any(type(s) is not RigidBodyState for s in states.values()):
        raise MooringEquilibriumError('exact public stationary body state inventory required')
    if any(np.any(s.velocity) or np.any(s.acceleration) for s in states.values()):
        raise MooringEquilibriumError('static bodies require zero velocity/acceleration')
    if model.mooring is None:
        if nodes is not None:raise MooringEquilibriumError('nodes supplied without line model')
    elif (type(nodes) is not MooringNodeState or nodes.positions.shape!=(model.mooring.node_count,3)
          or np.any(nodes.velocities) or np.any(nodes.accelerations)):
        raise MooringEquilibriumError('exact stationary free-node inventory required')
    for c in model.contributions:
        if type(c) not in (PotentialFlowDevelopmentContribution,MorisonMemberSetContribution,WorldWrenchContribution):
            raise MooringEquilibriumError(f'no admitted static contract for {type(c).__name__}; provider is not ignored')
        if type(c) is MorisonMemberSetContribution and any(x.amplitude!=0. for x in c.environment.components):
            raise MooringEquilibriumError('active member waves are not static loads')
    if model.mooring is not None and model.mooring.environment is not None and any(
            c.amplitude!=0. for c in model.mooring.environment.components):
        raise MooringEquilibriumError('active line waves are not static loads')
    return states


def _providers(model):
    providers=[]
    if model.links:providers.append(MechanicalLinkSetContribution(model.links))
    if model.gravity_world is not None:
        providers.extend(DryGravityContribution(b,model.gravity_world) for b in model.bodies)
    providers.extend(model.contributions)
    return tuple(providers)


@dataclass(frozen=True)
class CoupledStaticAssembly:
    force: np.ndarray
    stiffness: object
    components: object
    diagnostics: object
    mooring: object


def evaluate_marine_static(model,states,nodes=None):
    """Full physical force/K, with no transient clock/history or support load.

    Reference buoyancy/preload and selected gravity occur exactly once. Potential
    flow uses its public static contract; Morison uses stationary still/current
    kinematics. Original geometry/bounds and common ownership gates still apply.
    Force and components are immutable; sparse K is a fresh disposable matrix.
    """
    states=_admit(model,states,nodes);providers=_providers(model)
    owner=ContributionSet(model.body_ids,providers)
    n=owner.layout.size;total=n if model.mooring is None else model.mooring.size
    force=np.zeros(total);K=sparse.csr_matrix((total,total));components={};diagnostics={}
    for c in providers:
        trial=(c.evaluate_static(states,owner.layout) if type(c) in
               (PotentialFlowDevelopmentContribution,MorisonMemberSetContribution)
               else c.evaluate_trial(CommittedHistory(),states,owner.layout))
        if trial.trial_history is not None:
            raise MooringEquilibriumError('static trial cannot propose committed history')
        f=_array(trial.external_force,(n,),'static provider force')
        k=_array(trial.stiffness,(n,n),'static provider stiffness')
        padded=np.r_[f,np.zeros(total-n)];force+=padded
        K+=sparse.block_diag((sparse.csr_matrix(k),sparse.csr_matrix((total-n,total-n))),format='csr')
        components[c.contribution_id]=padded;diagnostics[c.contribution_id]=trial.diagnostics
    mooring=None
    if model.mooring is not None:
        mooring=model.mooring.evaluate(states,nodes)
        force+=mooring.force;K+=mooring.stiffness
        components.update({'mooring:'+k:v for k,v in mooring.force_components.items()})
    if not np.isfinite(force).all() or not np.isfinite(K.data).all():
        raise MooringEquilibriumError('nonfinite static force/tangent')
    return CoupledStaticAssembly(_array(force,(total,),'static force'),K,
                                 freeze_value(components),freeze_value(diagnostics),mooring)


@dataclass(frozen=True)
class CoupledStaticSpec:
    force_scale: float=1.                 # N
    moment_scale: float=1.                # Nm
    force_atol: float=1e-8                # N
    moment_atol: float=1e-8               # Nm
    relative_tolerance: float=1e-8
    release_fractions: tuple=(.25,.5,.75,1.)
    max_iterations: int=40
    max_line_search: int=20

    def __post_init__(self):
        for name in ('force_scale','moment_scale','force_atol','moment_atol','relative_tolerance'):
            object.__setattr__(self,name,_real(getattr(self,name),name,
                positive=name.endswith('scale'),nonnegative=not name.endswith('scale')))
        for limit in (self.force_limit,self.moment_limit):
            if not math.isfinite(limit) or limit<=0.:
                raise MooringEquilibriumError('positive finite derived force/moment limits required')
        fractions=tuple(_real(f,'support release fraction',positive=True) for f in self.release_fractions)
        if not fractions or fractions[-1]!=1. or any(b<=a for a,b in zip(fractions,fractions[1:])):
            raise MooringEquilibriumError('release fractions must increase to one')
        object.__setattr__(self,'release_fractions',fractions)
        _integer(self.max_iterations,'max iterations');_integer(self.max_line_search,'max line search')

    @property
    def force_limit(self):return self.force_atol+self.relative_tolerance*self.force_scale

    @property
    def moment_limit(self):return self.moment_atol+self.relative_tolerance*self.moment_scale


def _world(force,model,states):
    world=np.array(force,copy=True)
    for i,b in enumerate(model.bodies):
        world[6*i:6*i+6]=np.linalg.solve(states[b.body_id].rate_map.T,force[6*i:6*i+6])
    return world


def _metrics(force,model,states,spec):
    w=_world(force,model,states);offset=6*len(model.bodies)
    translations=np.r_[np.array([w[6*i:6*i+3] for i in range(len(model.bodies))]).ravel(),w[offset:]]
    rotations=np.array([w[6*i+3:6*i+6] for i in range(len(model.bodies))]).ravel()
    f=float(np.max(abs(translations),initial=0.));m=float(np.max(abs(rotations),initial=0.))
    metric=max(f/spec.force_limit,m/spec.moment_limit)
    normalized=max(f/spec.force_scale,m/spec.moment_scale)
    if not all(math.isfinite(x) for x in (f,m,metric,normalized)):
        raise MooringEquilibriumError('nonfinite WORLD residual metric/normalization')
    return dict(force_residual=f,moment_residual=m,tolerance_ratio=metric,normalized_residual=normalized)


@dataclass(frozen=True)
class CoupledStaticStage:
    fraction: float
    states: object
    nodes: object
    metrics: object
    iterations: int
    least_norm_steps: int

    def __post_init__(self):
        object.__setattr__(self,'states',MappingProxyType(dict(self.states)))
        object.__setattr__(self,'metrics',freeze_value(self.metrics))


@dataclass(frozen=True)
class CoupledStaticResult:
    status: str
    stages: tuple
    message: str
    line_result: object=None
    audit: object=None

    @property
    def states(self):return self.stages[-1].states if self.stages else None

    @property
    def nodes(self):
        return self.stages[-1].nodes if self.stages else (None if self.line_result is None else self.line_result.nodes)


def solve_coupled_marine_equilibrium(model,initial_states,initial_nodes=None,spec=None,*,cancel=None):
    """Line-only staged initialization, then release explicit WORLD body holds.

    Holds are continuation loads at body origins, never penalty constraints.
    Their forces/follower tangents vanish at fraction1 and are excluded from
    final physical audit. No history or coefficient-frame rebind occurs.
    """
    spec=CoupledStaticSpec() if spec is None else spec
    if type(spec) is not CoupledStaticSpec:raise MooringEquilibriumError('named coupled static spec required')
    if type(model) is not MarineMechanicalModel:
        raise MooringEquilibriumError('public MarineMechanicalModel required')
    # If nodes need initializing, admit an empty stationary placeholder first.
    if model.mooring is not None and initial_nodes is None:
        placeholder=np.zeros((model.mooring.node_count,3))
        nodes=MooringNodeState(placeholder,placeholder,placeholder)
    else:nodes=initial_nodes
    states=_admit(model,initial_states,nodes);accepted=[];line_result=None
    if cancel is not None and not callable(cancel):raise MooringEquilibriumError('cancel must be callable')
    stopped=lambda:bool(cancel is not None and cancel())
    result=lambda status,message,audit=None:CoupledStaticResult(status,tuple(accepted),message,line_result,audit)
    if stopped():return result('cancelled','cancelled before line initialization')
    if model.mooring is not None:
        line_spec=MooringStaticSpec(spec.force_scale,spec.force_atol,spec.relative_tolerance,
                                   max_iterations=spec.max_iterations,max_line_search=spec.max_line_search)
        line_result=solve_mooring_equilibrium(model.mooring,states,initial_nodes,line_spec,cancel=cancel)
        if line_result.status!='converged':return result(line_result.status,line_result.message)
        nodes=line_result.nodes
    try:
        original=evaluate_marine_static(model,states,nodes)
        holding=-_world(original.force,model,states)
        holds=tuple(WorldWrenchContribution('static-support:'+b.body_id,b,(0.,0.,0.),
                    holding[6*i:6*i+3],holding[6*i+3:6*i+6]) for i,b in enumerate(model.bodies))
        layout=ContributionSet(model.body_ids,()).layout
        n=6*len(model.bodies);size=len(original.force)
        def evaluate(ss,nn,fraction):
            out=evaluate_marine_static(model,ss,nn);force=np.array(out.force);K=out.stiffness.copy()
            if fraction!=1.:
                for hold in holds:
                    trial=hold.evaluate_trial(CommittedHistory(),ss,layout)
                    force[:n]+=(1.-fraction)*trial.external_force
                    K+=sparse.block_diag(((1.-fraction)*sparse.csr_matrix(trial.stiffness),
                                         sparse.csr_matrix((size-n,size-n))),format='csr')
            return out,force,K,_metrics(force,model,ss,spec)
        for fraction in spec.release_fractions:
            least_norm_steps=0
            for iteration in range(spec.max_iterations+1):
                if stopped():return result('cancelled',f'cancelled at support release {fraction:g}')
                out,force,K,metrics=evaluate(states,nodes,fraction)
                if metrics['tolerance_ratio']<=1.:
                    # Static rates are zero: exact accepted rechart needs no
                    # dynamic rate/history transport. Re-evaluate before commit.
                    recharted={bid:replace(s,rotation_vector=(0.,0.,0.),anchor=s.rotation)
                               for bid,s in states.items()}
                    _,_,_,check=evaluate(recharted,nodes,fraction)
                    if check['tolerance_ratio']>1.:
                        raise MooringEquilibriumError('accepted static rechart residual audit failed')
                    states= recharted;metrics=check
                    accepted.append(CoupledStaticStage(fraction,states,nodes,metrics,iteration,least_norm_steps));break
                if iteration==spec.max_iterations:raise MooringEquilibriumError('coupled static iteration limit')
                with warnings.catch_warnings():
                    warnings.simplefilter('error',MatrixRankWarning)
                    try:delta=spsolve(K,force)
                    except MatrixRankWarning:
                        # Row-scaled, same physical operator, no regularization.
                        row=np.full(size,1./spec.force_scale)
                        for i in range(len(model.bodies)):row[6*i+3:6*i+6]=1./spec.moment_scale
                        scaled=sparse.diags(row)@K
                        delta=lsmr(scaled,row*force,atol=1e-12,btol=1e-12,maxiter=min(1000,max(10,3*size)))[0]
                        least_norm_steps+=1
                if not np.isfinite(delta).all():raise MooringEquilibriumError('nonfinite coupled Newton direction')
                linear=_metrics(K@delta-force,model,states,spec)['tolerance_ratio']
                if linear>max(1.,.1*metrics['tolerance_ratio']):
                    raise MooringEquilibriumError('coupled tangent cannot balance residual on its range')
                for search in range(spec.max_line_search):
                    if stopped():return result('cancelled','cancelled during coupled line search')
                    d=2.**(-search)*delta
                    try:
                        trial_states={b.body_id:replace(states[b.body_id],
                            position=np.array(states[b.body_id].position)+d[6*i:6*i+3],
                            rotation_vector=np.array(states[b.body_id].rotation_vector)+d[6*i+3:6*i+6])
                            for i,b in enumerate(model.bodies)}
                        trial_nodes=None if nodes is None else replace(nodes,positions=nodes.positions+d[n:].reshape(-1,3))
                        _,_,_,trial_metrics=evaluate(trial_states,trial_nodes,fraction)
                    except (ValueError,RuntimeError,OverflowError,np.linalg.LinAlgError):continue
                    if trial_metrics['tolerance_ratio']<(1.-1e-4*2.**(-search))*metrics['tolerance_ratio']:
                        states,nodes=trial_states,trial_nodes;break
                else:raise MooringEquilibriumError('coupled admissible line search exhausted')
        if stopped():return result('cancelled','cancelled before coupled final audit')
        physical=evaluate_marine_static(model,states,nodes);metrics=_metrics(physical.force,model,states,spec)
        if metrics['tolerance_ratio']>1.:raise MooringEquilibriumError('coupled physical final residual audit failed')
        discrepancy=sum(physical.components.values())-physical.force
        component_error=float(np.max(abs(discrepancy),initial=0.))
        if _metrics(discrepancy,model,states,spec)['tolerance_ratio']>1.:
            raise MooringEquilibriumError('coupled force decomposition audit failed')
        geometry=[]
        for c in model.contributions:
            if type(c) is PotentialFlowDevelopmentContribution:
                frames=tuple(replace(f,equilibrium_position=states[f.body_id].position,
                                     equilibrium_rotation=states[f.body_id].rotation) for f in c.data.frames)
                new_hash=equilibrium_geometry_hash(frames)
                geometry.append(dict(source_equilibrium_hash=c.data.equilibrium_hash,
                    final_equilibrium_hash=new_hash,coefficient_rebind_required=new_hash!=c.data.equilibrium_hash))
        audit=freeze_value({**metrics,'force_limit':spec.force_limit,'moment_limit':spec.moment_limit,
            'body_world_residual':_world(physical.force,model,states)[:n].reshape(-1,6),
            'node_residual':physical.force[n:].reshape(-1,3),'component_error':component_error,
            'temporary_support_fraction':0.,'potential_geometry':geometry,'components':physical.components,
            'max_penetration':0. if physical.mooring is None or physical.mooring.contact_diagnostics is None
                else physical.mooring.contact_diagnostics['max_penetration']})
        return result('converged','coupled physical residual audited; development only',audit)
    except (ValueError,RuntimeError,OverflowError,np.linalg.LinAlgError) as error:
        return result('failed',str(error))
