"""Distributed prescribed Morison contributions; DEVELOPMENT, unqualified.

SI/WORLD, instantaneous straight members, full nominal circular section over
the centreline interval between flat bed and still-water level. No stretching,
slamming, structural member stiffness, large-excursion BEM admission or
variational added-mass forces. ANYloads owns the environment and force density.
"""
from dataclasses import dataclass,field
from collections.abc import Mapping
import numpy as np
from .marine_rigid import (RigidBody6D, RigidBodyState, skew,
                           left_jacobian_rate, left_jacobian_second_rate)
from .marine_contributions import (DynamicContribution, ContributionResult,
    MarineContributionsError, CommittedHistory, StepInformation, HISTORY_PHYSICAL)

API_VERSION = 'marine_morison_development_v1'


class MarineMorisonError(MarineContributionsError):
    """Invalid member, environment, state, history or branch; no fallback."""


def _identifier(value, name):
    if not isinstance(value,str) or not value.strip():
        raise MarineMorisonError(f'{name} must be a nonempty string')
    return value


def _point(value):
    try:
        values=tuple(value)
    except TypeError as error:
        raise MarineMorisonError('point_body must be a finite real3-vector') from error
    if any(isinstance(v,(bool,np.bool_)) for v in values):
        raise MarineMorisonError('point_body cannot contain boolean coordinates')
    raw=np.asarray(values)
    if raw.shape!=(3,) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise MarineMorisonError('point_body must be a finite real3-vector')
    return tuple(float(v) for v in raw)


def _loads():
    try:
        from anyloads import marine
        return marine
    except ImportError as error:
        raise MarineMorisonError('Morison requires the tested ANYloads marine API') from error


@dataclass(frozen=True)
class MemberEndpoint:
    body_id: str
    point_body: tuple

    def __post_init__(self):
        _identifier(self.body_id,'body_id')
        object.__setattr__(self,'point_body',_point(self.point_body))


@dataclass(frozen=True)
class MorisonMember:
    member_id: str
    end_a: MemberEndpoint
    end_b: MemberEndpoint
    section: object
    ownership: object
    quadrature_order: int = 4

    def __post_init__(self):
        loads=_loads()
        _identifier(self.member_id,'member_id')
        if type(self.end_a) is not MemberEndpoint or type(self.end_b) is not MemberEndpoint:
            raise MarineMorisonError('named MemberEndpoints required')
        if self.end_a==self.end_b:
            raise MarineMorisonError('identical member endpoints have zero current length')
        if type(self.section) is not loads.MorisonSection or type(self.ownership) is not loads.HydrodynamicOwnership:
            raise MarineMorisonError('named ANYloads section and ownership required')
        if type(self.quadrature_order) is not int or not 2<=self.quadrature_order<=16:
            raise MarineMorisonError('quadrature_order must be integer2..16')


def _endpoint(body,state,point,layout):
    """Public point kinematics plus exact local chart partials, at fixed rho."""
    kin=body.point_kinematics(state,point)
    n=layout.size;base=layout.index(body.body_id)
    J=np.zeros((3,n));J[:,base:base+6]=kin.jacobian
    dJ=np.zeros((3,n,n));vq=np.zeros((3,n));aq=np.zeros((3,n));av=np.zeros((3,n))
    p=state.rotation@np.asarray(point)
    theta=np.asarray(state.rotation_vector);rate=np.asarray(state.velocity[3:])
    acceleration=np.asarray(state.acceleration[3:])
    L=state.rate_map[3:,3:];Ld=state.rate_map_derivative[3:,3:]
    omega=state.world_velocity[3:];alpha=state.world_acceleration[3:]
    for j in range(3):
        e=np.eye(3)[j];dL=left_jacobian_rate(theta,e)
        dp=np.cross(L@e,p)
        dw=dL@rate
        da=dL@acceleration+left_jacobian_second_rate(theta,e,rate)@rate
        col=base+3+j
        dJ[:,base+3:base+6,col]=-skew(dp)@L-skew(p)@dL
        vq[:,col]=np.cross(dw,p)+np.cross(omega,dp)
        aq[:,col]=(np.cross(da,p)+np.cross(alpha,dp)
            +np.cross(dw,np.cross(omega,p))
            +np.cross(omega,np.cross(dw,p)+np.cross(omega,dp)))
        dw=L@e
        da=dL@rate+Ld@e
        av[:,col]=(np.cross(da,p)+np.cross(dw,np.cross(omega,p))
                   +np.cross(omega,np.cross(dw,p)))
    return kin,J,dJ,vq,aq,av


def _wet_nodes(xa,xb,Ja,Jb,environment,order):
    """Moving Gauss intervals split at bed/SWL/current knots.

    Open quadrature nodes; inclusively wet horizontal segments. At an exact
    endpoint/plane or horizontal/plane event, report the branch convention;
    no differentiability claim at topology changes. Parameter derivatives
    move sample locations only, never redefine material velocity/acceleration.
    """
    delta=xb-xa;length=float(np.linalg.norm(delta))
    if not np.isfinite(length) or length<=1e-12:
        raise MarineMorisonError('member current length must exceed1e-12m')
    axis=delta/length;Dq=Jb-Ja
    lq=axis@Dq;axis_q=(np.eye(3)-np.outer(axis,axis))@Dq/length
    planes=tuple(environment.wet_levels)
    dz=delta[2];bounds=[(0.,np.zeros(Ja.shape[1])),(1.,np.zeros(Ja.shape[1]))]
    events=[]
    for h in planes:
        if xa[2]==h or xb[2]==h:
            events.append(dict(level=float(h),kind='endpoint_on_plane'))
        if dz!=0.:
            s=float((h-xa[2])/dz)
            if 0.<s<1.:
                sq=-((1.-s)*Ja[2]+s*Jb[2])/dz
                bounds.append((s,sq))
        elif xa[2]==h:
            events.append(dict(level=float(h),kind='horizontal_on_plane'))
    bounds.sort(key=lambda b:b[0])
    xi,wi=np.polynomial.legendre.leggauss(order)
    edges=np.asarray([bound[0] for bound in bounds])
    edges_q=np.stack([bound[1] for bound in bounds])
    z=xa[2]+.5*(edges[:-1]+edges[1:])*dz
    wet=(z>=environment.still_water_level-environment.water_depth)&(z<=environment.still_water_level)
    lo,hi=edges[:-1][wet],edges[1:][wet]
    loq,hiq=edges_q[:-1][wet],edges_q[1:][wet]
    # Interval/boundary records are independent of Gauss order. There are no
    # Python records or iterations per quadrature point in the hot path.
    s=.5*((1.-xi[None,:])*lo[:,None]+(1.+xi[None,:])*hi[:,None])
    sq=.5*((1.-xi[None,:,None])*loq[:,None,:]+(1.+xi[None,:,None])*hiq[:,None,:])
    weight=.5*wi[None,:]*length*(hi-lo)[:,None]
    weight_q=.5*wi[None,:,None]*((hi-lo)[:,None,None]*lq[None,None,:]
                              +length*(hiq-loq)[:,None,:])
    n=Ja.shape[1]
    nodes=(s.reshape(-1),sq.reshape(-1,n),weight.reshape(-1),weight_q.reshape(-1,n))
    segments=np.column_stack((lo,hi)).tolist()
    return axis,axis_q,length,nodes,segments,events


@dataclass(frozen=True)
class MorisonMemberSetContribution(DynamicContribution):
    contribution_id: str = field()
    bodies: tuple
    members: tuple
    environment: object
    history_basis = HISTORY_PHYSICAL
    history_transport = None

    def __post_init__(self):
        _identifier(self.contribution_id,'contribution_id')
        loads=_loads()
        bodies=tuple(self.bodies);members=tuple(self.members)
        if not bodies or any(type(b) is not RigidBody6D for b in bodies) or len({b.body_id for b in bodies})!=len(bodies):
            raise MarineMorisonError('unique named RigidBody6D instances required')
        if not members or any(type(m) is not MorisonMember for m in members) or len({m.member_id for m in members})!=len(members):
            raise MarineMorisonError('unique named MorisonMember instances required')
        ids={b.body_id for b in bodies}
        if any(e.body_id not in ids for m in members for e in (m.end_a,m.end_b)):
            raise MarineMorisonError('member endpoint references unknown body')
        if type(self.environment) is not loads.AiryEnvironment:
            raise MarineMorisonError('named ANYloads AiryEnvironment required')
        object.__setattr__(self,'bodies',bodies);object.__setattr__(self,'members',members)

    @property
    def ownership_key(self):
        return 'morison-members:'+self.contribution_id

    def evaluate_trial(self,committed_history,states,layout):
        raise MarineMorisonError('Morison requires explicit StepInformation')

    def evaluate_static(self,states,layout):
        """Stationary buoyancy/current trial, same public distributed force law.

        Active waves are refused. No history or inertia is advanced; zero
        structural/fluid acceleration makes added-inertia forces vanish.
        """
        from .marine_contributions import BodyLayout
        if (not isinstance(layout,BodyLayout) or not isinstance(states,Mapping)
                or set(states)!=set(layout.body_ids)
                or any(not isinstance(s,RigidBodyState) for s in states.values())):
            raise MarineMorisonError('static Morison requires exact immutable states/layout')
        if any(c.amplitude!=0. for c in self.environment.components):
            raise MarineMorisonError('active waves cannot be frozen as static Morison loads')
        if any(np.any(s.velocity) or np.any(s.acceleration) for s in states.values()):
            raise MarineMorisonError('static Morison requires stationary states')
        trial=self.evaluate_trial_with_context(CommittedHistory(),states,layout,
                                              StepInformation(0,1.,'initialization'))
        zeros=np.zeros((layout.size,layout.size))
        return ContributionResult(trial.contribution_id,trial.external_force,trial.stiffness,
                                  zeros,zeros.copy(),trial.diagnostics)

    def evaluate_trial_with_context(self,committed_history,states,layout,information):
        if not isinstance(information,StepInformation):
            raise MarineMorisonError('Morison requires explicit StepInformation')
        if not isinstance(committed_history,CommittedHistory) or committed_history.get(self.contribution_id) is not None:
            raise MarineMorisonError('stateless Morison cannot discard unexpected committed history')
        if not isinstance(states,Mapping) or set(states)!=set(layout.body_ids) or any(not isinstance(s,RigidBodyState) for s in states.values()):
            raise MarineMorisonError('immutable states must match layout body ids')
        if any(body.body_id not in states for body in self.bodies):
            raise MarineMorisonError('Morison body is absent from the trial layout')
        by_id={body.body_id:body for body in self.bodies}
        loads=_loads();n=layout.size
        Q=np.zeros(n);K=np.zeros((n,n));C=np.zeros_like(K);Ma=np.zeros_like(K)
        diagnostics={}
        for member in self.members:
            ends=[_endpoint(by_id[e.body_id],states[e.body_id],e.point_body,layout)
                  for e in (member.end_a,member.end_b)]
            (a,Ja,dJa,vqa,aqa,ava),(b,Jb,dJb,vqb,aqb,avb)=ends
            axis,axis_q,length,nodes,segments,events=_wet_nodes(a.position,b.position,Ja,Jb,
                self.environment,member.quadrature_order)
            components={key:np.zeros(n) for key in ('froude_krylov','added_mass','drag','buoyancy')}
            attribution={key:np.zeros(n) for key in ('wave_and_motion','current_alone','interaction')}
            power=0.;wet_length=0.;member_Q=np.zeros(n)
            s,sq,w,wq=nodes
            if s.size:
                ss=s[:,None]
                x=(1.-ss)*a.position+ss*b.position
                v=(1.-ss)*a.velocity+ss*b.velocity
                acc=(1.-ss)*a.acceleration+ss*b.acceleration
                fluid=self.environment.evaluate(x,information.time)
                force=loads.morison_force(member.section,member.ownership,self.environment.density,
                    fluid.velocity,fluid.acceleration,v,acc,axis,
                    gravity=self.environment.gravity,wave_velocity=fluid.wave_velocity,
                    current_velocity=fluid.current_velocity)
                scale=s[:,None,None]
                J=(1.-scale)*Ja+scale*Jb
                xq=J+(b.position-a.position)[None,:,None]*sq[:,None,:]
                vq=(1.-scale)*vqa+scale*vqb+(b.velocity-a.velocity)[None,:,None]*sq[:,None,:]
                aq=(1.-scale)*aqa+scale*aqb+(b.acceleration-a.acceleration)[None,:,None]*sq[:,None,:]
                av=(1.-scale)*ava+scale*avb
                fq=(force.fluid_velocity_derivative@fluid.velocity_gradient@xq
                    +force.fluid_acceleration_derivative@fluid.acceleration_gradient@xq
                    +force.structure_velocity_derivative@vq
                    +force.structure_acceleration_derivative@aq+force.axis_derivative@axis_q)
                fv=force.structure_velocity_derivative@J+force.structure_acceleration_derivative@av
                fa=force.structure_acceleration_derivative@J
                covector=np.einsum('pan,pa->pn',J,force.force)
                member_Q=np.einsum('p,pn->n',w,covector)
                # Contract dJ before broadcasting: avoid a points×n×n tangent
                # temporary while retaining the moving-sample follower term.
                follower=(np.einsum('ank,a->nk',dJa,np.einsum('p,p,pa->a',w,1.-s,force.force))
                    +np.einsum('ank,a->nk',dJb,np.einsum('p,p,pa->a',w,s,force.force))
                    +np.einsum('an,ak->nk',Jb-Ja,np.einsum('p,pa,pk->ak',w,force.force,sq)))
                K-=(follower
                    +np.einsum('p,pan,pak->nk',w,J,fq)+np.einsum('pn,pk->nk',covector,wq))
                C-=np.einsum('p,pan,pak->nk',w,J,fv)
                Ma-=np.einsum('p,pan,pak->nk',w,J,fa)
                for key,values in force.components.items():
                    components[key]=np.einsum('p,pan,pa->n',w,J,values)
                for key,values in force.drag_attribution.items():
                    attribution[key]=np.einsum('p,pan,pa->n',w,J,values)
                power=float(w@force.relative_drag_power);wet_length=float(w.sum())
            Q+=member_Q
            diagnostics[member.member_id]=dict(force=member_Q,components=components,
                drag_attribution=attribution,relative_drag_power=power,wet_length=wet_length,
                total_length=length,wet_parameter_segments=segments,branch_events=events,
                quadrature_order=member.quadrature_order,
                ownership=loads.ownership_to_dict(member.ownership))
        return ContributionResult(self.contribution_id,Q,K,C,Ma,
            dict(api_version=API_VERSION,time=information.time,members=diagnostics,
                 approximation='flat_bed_swl_centerline_full_nominal_section',
                 source_admission='development declaration only; no real BEM exclusion proof'),None)


def validate_member_ownership(contributions,body_ids,canonical_bodies=None):
    """Shared model/owner guard: one force provider per persistent member id."""
    seen=set();ids=set(body_ids)
    by_id=None if canonical_bodies is None else {b.body_id:b for b in canonical_bodies}
    for contribution in contributions:
        if isinstance(contribution,MorisonMemberSetContribution):
            if any(b.body_id not in ids or by_id is not None and by_id[b.body_id]!=b for b in contribution.bodies):
                raise MarineMorisonError('Morison bodies must match canonical model/layout bodies')
            keys={member.member_id for member in contribution.members}
            if seen.intersection(keys):
                raise MarineMorisonError('duplicate Morison member physical ownership refused')
            seen.update(keys)


def data_to_dict(contribution):
    if type(contribution) is not MorisonMemberSetContribution:
        raise MarineMorisonError('serialization cannot erase subclass physics')
    loads=_loads()
    return dict(api_version=API_VERSION,contribution_id=contribution.contribution_id,
        environment=loads.environment_to_dict(contribution.environment),members=[dict(
            member_id=m.member_id,end_a=dict(body_id=m.end_a.body_id,point_body=list(m.end_a.point_body)),
            end_b=dict(body_id=m.end_b.body_id,point_body=list(m.end_b.point_body)),
            section=loads.section_to_dict(m.section),ownership=loads.ownership_to_dict(m.ownership),
            quadrature_order=m.quadrature_order) for m in contribution.members])


def data_from_dict(data,bodies):
    loads=_loads()
    if type(data) is not dict or set(data)!={'api_version','contribution_id','environment','members'} or data['api_version']!=API_VERSION:
        raise MarineMorisonError('Morison fields/version must match the named schema')
    if not isinstance(data['members'],(tuple,list)):
        raise MarineMorisonError('Morison members must be a sequence')
    members=[]
    try:
        for item in data['members']:
            if type(item) is not dict or set(item)!={'member_id','end_a','end_b','section','ownership','quadrature_order'}:
                raise MarineMorisonError('member fields must match the named schema')
            ends=[]
            for name in ('end_a','end_b'):
                endpoint=item[name]
                if type(endpoint) is not dict or set(endpoint)!={'body_id','point_body'}:
                    raise MarineMorisonError('endpoint fields must match the named schema')
                ends.append(MemberEndpoint(**endpoint))
            members.append(MorisonMember(item['member_id'],*ends,loads.section_from_dict(item['section']),
                loads.ownership_from_dict(item['ownership']),item['quadrature_order']))
        return MorisonMemberSetContribution(data['contribution_id'],bodies,tuple(members),
                                           loads.environment_from_dict(data['environment']))
    except (TypeError,ValueError,KeyError) as error:
        raise MarineMorisonError(f'invalid named Morison data: {error}') from error
