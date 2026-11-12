"""Connected retained native midpoint DEVELOPMENT owner, source unverified.

Boolean restrictions assemble shared nodes/frames and independent physical cells.
No existing single-element owner/admission or constitutive formula is modified.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import Lock
from time import monotonic
import numpy as np
from ._ge_beam3_native_analysis import NativeBeamAnalysis
from ._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
from ._ge_beam3_retained_generalized import RetainedGeneralizedOperator
from ._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
from ._ge_beam3_generalized_ellipsoid_section import EllipsoidalGeneralizedSection
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_centered_finite_inertia import evaluate_centered_finite_inertia
from ._ge_beam3_p5.algebra import rotation
from ._ge_beam3_p5.chart import exp_chart_terms
from ._ge_beam3_p5.compensated_coordinates import split_sum, validate_pair
from ._ge_beam3_p5.arrays import _frames, _array
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._native_reference_modal import _owned
from .control import cancellation_safe_point

PROFILE='GE_BEAM3_CONNECTED_CENTERED_ELASTIC_MIDPOINT_DEV_V1'
SCHEMA='GE_BEAM3_CONNECTED_NATIVE_DYNAMIC_CHECKPOINT_DEV_V1'
MAX_BYTES=2*1024**2
MAX_COORDINATES=512


@dataclass(frozen=True)
class ConnectedState:
    positions: np.ndarray
    position_low: np.ndarray
    nodal_frames: np.ndarray
    cell_rotations: np.ndarray
    resultants: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    time: float
    epoch: int
    identity: str
    histories: tuple


@dataclass(frozen=True)
class ConnectedCandidate:
    predecessor: str
    endpoint: ConnectedState
    stage: ConnectedState
    residual: np.ndarray
    jacobian: np.ndarray
    accounting: dict
    dt: float
    unknown: np.ndarray
    iterations: int
    backtracks: int


class _RetainedMidpointAssembly:
    """One globally coupled physical/algebraic owner, no child time steppers."""
    route_profile=PROFILE
    state_type=ConnectedState
    checkpoint_schema=SCHEMA
    extra_step_keys=()
    retryable=('connected midpoint iteration limit','connected midpoint backtracking failed')
    def __init__(self, definitions, boundaries, *, inertia_order=24,
                 max_iterations=12, max_backtracks=8):
        self.started=monotonic();self._captured_started=self.started
        if type(inertia_order) is not int or inertia_order not in (4,8,24):
            raise ValueError('existing native inertia orders required')
        if type(max_iterations) is not int or not 1<=max_iterations<=12 or type(max_backtracks) is not int or not 0<=max_backtracks<=8:
            raise ValueError('bounded native Newton controls required')
        self.analysis=NativeBeamAnalysis(definitions,boundaries)
        self.elements=self.analysis._elements;self._captured_elements=self.elements
        self._admit_element_count(len(self.elements))
        self.nodes=tuple(sorted(self.analysis.model.mesh.nodes))
        self.node_index={n:i for i,n in enumerate(self.nodes)}
        self.n=len(self.nodes);self.e=len(self.elements);self.size=6*self.n+24*self.e
        if self.size>MAX_COORDINATES:
            raise ValueError('connected512-coordinate dense budget exceeded before allocation')
        self.inertia_order=inertia_order;self.max_iterations=max_iterations;self.max_backtracks=max_backtracks
        self._lock=Lock();self._current=None;self._visible=None;self._issued={};self._candidates={}
        ref={};frames={};maps=[];seen=set(self.elements[0].node_ids)
        for k,element in enumerate(self.elements):
            if type(element) is not NativeGeneralizedStaticElement or type(element.operator) is not RetainedGeneralizedOperator or type(element.operator.reference) is not CenteredCurvedBeam3ReferenceGeometry or type(element.section) is not EllipsoidalGeneralizedSection:
                raise ValueError('exact centered generalized elastic operators required')
            reference=element.operator.reference
            x=reference.coordinates
            direction=x[-1]-x[0]
            if not np.array_equal(x[1],.5*(x[0]+x[-1])) or np.linalg.norm(direction)==0:
                raise ValueError('first connected candidate requires straight zero-lift elements')
            for node,position,frame in zip(element.node_ids,x,reference.nodal_triads):
                if node in ref and (not np.array_equal(ref[node],position) or not np.array_equal(frames[node],frame)):
                    raise ValueError('shared reference position/frame trace disagrees')
                ref[node]=position;frames[node]=frame
            inertia=np.asarray(self.analysis._inertias[element.element_id])
            if inertia.shape!=(6,6) or not np.array_equal(inertia,np.diag(np.diag(inertia))) or np.any(np.diag(inertia)<=0):
                raise ValueError('positive centered diagonal physical inertia required')
            maps.append(tuple(6*self.node_index[node]+axis for node in element.node_ids for axis in range(6))+
                tuple(range(6*self.n+6*k,6*self.n+6*k+6))+
                tuple(range(6*self.n+6*self.e+18*k,6*self.n+6*self.e+18*k+18)))
        self._admit_graph()
        self.maps=tuple(maps)
        self.reference_positions=_owned([ref[n] for n in self.nodes]);self.reference_frames=_owned([frames[n] for n in self.nodes])
        self.fixed=self.analysis._admit_boundaries()
        if self.fixed not in ((),tuple(range(6))):
            raise ValueError('free or homogeneous first sorted-node full clamp only')
        self.physical=tuple(6*i+a for i in range(self.n) for a in range(3))+tuple(range(6*self.n,6*self.n+6*self.e))
        self.algebraic=tuple(i for i in range(self.size) if i not in self.physical)
        self.p=tuple(i for i in self.physical if i not in self.fixed);self.z=tuple(i for i in self.algebraic if i not in self.fixed)
        self.free=tuple(i for i in range(self.size) if i not in self.fixed)
        self.histories=tuple(e.operator.cell.virgin() for e in self.elements)
        self.length=min(float(np.linalg.norm(e.operator.reference.coordinates[-1]-e.operator.reference.coordinates[0])) for e in self.elements)
        self.scale=np.ones(self.size)
        for i in range(self.n):self.scale[6*i+3:6*i+6]=self.length
        self.scale[6*self.n:6*self.n+6*self.e]=self.length
        for k in range(self.e):self.scale[6*self.n+6*self.e+18*k:6*self.n+6*self.e+18*k+6]=self.length
        self._binding=canonical(self._definition());self.identity=sha(self._definition())

    def _definition(self):
        return dict(profile=self.route_profile,analysis=self.analysis.identity,elements=[e.identity for e in self.elements],
            maps=self.maps,nodes=self.nodes,node_index=self.node_index,n=self.n,e=self.e,size=self.size,
            fixed=self.fixed,free=self.free,p=self.p,z=self.z,physical=self.physical,algebraic=self.algebraic,
            reference_positions=self.reference_positions,reference_frames=self.reference_frames,
            histories=self.histories,inertia=[self.analysis._inertias[e.element_id] for e in self.elements],
            inertia_order=self.inertia_order,iterations=self.max_iterations,backtracks=self.max_backtracks,
            coordinate_budget=MAX_COORDINATES,runtime_limit_seconds=120.,length=self.length,scale=self.scale)

    def _admit_element_count(self,count):
        if count<2:
            raise ValueError('explicit connected multi-element route requires at least two elements')

    def _admit_graph(self):
        # Legacy graph closure is unchanged and independent of traversal order.
        seen=set(self.elements[0].node_ids);pending=list(self.elements[1:])
        while pending:
            found=next((e for e in pending if seen.intersection(e.node_ids)),None)
            if found is None:raise ValueError('declared element graph is disconnected')
            seen.update(found.node_ids);pending.remove(found)

    def _step_extra(self,previous,unknown,dt,stage,endpoint,token):
        return {}

    def _reference_physical_pose(self,state):
        data=self._data(state)
        data.update(positions=self.reference_positions,position_low=np.zeros((self.n,3)),
                    cell_rotations=np.tile(np.eye(3),(self.e,2,1,1)))
        return self._state(data)

    def _physical_pose_matches(self,actual,expected):
        return all(canonical(getattr(actual,name))==canonical(getattr(expected,name))
                   for name in ('positions','position_low','cell_rotations'))

    def _validate_initial_contact(self,state):
        pass

    def _admit_saved_state(self,state):
        pass

    def _guard(self,token=None,stage='connected.guard'):
        cancellation_safe_point(token,stage);self.analysis._guard()
        if self.started!=self._captured_started:raise ValueError('construction clock changed')
        if monotonic()-self._captured_started>120.:raise RuntimeError('connected native construction lifetime120s exhausted')
        if canonical(self._definition())!=self._binding or self.identity!=sha(self._definition()):
            raise ValueError('connected captured operator/routing/settings changed')
        if len(self.elements)!=len(self._captured_elements) or any(e is not held for e,held in zip(self.elements,self._captured_elements)) or any(e is not held for e,held in zip(self.elements,self.analysis._elements)):
            raise ValueError('exact owned element object inventory changed')
        for e in self.elements:e._check(self.analysis.model.mesh)

    @contextmanager
    def _operation(self,token=None):
        if not self._lock.acquire(blocking=False):raise RuntimeError('connected concurrent use refused')
        saved=(self._current,self._visible,dict(self._issued),dict(self._candidates))
        try:
            with self.analysis._operation():
                self._guard(token);yield;self._guard(token);self._visible=self._current
        except BaseException:
            self._current,self._visible,self._issued,self._candidates=saved
            raise
        finally:self._lock.release()

    @property
    def current(self):return self._visible

    def _state(self,data):
        positions=_array(data['positions'],(self.n,3),'shared positions')
        low=_array(data['position_low'],(self.n,3),'shared low positions')
        for h,l in zip(positions.flat,low.flat):validate_pair(float(h),float(l))
        frames=_frames(data['nodal_frames'],self.n,'shared algebraic frames')
        cells_array=np.asarray(data['cell_rotations'])
        if cells_array.shape!=(self.e,2,3,3):
            raise ValueError('exact complete cell frame inventory required before reshape')
        cells=_frames(cells_array.reshape(2*self.e,3,3),2*self.e,'physical cells').reshape(self.e,2,3,3)
        resultants=_array(data['resultants'],(self.e,18),'retained resultants')
        velocity=_array(data['velocity'],(self.size,),'global physical velocity')
        acceleration=_array(data['acceleration'],(self.size,),'global physical acceleration')
        if np.any(velocity[list(self.algebraic)+list(self.fixed)]) or np.any(acceleration[list(self.algebraic)+list(self.fixed)]):
            raise ValueError('no algebraic/fixed inertia or rates')
        if type(data['time']) is not float or not np.isfinite(data['time']) or data['time']<0 or type(data['epoch']) is not int or data['epoch']<0:
            raise ValueError('finite time/nonnegative exact epoch required')
        if data['identity']!=self.identity or canonical(data['histories'])!=canonical(self.histories):
            raise ValueError('connected identity/virgin histories mismatch')
        if self.fixed:
            if not np.array_equal(positions[0],self.reference_positions[0]) or np.any(low[0]) or not np.array_equal(frames[0],self.reference_frames[0]):
                raise ValueError('homogeneous clamp changed')
        return ConnectedState(*map(_owned,(positions,low,frames,cells,resultants,velocity,acceleration)),
            data['time'],data['epoch'],self.identity,self.histories)

    def _data(self,state):return {name:getattr(state,name) for name in self.state_type.__dataclass_fields__}

    def _require(self,state):
        record=self._issued.get(id(state))
        if type(state) is not self.state_type or state is not self._current or record is None or record[0] is not state or record[1]!=canonical(state):
            raise ValueError('issued unmutated current connected state required')
        return sha(state)

    def _issue(self,state):
        self._issued={id(state):(state,canonical(state))};self._current=state;return state

    def _evaluate(self,state,token=None):
        residual=np.zeros(self.size);jacobian=np.zeros((self.size,self.size));mass=np.zeros_like(jacobian)
        dv=np.zeros_like(jacobian);dq=np.zeros_like(jacobian);inertia_force=np.zeros(self.size)
        energy=0.;kinetic=0.;linear=np.zeros(3);angular=np.zeros(3);linear_low=np.zeros(3);angular_low=np.zeros(3)
        for k,(e,mapping) in enumerate(zip(self.elements,self.maps)):
            self._guard(token,'connected.element');idx=list(mapping);nodes=[self.node_index[n] for n in e.node_ids]
            args=(state.positions[nodes],state.position_low[nodes],state.nodal_frames[nodes],state.cell_rotations[k],state.resultants[k])
            response=e.operator.evaluate(*args,origin=self.histories[k],check=lambda:self._guard(token))
            material=json.loads(response.material)
            if canonical(response.history)!=canonical(self.histories[k]) or any(r['branch']!='ELASTIC' or any(v!=0. for pair in r['dissipation'] for v in pair) for r in material['stations']):
                raise ValueError('strict elastic unchanged complete histories required')
            local_r=response.residual;local_j=spatial_jacobian(local_r,response.hessian+response.hessian_low)
            iv=evaluate_centered_finite_inertia(e.operator.reference,self.analysis._inertias[e.element_id],
                state.positions[nodes],state.position_low[nodes],state.cell_rotations[k],
                state.velocity[idx[:24]],state.acceleration[idx[:24]],order=self.inertia_order)
            residual[idx]+=local_r;jacobian[np.ix_(idx,idx)]+=local_j
            mass[np.ix_(idx,idx)]+=iv.mixed_mass;dv[np.ix_(idx,idx)]+=iv.mixed_velocity_derivative
            dq[np.ix_(idx,idx)]+=iv.mixed_configuration_derivative;inertia_force[idx]+=iv.mixed_inertia
            energy+=float(response.potential);kinetic+=iv.kinetic_energy
            for axis in range(3):
                linear[axis],linear_low[axis]=split_sum((float(linear[axis]),float(linear_low[axis]),float(iv.linear_momentum[axis]),float(iv.linear_momentum_low[axis])))
                angular[axis],angular_low[axis]=split_sum((float(angular[axis]),float(angular_low[axis]),float(iv.angular_momentum[axis]),float(iv.angular_momentum_low[axis])))
        values=(residual,jacobian,mass,dv,dq,inertia_force,linear,linear_low,angular,angular_low,np.array((energy,kinetic,energy+kinetic)))
        if any(not np.isfinite(value).all() for value in values):raise ValueError('nonfinite complete connected aggregate')
        if np.any(mass[list(self.algebraic),:]) or np.any(mass[:,list(self.algebraic)]):raise ValueError('algebraic mass contamination')
        return residual,jacobian,inertia_force,mass,dv,dq,dict(elastic_energy=energy,kinetic_energy=kinetic,total_energy=energy+kinetic,linear_momentum=linear,linear_momentum_low=linear_low,angular_momentum=angular,angular_momentum_low=angular_low,momentum_origin_high=(0.,0.,0.),momentum_origin_low=(0.,0.,0.))

    def _norm(self,r,indices):
        value=float(np.linalg.norm((r/self.scale)[list(indices)]))
        if not np.isfinite(value):raise ValueError('nonfinite connected residual norm')
        return value

    def _advance(self,state,step,factor=1.):
        d=self._data(state);high=np.array(state.positions,copy=True);low=np.array(state.position_low,copy=True)
        frames=np.array(state.nodal_frames,copy=True);cells=np.array(state.cell_rotations,copy=True)
        for n in range(self.n):
            for a in range(3):high[n,a],low[n,a]=split_sum((float(high[n,a]),float(low[n,a]),factor*float(step[6*n+a])))
            frames[n]=rotation(step[6*n+3:6*n+6])@frames[n]
        for k in range(self.e):
            for c in range(2):
                first=6*self.n+6*k+3*c;cells[k,c]=rotation(factor*step[first:first+3])@cells[k,c]
        d.update(positions=high,position_low=low,nodal_frames=frames,cell_rotations=cells,
            resultants=state.resultants+step[6*self.n+6*self.e:6*self.n+24*self.e].reshape(self.e,18))
        return self._state(d)

    def _consistent(self,state,token):
        z=list(self.z)
        for iteration in range(self.max_iterations+1):
            r,j,*_=self._evaluate(state,token);step=np.zeros(self.size)
            step[z]=np.linalg.solve(j[np.ix_(z,z)],-r[z]);norm=self._norm(r,z)
            if norm<=1e-11 and np.linalg.norm(step/self.scale)<=1e-11:return state
            if iteration==self.max_iterations:raise ValueError('connected algebraic iteration limit')
            for cut in range(self.max_backtracks+1):
                proposed=self._advance(state,step*2.**(-cut));rt,*_=self._evaluate(proposed,token)
                if self._norm(rt,z)<norm:state=proposed;break
            else:raise ValueError('connected algebraic backtracking failed')

    def _block_admission(self,j,m):
        """Admit the EXACT saved/candidate configuration without modifying it."""
        p=list(self.p);z=list(self.z)
        mp=m[np.ix_(p,p)]
        if np.linalg.norm(mp-mp.T)>64*np.finfo(float).eps*np.linalg.norm(mp):raise ValueError('global physical mass symmetry failed')
        np.linalg.cholesky((mp+mp.T)/2)
        singular=np.linalg.svd(j[np.ix_(z,z)],compute_uv=False)
        if not np.isfinite(singular).all() or singular[-1]<=64*np.finfo(float).eps*len(z)*singular[0]:
            raise ValueError('assembled free algebraic C_z rank failed')

    def _endpoint(self,state,token):
        state=self._consistent(state,token);r,j,fi,m,*_=self._evaluate(state,token);p=list(self.p);z=list(self.z)
        self._block_admission(j,m)
        acceleration=np.zeros(self.size);acceleration[p]=np.linalg.solve(m[np.ix_(p,p)],-(r+fi)[p])
        data=self._data(state);data['acceleration']=acceleration;state=self._state(data)
        r,_,fi,*tail=self._evaluate(state,token)
        if self._norm(r+fi,p)>1e-11:raise ValueError('connected endpoint acceleration residual')
        return state,tail[-1]

    def initialize(self, *, physical_velocity=None, cancellation_token=None):
        with self._operation(cancellation_token):
            return self._initialize_state(physical_velocity,cancellation_token)

    def _initialize_state(self,physical_velocity,cancellation_token,physical_displacement=None):
        if self._current is not None:raise ValueError('fresh connected owner required')
        v=np.zeros(self.size)
        if physical_velocity is not None:v[list(self.physical)]=_array(physical_velocity,(len(self.physical),),'connected physical velocity')
        state=self._state(dict(positions=self.reference_positions,position_low=np.zeros((self.n,3)),
            nodal_frames=self.reference_frames,cell_rotations=np.tile(np.eye(3),(self.e,2,1,1)),resultants=np.zeros((self.e,18)),
            velocity=v,acceleration=np.zeros(self.size),time=0.,epoch=0,identity=self.identity,histories=self.histories))
        if physical_displacement is not None:
            step=np.zeros(self.size);step[list(self.physical)]=_array(physical_displacement,(len(self.physical),),'connected physical reference displacement')
            state=self._advance(state,step)
        state,_=self._endpoint(state,cancellation_token);canonical(state);return self._issue(state)

    def _midpoint_chart(self,increment,u,dt):
        p=list(self.p)
        chart=np.eye(self.size)
        for n in range(self.n):
            chart[6*n:6*n+3,6*n:6*n+3]*=.5
            eta=u[6*n+3:6*n+6];chart[6*n+3:6*n+6,6*n+3:6*n+6]=exp_chart_terms(eta)[0]
        for k in range(self.e):
            for c in range(2):
                first=6*self.n+6*k+3*c;half=.5*increment[first:first+3]
                chart[first:first+3,first:first+3]=.5*exp_chart_terms(half)[0]
        chart[:,p]*=.5*dt*dt
        return chart

    def _stage(self,accepted,u,dt,token):
        p=list(self.p);increment=np.zeros(self.size);increment[p]=dt*accepted.velocity[p]+.5*dt*dt*u[p]
        increment[list(self.z)]=u[list(self.z)]
        stage=self._advance(accepted,increment,.5)
        d=self._data(stage);v=np.zeros(self.size);a=np.zeros(self.size)
        v[p]=accepted.velocity[p]+.5*dt*u[p];a[p]=u[p];d.update(velocity=v,acceleration=a,time=float(accepted.time+.5*dt));stage=self._state(d)
        chart=self._midpoint_chart(increment,u,dt)
        r,j,fi,m,dv,dq,account=self._evaluate(stage,token)
        vmap=np.zeros_like(m);amap=np.zeros_like(m);vmap[p,p]=.5*dt;amap[p,p]=1.
        tangent=(j+dq)@chart+dv@vmap+m@amap
        if not np.isfinite(tangent).all() or not np.isfinite(r+fi).all():raise ValueError('nonfinite complete stage chart')
        return stage,r+fi,tangent,account

    def trial(self,accepted, *, dt,cancellation_token=None):
        with self._operation(cancellation_token):
            return self._trial_state(accepted,dt,cancellation_token)

    def _trial_state(self,accepted,dt,cancellation_token):
        previous=self._require(accepted)
        if type(dt) is not float or not np.isfinite(dt) or dt<=0 or dt*dt==0. or not np.isfinite(dt*dt) or not np.isfinite(2/(dt*dt)):
            raise ValueError('finite positive representable midpoint dt required')
        if not np.isfinite(accepted.time+dt) or accepted.time+dt<=accepted.time:
            raise ValueError('connected endpoint time increment not representable')
        if self._candidates:raise ValueError('discard or commit pending candidate first')
        p=list(self.p);free=list(self.free);u=np.zeros(self.size);u[p]=accepted.acceleration[p];backtracks=0
        for iteration in range(self.max_iterations+1):
            stage,r,j,_=self._stage(accepted,u,dt,cancellation_token)
            step=np.zeros(self.size);step[free]=np.linalg.solve(j[np.ix_(free,free)],-r[free])
            correction=step.copy();correction[p]*=.5*dt*dt
            if self._norm(r,free)<=1e-11 and np.linalg.norm(correction/self.scale)<=1e-11:break
            if iteration==self.max_iterations:raise ValueError('connected midpoint iteration limit')
            for cut in range(self.max_backtracks+1):
                next_u=u+step*2.**(-cut);_,rt,_,_=self._stage(accepted,next_u,dt,cancellation_token)
                if self._norm(rt,free)<self._norm(r,free):u=next_u;backtracks+=int(cut>0);break
            else:raise ValueError('connected midpoint backtracking failed')
        increment=np.zeros(self.size);increment[p]=dt*accepted.velocity[p]+.5*dt*dt*u[p]
        endpoint=self._advance(accepted,increment);data=self._data(endpoint)
        velocity=np.zeros(self.size);velocity[p]=accepted.velocity[p]+dt*u[p]
        data.update(velocity=velocity,acceleration=np.zeros(self.size),time=float(accepted.time+dt),epoch=accepted.epoch+1,
            nodal_frames=stage.nodal_frames,resultants=stage.resultants)
        endpoint,account=self._endpoint(self._state(data),cancellation_token)
        candidate=ConnectedCandidate(previous,endpoint,stage,_owned(r),_owned(j),account,dt,_owned(u),iteration,backtracks)
        self._candidates[id(candidate)]=(candidate,canonical(candidate));return candidate

    def _candidate(self,candidate):
        record=self._candidates.get(id(candidate))
        if type(candidate) is not ConnectedCandidate or record is None or record[0] is not candidate or record[1]!=canonical(candidate) or candidate.predecessor!=self._require(self._current):
            raise ValueError('authentic current connected candidate required')

    def commit(self,candidate, *, cancellation_token=None):
        with self._operation(cancellation_token):
            self._candidate(candidate);state=self._state(self._data(candidate.endpoint))
            self._issue(candidate.endpoint);self._candidates.clear();return candidate.endpoint

    def discard(self,candidate):
        with self._operation():self._candidate(candidate);del self._candidates[id(candidate)]

    def checkpoint(self):
        with self._operation():
            self._require(self._current)
            if self._candidates:raise ValueError('pending candidate cannot be checkpointed')
            return self._encode_state(self._current)

    def _encode_state(self,state):
        body=dict(schema=self.checkpoint_schema,profile=self.route_profile,identity=self.identity,state=state)
        raw=canonical(dict(body=body,sha256=sha(body)))
        if len(raw)>MAX_BYTES:raise ValueError('connected checkpoint byte cap')
        return raw

    def restore_checkpoint(self,raw, *, expected_sha256,cancellation_token=None):
        with self._operation(cancellation_token):
            return self._restore_state(raw,expected_sha256,cancellation_token)

    def _restore_state(self,raw,expected_sha256,cancellation_token):
        if self._current is not None:raise ValueError('fresh connected restore owner required')
        if type(raw) is not bytes or len(raw)>MAX_BYTES or sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('bounded trusted connected capsule required')
        # Bound lexical nesting before JSON; duplicate keys are never repaired.
        depth=0;quoted=False;escape=False
        for byte in raw:
            if quoted:
                if escape:escape=False
                elif byte==92:escape=True
                elif byte==34:quoted=False
            elif byte==34:quoted=True
            elif byte in (91,123):
                depth+=1
                if depth>32:raise ValueError('connected capsule nesting cap')
            elif byte in (93,125):depth-=1
        def pairs(rows):
            result={}
            for key,value in rows:
                if key in result:raise ValueError('duplicate connected JSON key')
                result[key]=value
            return result
        data=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite capsule')))
        if canonical(data)!=raw or set(data)!= {'body','sha256'}:raise ValueError('exact canonical connected envelope required')
        body=data['body']
        if set(body)!= {'schema','profile','identity','state'} or body['schema']!=self.checkpoint_schema or body['profile']!=self.route_profile or body['identity']!=self.identity or data['sha256']!=sha(body):
            raise ValueError('connected checkpoint identity/integrity mismatch')
        if type(body['state']) is not dict or set(body['state'])!=set(self.state_type.__dataclass_fields__):raise ValueError('complete state inventory required')
        state=self._state(body['state']);r,j,fi,m,_,_,_=self._evaluate(state,cancellation_token)
        self._block_admission(j,m)
        if canonical(self._data(state))!=canonical(body['state']):raise ValueError('restore must preserve exact stored numeric inventory')
        if self._norm(r,self.z)>1e-11 or self._norm(r+fi,self.p)>1e-11:raise ValueError('restored complete connected equilibrium invalid')
        self._admit_saved_state(state)
        return self._issue(state)

    def _prepare_initial(self,displacement,velocity, *, cancellation_token,prepare):
        with self._operation(cancellation_token):
            state=self._initialize_state(velocity,cancellation_token,displacement)
            _,_,_,_,_,_,account=self._evaluate(state,cancellation_token)
            native=self._encode_state(state);prepared=prepare(state,account,native)
            self._require(state);self._guard(cancellation_token,'connected.origin.ready')
            return state,prepared

    def _prepare_restore(self,raw, *, expected_sha256,cancellation_token,prepare):
        with self._operation(cancellation_token):
            state=self._restore_state(raw,expected_sha256,cancellation_token)
            _,_,_,_,_,_,account=self._evaluate(state,cancellation_token)
            prepared=prepare(state,account,raw)
            self._require(state);self._guard(cancellation_token,'connected.restore.ready')
            return state,prepared

    def _prepare_replayed_restore(self, *, cancellation_token,replay,prepare):
        """Trusted trajectory replay and native restoration share one atomic lease."""
        with self._operation(cancellation_token):
            controller,raw=replay()
            state=self._restore_state(raw,sha256(raw).hexdigest(),cancellation_token)
            _,_,_,_,_,_,account=self._evaluate(state,cancellation_token)
            prepared=prepare(state,account,raw,controller)
            self._require(state);self._guard(cancellation_token,'connected.restore.ready')
            return state,prepared

    def _prepare_trial(self,state, *, dt,cancellation_token,prepare):
        with self._operation(cancellation_token):
            candidate=self._trial_state(state,dt,cancellation_token)
            prepare(candidate);self._candidate(candidate)
            self._guard(cancellation_token,'connected.receipt.ready')
            return candidate

    def _discard_prepared_trial(self,candidate):
        with self._operation():
            if type(candidate) is not ConnectedCandidate:raise ValueError('exact connected receipt required')
            if id(candidate) not in self._candidates:return
            self._candidate(candidate);del self._candidates[id(candidate)]

    def _prepare_candidate(self,candidate, *, cancellation_token):
        with self._operation(cancellation_token):
            self._candidate(candidate)
            state=self._state(self._data(candidate.endpoint))
            native=self._encode_state(state)
            self._candidate(candidate);self._guard(cancellation_token,'connected.candidate.ready')
            return native


class ConnectedDynamicAnalysis(_RetainedMidpointAssembly):
    """Exact historical connected admission and profile over shared mechanics."""
