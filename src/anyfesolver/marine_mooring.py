"""Sparse body/node mooring assembly. Development mechanics, not W6 acceptance.

Free nodes have three translation coordinates after the body chart coordinates.
Anchors and fairleads introduce no duplicate endpoint DOFs. Inputs are immutable;
evaluation uses node arrays, with no state object allocated per line node.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import math
import numpy as np
from scipy import sparse

from .marine_rigid import RigidBody6D, RigidBodyState, CHART_BRANCH_MARGIN

BODY_DERIVATIVE_STEP = 2e-6
API_VERSION = 'marine_mooring_v1'
HYDRO_API_VERSION = 'marine_mooring_hydrodynamics_v1'
CONTACT_API_VERSION = 'marine_mooring_contact_v1'
__all__ = ['API_VERSION', 'HYDRO_API_VERSION', 'CONTACT_API_VERSION', 'MooringLineError', 'FixedEndpoint',
           'RigidBodyAttachmentEndpoint', 'DynamicLineNodeEndpoint',
           'MooringNodeState', 'MooringLine', 'MooringAssembly', 'MooringLineAssembler',
           'data_to_dict', 'data_from_dict']


def _indices(values):
    """Compiled indices backed by immutable bytes."""
    return np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


class MooringLineError(ValueError):
    """Invalid line data or unsupported line state; no silent fallback."""


def _array(value, shape, name, positive=False, nonnegative=False):
    try:
        if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
            raise ValueError()
        raw = np.asarray(value)
        if raw.shape != shape or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
            raise ValueError()
        a = np.asarray(raw, dtype=float)
        if (positive and np.any(a <= 0.)) or (nonnegative and np.any(a < 0.)):
            raise ValueError()
        return np.frombuffer(a.tobytes(), dtype=float).reshape(shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise MooringLineError(f'{name} requires finite numeric data of shape {shape}') from error


@dataclass(frozen=True)
class FixedEndpoint:
    position: tuple

    def __post_init__(self):
        object.__setattr__(self, 'position', tuple(_array(self.position, (3,), 'anchor position')))


@dataclass(frozen=True)
class RigidBodyAttachmentEndpoint:
    body_id: str
    point_body: tuple

    def __post_init__(self):
        if not isinstance(self.body_id, str) or not self.body_id.strip():
            raise MooringLineError('fairlead body identity required')
        object.__setattr__(self, 'point_body', tuple(_array(self.point_body, (3,), 'fairlead point')))


@dataclass(frozen=True)
class DynamicLineNodeEndpoint:
    node_index: int

    def __post_init__(self):
        if type(self.node_index) is not int or self.node_index < 0:
            raise MooringLineError('free node index must be a nonnegative integer')


@dataclass(frozen=True)
class MooringNodeState:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    def __post_init__(self):
        shape = np.shape(self.positions)
        if len(shape) != 2 or shape[1] != 3:
            raise MooringLineError('node positions require an N by 3 array')
        for name in ('positions', 'velocities', 'accelerations'):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))


@dataclass(frozen=True)
class MooringLine:
    line_id: str
    endpoints: tuple
    rest_lengths: np.ndarray
    axial_rigidity: np.ndarray
    axial_damping: np.ndarray
    mass_per_length: np.ndarray
    displaced_area: np.ndarray
    hydrodynamics: object = None
    contact_diameters: object = None

    def __post_init__(self):
        if not isinstance(self.line_id, str) or not self.line_id.strip():
            raise MooringLineError('line identity required')
        endpoints = tuple(self.endpoints)
        if len(endpoints) < 2 or any(type(e) not in (FixedEndpoint, RigidBodyAttachmentEndpoint, DynamicLineNodeEndpoint) for e in endpoints):
            raise MooringLineError('line requires named endpoint records')
        if any(type(e) is not DynamicLineNodeEndpoint for e in endpoints[1:-1]):
            raise MooringLineError('interior endpoints must be free line nodes')
        if any(type(e) is DynamicLineNodeEndpoint for e in (endpoints[0], endpoints[-1])):
            raise MooringLineError('line terminals must be anchors or body fairleads')
        if endpoints[0] == endpoints[-1]:
            raise MooringLineError('line terminals must be distinct')
        object.__setattr__(self, 'endpoints', endpoints)
        shape = (len(endpoints)-1,)
        for name in ('rest_lengths', 'axial_rigidity', 'mass_per_length'):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name, positive=True))
        for name in ('axial_damping', 'displaced_area'):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name, nonnegative=True))
        if self.hydrodynamics is not None:
            try:
                from anyloads.marine_line import LineHydrodynamics
                sections=tuple(self.hydrodynamics)
                if len(sections)!=shape[0] or any(type(s) is not LineHydrodynamics for s in sections):
                    raise ValueError('one named hydrodynamic section per segment required')
            except (ImportError,TypeError,ValueError) as error:
                raise MooringLineError('invalid or unavailable line hydrodynamics: '+str(error)) from error
            object.__setattr__(self,'hydrodynamics',sections)
        if self.contact_diameters is not None:
            diameters=_array(self.contact_diameters,shape,'contact diameters',positive=True)
            if self.hydrodynamics is not None and any(d!=s.diameter for d,s in zip(diameters,self.hydrodynamics)):
                raise MooringLineError('contact and hydrodynamic diameters must agree')
            object.__setattr__(self,'contact_diameters',diameters)


@dataclass(frozen=True)
class MooringAssembly:
    """Fresh result arrays/matrices may be edited without changing the model."""
    force: np.ndarray
    stiffness: object
    damping: object
    mass: object
    tensions: np.ndarray
    elastic_tensions: np.ndarray
    lengths: np.ndarray
    elastic_energy: float
    kinetic_energy: float
    damping_dissipation: float
    endpoint_forces: np.ndarray
    anchor_reactions: np.ndarray
    force_components: Mapping
    body_derivative_steps: Mapping
    relative_drag_power: float = 0.
    added_mass_kinetic_energy: float = 0.
    contact_diagnostics: object = None


@dataclass(frozen=True)
class MooringLineAssembler:
    """Stateless sparse segment assembler on a declared body+node layout.

    Reference material length controls mass/volume; volume is constant. Only
    fully submerged lines are admitted here. Optional Morison endpoint lumps
    use current segment axes and a public ANYloads environment. Optional compliant
    frictionless contact uses reference projected area and explicit surface gap.
    Body tangent columns use central differentiation of the public
    point map and geometric acceleration only; constitutive blocks and the
    acceleration derivative are analytic. No force probe crosses a segment
    branch or changes the segment geometry used by the constitutive tangent.
    """
    bodies: tuple
    lines: tuple
    node_count: int
    gravity_world: tuple
    fluid_density: float
    still_water_level: float
    environment: object = None
    seabed_contact: object = None

    def __post_init__(self):
        bodies, lines = tuple(self.bodies), tuple(self.lines)
        if any(type(b) is not RigidBody6D for b in bodies) or len({b.body_id for b in bodies}) != len(bodies):
            raise MooringLineError('unique public rigid bodies required')
        if not lines or any(type(l) is not MooringLine for l in lines) or len({l.line_id for l in lines}) != len(lines):
            raise MooringLineError('unique named mooring lines required')
        if type(self.node_count) is not int or self.node_count < 0:
            raise MooringLineError('node_count must be a nonnegative integer')
        for name in ('fluid_density', 'still_water_level'):
            val = getattr(self, name)
            if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float)) or not math.isfinite(val) or (name == 'fluid_density' and val <= 0.):
                raise MooringLineError(f'invalid {name}')
            object.__setattr__(self, name, float(val))
        object.__setattr__(self, 'gravity_world', _array(self.gravity_world, (3,), 'gravity'))
        object.__setattr__(self, 'bodies', bodies); object.__setattr__(self, 'lines', lines)
        ids = {b.body_id: i for i,b in enumerate(bodies)}
        endpoints=[]; starts=[]; ends=[]; coeff={k:[] for k in ('rest_lengths','axial_rigidity','axial_damping','mass_per_length','displaced_area')}
        for line in lines:
            first=len(endpoints); endpoints.extend(line.endpoints)
            starts.extend(range(first, first+len(line.endpoints)-1)); ends.extend(range(first+1, first+len(line.endpoints)))
            for k in coeff:coeff[k].extend(getattr(line,k))
        used=[]
        for ep in endpoints:
            if type(ep) is RigidBodyAttachmentEndpoint and ep.body_id not in ids:
                raise MooringLineError('fairlead references unknown body')
            if type(ep) is DynamicLineNodeEndpoint:used.append(ep.node_index)
        if set(used) != set(range(self.node_count)) or len(used) != len(set(used)):
            raise MooringLineError('each free node must occur exactly once in one line')
        object.__setattr__(self, '_endpoints', tuple(endpoints))
        object.__setattr__(self, '_starts', _indices(starts)); object.__setattr__(self, '_ends', _indices(ends))
        node_ep = _indices([i for i,e in enumerate(endpoints) if type(e) is DynamicLineNodeEndpoint])
        object.__setattr__(self, '_node_ep', node_ep)
        object.__setattr__(self, '_node_idx', _indices([endpoints[i].node_index for i in node_ep]))
        fixed = _indices([i for i,e in enumerate(endpoints) if type(e) is FixedEndpoint])
        object.__setattr__(self, '_fixed', fixed)
        object.__setattr__(self, '_fixed_positions', _array(
            np.asarray([endpoints[i].position for i in fixed]).reshape(-1,3), (len(fixed),3), 'anchors'))
        object.__setattr__(self, '_attachments', tuple(
            (i, ids[e.body_id], bodies[ids[e.body_id]], e) for i,e in enumerate(endpoints)
            if type(e) is RigidBodyAttachmentEndpoint))
        object.__setattr__(self, '_attached_body_indices', tuple(sorted({x[1] for x in self._attachments})))
        for k,v in coeff.items():object.__setattr__(self, '_'+k, _array(v,(len(starts),),k))
        sections=tuple(s for line in lines for s in
                       ((None,)*(len(line.endpoints)-1) if line.hydrodynamics is None else line.hydrodynamics))
        object.__setattr__(self,'_hydrodynamics',sections)
        diameters=tuple(d for line in lines for d in
                        ((None,)*(len(line.endpoints)-1) if line.contact_diameters is None else line.contact_diameters))
        object.__setattr__(self,'_contact_diameters',diameters)
        if any(d is not None for d in diameters)!=(self.seabed_contact is not None):
            raise MooringLineError('contact diameters and explicit seabed law must be provided together')
        if self.seabed_contact is not None:
            from .marine_seabed import SeabedContact
            if type(self.seabed_contact) is not SeabedContact or self.seabed_contact.elevation>=self.still_water_level:
                raise MooringLineError('public seabed law below still water required')
        if any(s is not None for s in sections)!=(self.environment is not None):
            raise MooringLineError('hydrodynamic sections and explicit environment must be provided together')
        if self.environment is not None:
            try:
                from anyloads.marine import AiryEnvironment
                if type(self.environment) is not AiryEnvironment:raise ValueError('public AiryEnvironment required')
                if self.environment.density!=self.fluid_density or self.environment.still_water_level!=self.still_water_level:
                    raise ValueError('environment and line density/still-water level must agree')
                if np.any(np.asarray(self.gravity_world)!=0.) and tuple(self.gravity_world)!=(0.,0.,-self.environment.gravity):
                    raise ValueError('selected line gravity must agree with the z-up wave environment')
                if self.seabed_contact is not None and self.seabed_contact.elevation!=self.still_water_level-self.environment.water_depth:
                    raise ValueError('seabed elevation must agree with environment water depth/datum')
            except (ImportError,TypeError,ValueError) as error:
                raise MooringLineError('invalid line environment: '+str(error)) from error
        lumped=np.zeros(len(endpoints)); weight=np.zeros(len(endpoints))
        half=.5*self._mass_per_length*self._rest_lengths
        net=.5*(self._mass_per_length-self.fluid_density*self._displaced_area)*self._rest_lengths
        np.add.at(lumped,self._starts,half);np.add.at(lumped,self._ends,half)
        np.add.at(weight,self._starts,net);np.add.at(weight,self._ends,net)
        object.__setattr__(self,'_lumped_mass',_array(lumped,lumped.shape,'lumped mass'))
        object.__setattr__(self,'_submerged_mass',_array(weight,weight.shape,'submerged mass'))

    @property
    def size(self):return 6*len(self.bodies)+3*self.node_count

    def _kinematics(self, body_states, nodes, *, admit=True):
        if not isinstance(body_states,Mapping) or set(body_states)!={b.body_id for b in self.bodies} or any(type(s) is not RigidBodyState for s in body_states.values()):
            raise MooringLineError('exact body state inventory required')
        if type(nodes) is not MooringNodeState or nodes.positions.shape!=(self.node_count,3):
            raise MooringLineError('exact immutable node state required')
        count=len(self._endpoints);p=np.zeros((count,3));v=p.copy();a=p.copy();rows=[];cols=[];data=[]
        node_ep=self._node_ep;node_idx=self._node_idx
        p[node_ep]=nodes.positions[node_idx];v[node_ep]=nodes.velocities[node_idx];a[node_ep]=nodes.accelerations[node_idx]
        rr=(3*node_ep[:,None]+np.arange(3)).ravel();cc=(6*len(self.bodies)+3*node_idx[:,None]+np.arange(3)).ravel()
        rows.extend(rr);cols.extend(cc);data.extend(np.ones(len(rr)))
        p[self._fixed]=self._fixed_positions
        for i,index,body,ep in self._attachments:
            kin=body.point_kinematics(body_states[ep.body_id],ep.point_body)
            p[i],v[i],a[i]=kin.position,kin.velocity,kin.acceleration
            rows.extend(np.repeat(3*i+np.arange(3),6));cols.extend(np.tile(6*index+np.arange(6),3));data.extend(kin.jacobian.ravel())
        if not all(np.isfinite(x).all() for x in (p,v,a,np.asarray(data))):
            raise MooringLineError('nonfinite endpoint kinematics')
        if admit and np.any(p[:,2]>self.still_water_level):
            raise MooringLineError('partial wetting is not admitted by the fully submerged line assembler')
        J=sparse.coo_matrix((data,(rows,cols)),shape=(3*count,self.size)).tocsr()
        return p,v,a,J

    def _hydro_values(self,p,v,a,n,length,time):
        """Sparse analytic endpoint partials at fixed endpoint kinematics."""
        from anyloads.marine_line import line_morison_force
        from anyloads.errors import LoadError
        count=len(p);components={name:np.zeros_like(p) for name in
                                 ('drag_normal','drag_tangential','fluid_inertia','added_mass')}
        rows=[];cols=[];kb=[];cb=[];mb=[];power=0.;energy=0.
        def block(i,j,K,C=None,M=None):
            rows.extend(np.repeat(3*i+np.arange(3),3));cols.extend(np.tile(3*j+np.arange(3),3))
            kb.extend(K.ravel());cb.extend((np.zeros((3,3)) if C is None else C).ravel())
            mb.extend((np.zeros((3,3)) if M is None else M).ravel())
        try:
            fluid=self.environment.evaluate(p,time)
            for segment,section in enumerate(self._hydrodynamics):
                if section is None:continue
                ends=(self._starts[segment],self._ends[segment]);idx=np.asarray(ends)
                f=line_morison_force(section,self.fluid_density,.5*length[segment],
                    .5*self._rest_lengths[segment]*self._displaced_area[segment],
                    fluid.velocity[idx],fluid.acceleration[idx],v[idx],a[idx],n[segment])
                for name,value in f.components.items():np.add.at(components[name],idx,value)
                power+=float(np.sum(f.relative_drag_power))
                energy+=float(-.5*np.einsum('ni,nij,nj->',v[idx],f.structure_acceleration_derivative,v[idx]))
                axis_map=(np.eye(3)-np.outer(n[segment],n[segment]))/length[segment]
                for j,i in enumerate(ends):
                    position=f.fluid_velocity_derivative[j]@fluid.velocity_gradient[i]+f.fluid_acceleration_derivative[j]@fluid.acceleration_gradient[i]
                    axis=f.axis_derivative[j]@axis_map+.5*np.outer(f.length_derivative[j],n[segment])
                    block(i,i,-position,-f.structure_velocity_derivative[j],-f.structure_acceleration_derivative[j])
                    block(i,ends[0],axis);block(i,ends[1],-axis)
        except (LoadError,ValueError,OverflowError) as error:
            raise MooringLineError('line hydrodynamic domain: '+str(error)) from error
        matrices=tuple(sparse.coo_matrix((values,(rows,cols)),shape=(3*count,3*count)).tocsr() for values in (kb,cb,mb))
        return components,matrices,power,energy

    def _contact_values(self,p,v):
        from .marine_seabed import seabed_contact_force,SeabedContactError
        components={name:np.zeros_like(p) for name in ('contact_elastic','contact_damping')}
        kz=np.zeros(len(p));cz=kz.copy();gaps=[];active=[];segments=[];endpoints=[]
        energy=0.;dissipation=0.
        try:
            for segment,diameter in enumerate(self._contact_diameters):
                if diameter is None:continue
                idx=np.array([self._starts[segment],self._ends[segment]])
                response=seabed_contact_force(self.seabed_contact,diameter,.5*self._rest_lengths[segment],p[idx],v[idx])
                np.add.at(components['contact_elastic'],idx,response.elastic_force)
                np.add.at(components['contact_damping'],idx,response.damping_force)
                np.add.at(kz,idx,response.stiffness[:,2,2]);np.add.at(cz,idx,response.damping[:,2,2])
                energy+=float(response.potential_energy.sum());dissipation+=float(response.dissipation.sum())
                gaps.extend(response.gaps.tolist());active.extend(response.active.tolist())
                segments.extend((segment,segment));endpoints.extend(idx.tolist())
        except SeabedContactError as error:raise MooringLineError('seabed contact domain: '+str(error)) from error
        K=np.zeros((len(p),3));C=K.copy();K[:,2]=kz;C[:,2]=cz
        reaction=-np.sum(components['contact_elastic']+components['contact_damping'],axis=0)
        diagnostics=dict(segment_indices=segments,endpoint_indices=endpoints,gaps=gaps,active=active,
            max_penetration=max(0.,-min(gaps)),potential_energy=energy,dissipation=dissipation,seabed_reaction=reaction)
        if not all(np.isfinite(x).all() for x in (kz,cz,energy,dissipation,reaction)):
            raise MooringLineError('nonfinite assembled contact tangent/energy')
        return components,(sparse.diags(K.ravel()),sparse.diags(C.ravel())),diagnostics

    def _values(self, body_states, nodes, time):
        p,v,a,J=self._kinematics(body_states,nodes)
        delta=p[self._ends]-p[self._starts];length=np.linalg.norm(delta,axis=1)
        if np.any(length<=0.) or not np.isfinite(length).all():raise MooringLineError('collapsed or nonfinite segment')
        n=delta/length[:,None];relative_v=v[self._ends]-v[self._starts];rate=np.einsum('ij,ij->i',n,relative_v)
        stretch=np.maximum(length-self._rest_lengths,0.)
        elastic=self._axial_rigidity*stretch/self._rest_lengths
        c=np.where(stretch>0.,self._axial_damping,0.)
        raw=elastic+c*rate;tension=np.maximum(raw,0.);active=(stretch>0.)&(raw>0.)
        elastic_force=np.zeros_like(p);damping_force=np.zeros_like(p)
        for f,magnitude in ((elastic_force,elastic),(damping_force,tension-elastic)):
            segment_f=magnitude[:,None]*n
            np.add.at(f,self._starts,segment_f);np.add.at(f,self._ends,-segment_f)
        weight=self._submerged_mass[:,None]*self.gravity_world
        inertia=-self._lumped_mass[:,None]*a
        endpoint_force=elastic_force+damping_force+weight+inertia
        hydro=None
        if self.environment is not None:
            hydro=self._hydro_values(p,v,a,n,length,time)
            endpoint_force+=sum(hydro[0].values())
        contact=None
        if self.seabed_contact is not None:
            contact=self._contact_values(p,v)
            endpoint_force+=sum(contact[0].values())
        force=np.asarray(J.T@endpoint_force.ravel()).ravel()
        if not all(np.isfinite(x).all() for x in (force,endpoint_force,tension,elastic)):
            raise MooringLineError('nonfinite assembled line force')
        return dict(p=p,v=v,a=a,J=J,length=length,n=n,rate=rate,relative_v=relative_v,
                    stretch=stretch,elastic=elastic,c=c,tension=tension,active=active,
                    elastic_force=elastic_force,damping_force=damping_force,
                    weight=weight,inertia=inertia,endpoint_force=endpoint_force,force=force,hydro=hydro,contact=contact)

    def force(self, body_states, nodes, *, time=0.):
        """Fresh generalized force, without tangent work; no state mutation."""
        return self._values(body_states,nodes,time)['force']

    def _segment_matrix(self, blocks):
        starts,ends=self._starts,self._ends;rows=[];cols=[];values=[]
        for aa,bb,sign in ((starts,starts,1.),(ends,ends,1.),(starts,ends,-1.),(ends,starts,-1.)):
            rows.append(np.repeat(3*aa[:,None]+np.arange(3),3,axis=1).ravel())
            cols.append(np.tile(3*bb[:,None]+np.arange(3),(1,3)).ravel());values.append((sign*blocks).ravel())
        count=3*len(self._endpoints)
        return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(count,count)).tocsr()

    def evaluate(self, body_states, nodes, *, time=0.):
        values=self._values(body_states,nodes,time);J=values['J'];n=values['n'];length=values['length']
        normal=n[:,:,None]*n[:,None,:];P=np.eye(3)[None,:,:]-normal
        gradient=self._axial_rigidity[:,None]/self._rest_lengths[:,None]*n+values['c'][:,None]/length[:,None]*np.einsum('ijk,ik->ij',P,values['relative_v'])
        T=values['tension'][:,None,None]/length[:,None,None]*P+n[:,:,None]*gradient[:,None,:]
        T=np.where(values['active'][:,None,None],T,0.)
        D=np.where(values['active'][:,None,None],values['c'][:,None,None]*normal,0.)
        if not all(np.isfinite(x).all() for x in (T,D)):
            raise MooringLineError('nonfinite constitutive tangent')
        endpoint_K=self._segment_matrix(T);endpoint_C=self._segment_matrix(D)
        endpoint_M=sparse.diags(np.repeat(self._lumped_mass,3))
        if values['hydro'] is not None:
            hK,hC,hM=values['hydro'][1]
            endpoint_K+=hK;endpoint_C+=hC;endpoint_M+=hM
        if values['contact'] is not None:
            cK,cC=values['contact'][1];endpoint_K+=cK;endpoint_C+=cC
        K=(J.T@endpoint_K@J).tolil();C=(J.T@endpoint_C@J).tolil()
        M=(J.T@endpoint_M@J).tocsr()
        # Include follower/pullback and geometric acceleration derivatives on
        # finite-rotation body columns. Differentiate only public point maps,
        # never clipped constitutive forces or segment lengths. The analytic
        # base-state segment blocks supply their selected branch exactly.
        derivative_steps={}
        for i in self._attached_body_indices:
            body=self.bodies[i]
            state=body_states[body.body_id]
            clearance=math.pi-CHART_BRANCH_MARGIN-np.linalg.norm(state.rotation_vector)
            rotation_step=min(BODY_DERIVATIVE_STEP,.25*clearance)
            if rotation_step<=64*np.finfo(float).eps*max(1.,np.linalg.norm(state.rotation_vector)):
                raise MooringLineError('body tangent differentiation domain exhausted near chart seam; rechart before assembly')
            derivative_steps[body.body_id]=(rotation_step,BODY_DERIVATIVE_STEP)
            for j in range(3,6):  # point maps are independent of r and r-dot
                for field,target in (('q',K),('v',C)):
                    h=rotation_step if field=='q' else BODY_DERIVATIVE_STEP
                    plus=dict(body_states);minus=dict(body_states)
                    if field=='q':
                        name='position' if j<3 else 'rotation_vector';k=j if j<3 else j-3
                    else:name='velocity';k=j
                    xp=np.array(getattr(state,name),copy=True);xm=xp.copy();xp[k]+=h;xm[k]-=h
                    plus[body.body_id]=replace(state,**{name:xp});minus[body.body_id]=replace(state,**{name:xm})
                    # Kinematic probes do not admit emerged public trial states.
                    _,vp,ap,Jp=self._kinematics(plus,nodes,admit=False)
                    _,vm,am,Jm=self._kinematics(minus,nodes,admit=False)
                    da=(ap-am).ravel()/(2*h)
                    correction=np.asarray(J.T@(endpoint_M@da)).ravel()
                    if field=='q':
                        dv=(vp-vm).ravel()/(2*h);dJ=(Jp-Jm)/(2*h)
                        correction+=np.asarray(J.T@(endpoint_C@dv)-dJ.T@values['endpoint_force'].ravel()).ravel()
                    target[:,6*i+j]=target[:,6*i+j]+correction[:,None]
        dissipation=float(np.sum((values['tension']-values['elastic'])*values['rate']))
        energy=float(np.sum(.5*self._axial_rigidity/self._rest_lengths*values['stretch']**2))
        kinetic=float(.5*np.sum(self._lumped_mass[:,None]*values['v']**2))
        components={k:np.asarray(J.T@values[k].ravel()).ravel() for k in ('elastic_force','damping_force','weight','inertia')}
        power=0.;added_energy=0.
        if values['hydro'] is not None:
            components.update({k:np.asarray(J.T@f.ravel()).ravel() for k,f in values['hydro'][0].items()})
            power,added_energy=values['hydro'][2:]
            kinetic+=added_energy
        contact_diagnostics=None
        if values['contact'] is not None:
            components.update({k:np.asarray(J.T@f.ravel()).ravel() for k,f in values['contact'][0].items()})
            contact_diagnostics=values['contact'][2]
        K=K.tocsr();C=C.tocsr()
        if not (all(np.isfinite(x).all() for x in (K.data,C.data,M.data,*components.values()))
                and all(math.isfinite(x) for x in (dissipation,energy,kinetic,power,added_energy))):
            raise MooringLineError('nonfinite assembly tangent or energy')
        return MooringAssembly(values['force'],K,C,M,values['tension'],values['elastic'],length,
                               energy,kinetic,dissipation,values['endpoint_force'],-values['endpoint_force'][self._fixed],components,
                               derivative_steps,power,added_energy,contact_diagnostics)


def data_to_dict(assembler):
    """Named immutable line data only; never executable provider code."""
    if type(assembler) is not MooringLineAssembler:
        raise MooringLineError('exact MooringLineAssembler required')
    lines=[]
    for line in assembler.lines:
        endpoints=[]
        for ep in line.endpoints:
            if type(ep) is FixedEndpoint:endpoints.append(dict(type='fixed',position=list(ep.position)))
            elif type(ep) is RigidBodyAttachmentEndpoint:endpoints.append(dict(type='rigid',body_id=ep.body_id,point_body=list(ep.point_body)))
            else:endpoints.append(dict(type='node',node_index=ep.node_index))
        item=dict(line_id=line.line_id,endpoints=endpoints)
        for name in ('rest_lengths','axial_rigidity','axial_damping','mass_per_length','displaced_area'):
            item[name]=getattr(line,name).tolist()
        if line.hydrodynamics is not None:item['hydrodynamics']=[s.to_dict() for s in line.hydrodynamics]
        if line.contact_diameters is not None:item['contact_diameters']=line.contact_diameters.tolist()
        lines.append(item)
    data=dict(api_version=API_VERSION,body_ids=[b.body_id for b in assembler.bodies],
                lines=lines,node_count=assembler.node_count,gravity_world=list(assembler.gravity_world),
                fluid_density=assembler.fluid_density,still_water_level=assembler.still_water_level)
    if assembler.environment is not None:
        from anyloads.marine import environment_to_dict
        data.update(api_version=HYDRO_API_VERSION,environment=environment_to_dict(assembler.environment))
    if assembler.seabed_contact is not None:
        data.update(api_version=CONTACT_API_VERSION,seabed_contact=assembler.seabed_contact.to_dict())
    return data


def data_from_dict(data,bodies):
    """Reconstruct exact schema against the supplied ordered body inventory."""
    fields={'api_version','body_ids','lines','node_count','gravity_world','fluid_density','still_water_level'}
    if isinstance(data,dict) and (data.get('api_version')==HYDRO_API_VERSION or
            (data.get('api_version')==CONTACT_API_VERSION and 'environment' in data)):fields.add('environment')
    if isinstance(data,dict) and data.get('api_version')==CONTACT_API_VERSION:fields.add('seabed_contact')
    if not isinstance(data,dict) or set(data)!=fields or data['api_version'] not in (API_VERSION,HYDRO_API_VERSION,CONTACT_API_VERSION):
        raise MooringLineError('named mooring schema required')
    try:bodies=tuple(bodies)
    except TypeError as error:raise MooringLineError('ordered rigid bodies required') from error
    if any(type(body) is not RigidBody6D for body in bodies):
        raise MooringLineError('ordered rigid bodies required')
    if data['body_ids']!=[b.body_id for b in bodies] or not isinstance(data['lines'],(list,tuple)):
        raise MooringLineError('mooring body order and line inventory must match')
    lines=[]
    coefficient_names=('rest_lengths','axial_rigidity','axial_damping','mass_per_length','displaced_area')
    try:
        for item in data['lines']:
            line_fields={'line_id','endpoints',*coefficient_names}
            if data['api_version'] in (HYDRO_API_VERSION,CONTACT_API_VERSION) and isinstance(item,dict) and 'hydrodynamics' in item:line_fields.add('hydrodynamics')
            if data['api_version']==CONTACT_API_VERSION and isinstance(item,dict) and 'contact_diameters' in item:line_fields.add('contact_diameters')
            if not isinstance(item,dict) or set(item)!=line_fields:
                raise MooringLineError('line fields must exactly match schema')
            if not isinstance(item['endpoints'],(list,tuple)):
                raise MooringLineError('endpoints must be named sequence')
            endpoints=[]
            for ep in item['endpoints']:
                if not isinstance(ep,dict):raise MooringLineError('named endpoint required')
                kind=ep.get('type')
                if kind=='fixed' and set(ep)=={'type','position'}:endpoints.append(FixedEndpoint(ep['position']))
                elif kind=='rigid' and set(ep)=={'type','body_id','point_body'}:endpoints.append(RigidBodyAttachmentEndpoint(ep['body_id'],ep['point_body']))
                elif kind=='node' and set(ep)=={'type','node_index'}:endpoints.append(DynamicLineNodeEndpoint(ep['node_index']))
                else:raise MooringLineError('unknown endpoint or fields refused')
            hydro=None
            if 'hydrodynamics' in item:
                from anyloads.marine_line import LineHydrodynamics
                if not isinstance(item['hydrodynamics'],(list,tuple)):raise MooringLineError('named hydrodynamic section sequence required')
                hydro=tuple(LineHydrodynamics.from_dict(s) for s in item['hydrodynamics'])
            lines.append(MooringLine(item['line_id'],endpoints,*(item[name] for name in coefficient_names),
                         hydrodynamics=hydro,contact_diameters=item.get('contact_diameters')))
        environment=None
        if 'environment' in data:
            from anyloads.marine import environment_from_dict
            environment=environment_from_dict(data['environment'])
        contact=None
        if data['api_version']==CONTACT_API_VERSION:
            from .marine_seabed import SeabedContact
            contact=SeabedContact.from_dict(data['seabed_contact'])
        return MooringLineAssembler(bodies,lines,data['node_count'],data['gravity_world'],data['fluid_density'],data['still_water_level'],environment,contact)
    except (TypeError,KeyError,ValueError) as error:
        raise MooringLineError(f'invalid mooring data: {error}') from error
