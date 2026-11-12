"""Development global native/S3 common-midpoint descriptor development owner.

Bodies share one retained solve. Shell structural triangles assemble once;
selected interior contact contributors never own independent states/steppers.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction as F
from math import nextafter, inf, sqrt
from types import SimpleNamespace
import numpy as np
import json
from anymaterial import IsotropicMaterial
from ._ge_beam3_connected_dynamic import ConnectedState, _RetainedMidpointAssembly, MAX_COORDINATES
from ._ge_beam3_native_shell_contact import NativeShellContactAnalysis, PATH_FAILURE, _line, _cross, _difference, _dot, _range, _absolute
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5.compensated_coordinates import split_sum
from ._ge_beam3_p5.arrays import _array
from ._native_reference_modal import _owned
from .control import cancellation_safe_point
from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from .general_contact.s3_skin_work import S3SkinWorkAdapter

PROFILE='NATIVE_MULTI_B3_GE_S3_SMALL_MOTION_DESCRIPTOR_MIDPOINT_DEV_V1'
SCHEMA='NATIVE_MULTI_B3_GE_S3_NATIVE_CHECKPOINT_DEV_V1'
TRANSITION='UNADMITTED_PATCH_TRANSITION_BAND'
BOUNDARY='UNADMITTED_SELECTED_PATCH_BOUNDARY'
RESOURCE='native multi-S3 aggregate operation resource admission exhausted'


@dataclass(frozen=True)
class NativeMultiShellState(ConnectedState):
    shell_displacement: np.ndarray


class AggregateWorkBudget:
    """One nonrenewable operation ledger; refusal precedes the next unit of work."""
    def __init__(self,geometry,integration,swept):
        self.caps=dict(geometry=geometry,integration=integration,swept=swept)
        self.used=dict.fromkeys(self.caps,0)
    def debit(self,kind):
        if self.used[kind]>=self.caps[kind]:
            error=MemoryError(RESOURCE+': '+kind)
            error.aggregate_work=dict(self.used);error.aggregate_caps=dict(self.caps)
            raise error
        self.used[kind]+=1


@dataclass(frozen=True)
class ContactContributor:
    selection: object
    work: object
    native_index: int
    shell_index: int
    mapping: tuple
    error_factor: float
    boundaries: tuple


def _outward_sum(values):
    exact=sum((F(float(v)) for v in values),F(0))
    result=float(exact)
    return nextafter(result,inf) if F(result)<exact else result


def _topology(model,skins):
    """Exact planar triangle interior exclusion; shared edges retain mesh identity."""
    first=skins[0];d=first.director
    points={n:tuple(F(float(x)) for x in node.coords()) for n,node in model.mesh.nodes.items()}
    origin=points[first.node_ids[0]];normal=tuple(F(float(x)) for x in d)
    if any(sum((p[i]-origin[i])*normal[i] for i in range(3))!=0 for p in points.values()):
        raise ValueError('exact planar shell body required')
    axis=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=axis]
    edges={};triangles=[]
    for skin in skins:
        if not np.array_equal(skin.director,d) or skin.physical_offset!=first.physical_offset:
            raise ValueError('common top director and physical offset per shell body required')
        triangle=[tuple(points[n][i] for i in axes) for n in skin.node_ids]
        triangles.append(triangle)
        for a,b in zip(skin.node_ids,skin.node_ids[1:]+skin.node_ids[:1]):
            edge=tuple(sorted((a,b)));edges[edge]=edges.get(edge,0)+1
    if any(count>2 for count in edges.values()):raise ValueError('nonmanifold shell edge')
    for i,a in enumerate(triangles):
        for b in triangles[i+1:]:
            separate=False
            for tri in (a,b):
                for p,q in zip(tri,tri[1:]+tri[:1]):
                    direction=(q[1]-p[1],p[0]-q[0])
                    aa=[sum(x*y for x,y in zip(v,direction)) for v in a]
                    bb=[sum(x*y for x,y in zip(v,direction)) for v in b]
                    if max(aa)<=min(bb) or max(bb)<=min(aa):separate=True
            if not separate:raise ValueError('overlapping or duplicate shell triangle interiors')
    return dict(physical_exterior_edges=tuple(sorted(e for e,c in edges.items() if c==1)),
                internal_edges=tuple(sorted(e for e,c in edges.items() if c==2)),
                internal_edge_semantics=TRANSITION)


class NativeMultiShellContactAnalysis(_RetainedMidpointAssembly):
    route_profile=PROFILE
    checkpoint_schema=SCHEMA
    state_type=NativeMultiShellState
    trajectory_profile='NATIVE_MULTI_B3_GE_S3_TRAJECTORY_DEV_V1'
    trajectory_schema='NATIVE_MULTI_B3_GE_S3_TRAJECTORY_CHECKPOINT_DEV_V1'
    contact_result_label='GLOBAL_NATIVE_MULTIPLE_S3_INTERIOR_CONTRIBUTORS'
    extra_step_keys=('native_multi_shell_contact',)
    retryable=_RetainedMidpointAssembly.retryable+(PATH_FAILURE,BOUNDARY)

    def __init__(self,native_bodies,shell_bodies,pairs,works,*,maximum_penetration_m,
                 maximum_geometry_evaluations,maximum_integration_intervals,maximum_swept_intervals,
                 maximum_shell_displacement_m,maximum_shell_rotation_rad,energy_tolerance_j,
                 force_tolerance_n,tangent_tolerance_n_per_m,inertia_order=24,max_iterations=12,max_backtracks=8):
        self.native_bodies=native_bodies;self.shell_bodies=shell_bodies;self.pairs=pairs;self.works=works
        if any(type(x) not in (int,float) or not np.isfinite(x) or x<=0 for x in
               (maximum_penetration_m,maximum_shell_displacement_m,maximum_shell_rotation_rad)):
            raise ValueError('positive finite penetration/small-motion limits required')
        if maximum_shell_rotation_rad>.05 or any(maximum_penetration_m>=w.contact.radius for w in works):
            raise ValueError('penetration below every radius and shell chart<=.05rad required')
        for cap in (maximum_geometry_evaluations,maximum_integration_intervals,maximum_swept_intervals):
            if type(cap) is not int or not 1<=cap<=4096:raise ValueError('aggregate work cap1..4096 required')
        self.maximum_penetration_m=float(maximum_penetration_m)
        self.maximum_shell_displacement_m=float(maximum_shell_displacement_m)
        self.maximum_shell_rotation_rad=float(maximum_shell_rotation_rad)
        self.maximum_geometry_evaluations=maximum_geometry_evaluations
        self.maximum_integration_intervals=maximum_integration_intervals
        self.maximum_swept_intervals=maximum_swept_intervals
        self.energy_tolerance_j=float(energy_tolerance_j);self.force_tolerance_n=float(force_tolerance_n)
        self.tangent_tolerance_n_per_m=float(tangent_tolerance_n_per_m)
        self._active_budget=None;self.last_operation_work=None
        self.shell_layout=[];offset=0
        for body in shell_bodies:
            model=body.model;size=model.mesh.dof_manager.total_dofs
            p=tuple(int(x) for n in sorted(model.mesh.nodes) for x in model.mesh.dof_manager.get_node_dofs(n)[:3])
            z=tuple(int(x) for n in sorted(model.mesh.nodes) for x in model.mesh.dof_manager.get_node_dofs(n)[3:6])
            if len(p)+len(z)!=size or len(set(p+z))!=size:raise ValueError('complete six-DOF shell nodes required')
            self.shell_layout.append((offset,size,p,z));offset+=size
        data=[json.loads(d.raw) for b in native_bodies for d in b.definitions]
        predicted=6*len({n for d in data for n in d['node_ids']})+24*len(data)+offset
        if predicted>MAX_COORDINATES:raise MemoryError('global retained coordinate budget exceeded before allocation')
        self._predicted_size=predicted
        self.shell_size=offset
        self._captured_models=tuple(b.model for b in shell_bodies)
        self._captured_shell_elements=tuple(tuple(b.model.mesh.elements.items()) for b in shell_bodies)
        self.shell_stiffness,self.shell_mass,self._shell_binding,self.shell_topologies=self._shell_contract()
        # Bind every inherited implementation descriptor, including helpers used
        # by the exact shell matrix routines; instance overrides are refused.
        self._shell_methods=tuple((cls,name,value) for cls in NativeParityE4PLS3V2DShellElement.__mro__
            for name,value in vars(cls).items() if callable(value) or isinstance(value,(staticmethod,classmethod,property)))
        self._shell_method_names=frozenset(name for _,name,_ in self._shell_methods)
        self._shell_class_keys=tuple((cls,frozenset(vars(cls))) for cls in NativeParityE4PLS3V2DShellElement.__mro__)
        self._shell_extra_binding=self._shell_configuration()
        definitions=tuple(d for body in native_bodies for d in body.definitions)
        super().__init__(definitions,(),inertia_order=inertia_order,max_iterations=max_iterations,max_backtracks=max_backtracks)
        self.native_size=self.size
        if self.native_size+self.shell_size>MAX_COORDINATES:raise MemoryError('global retained coordinate budget exceeded')
        self.size+=self.shell_size
        if self.size!=self._predicted_size:raise ValueError('authoritative coordinate inventory disagrees with preflight')
        self.shell_p=tuple(o+i for o,_,p,_ in self.shell_layout for i in p)
        self.shell_z=tuple(o+i for o,_,_,z in self.shell_layout for i in z)
        self.physical+=tuple(self.native_size+i for i in self.shell_p)
        self.algebraic+=tuple(self.native_size+i for i in self.shell_z)
        self.p=self.physical;self.z=self.algebraic;self.free=tuple(range(self.size))
        scale=np.r_[self.scale,np.ones(self.shell_size)];scale[[self.native_size+i for i in self.shell_z]]=self.length
        self.scale=_owned(scale)
        self.contributors=[]
        indices={e.element_id:i for i,e in enumerate(self.elements)}
        body_indices={b.body_id:i for i,b in enumerate(shell_bodies)}
        for pair,work in zip(pairs,works):
            k=indices[pair.native_element_id];bi=body_indices[pair.shell_body_id];o,size,_,_=self.shell_layout[bi]
            element=self.elements[k]
            if (element.node_ids!=work.contact.beam_node_ids or
                    sha(element.operator.reference.canonical_data())!=sha(work.contact.beam_reference.canonical_data())):
                raise ValueError('complete retained native contact reference mismatch')
            if any(np.any(element.operator.reference.half_cell_lift(c,.5)) for c in (0,1)):
                raise ValueError('native contact requires exact zero half-cell lift')
            factor=sqrt(27.)*max(4.,1.+2.*abs(work.shell.physical_offset)/self.length)
            factor=nextafter(factor,inf)
            mapping=tuple(self.maps[k])+tuple(self.native_size+o+i for i in range(size))
            topology=self.shell_topologies[bi];boundaries=[]
            ids=work.shell.node_ids
            for a,b in zip(ids,ids[1:]+ids[:1]):
                edge=tuple(sorted((a,b)))
                kind='PHYSICAL_EXTERIOR_RIM' if edge in topology['physical_exterior_edges'] else TRANSITION
                boundaries.append((edge,kind))
            self.contributors.append(ContactContributor(pair,work,k,bi,mapping,factor,tuple(boundaries)))
        self.contributors=tuple(self.contributors)
        self._binding=canonical(self._definition());self.identity=sha(self._definition())

    def _admit_element_count(self,count):
        if count<1:raise ValueError('at least one complete native element required')
    def _admit_graph(self):
        # Each explicit body is connected; distinct bodies remain distinct.
        by_id={e.element_id:e for e in self.elements}
        held=set()
        for body in self.native_bodies:
            elements=[by_id[json.loads(d.raw)['element_id']] for d in body.definitions]
            seen=set(elements[0].node_ids);pending=elements[1:]
            while pending:
                found=next((e for e in pending if seen.intersection(e.node_ids)),None)
                if found is None:raise ValueError('native body element graph disconnected')
                seen.update(found.node_ids);pending.remove(found)
            if seen.intersection(held):raise ValueError('native node alias between bodies')
            held.update(seen)

    def _shell_contract(self, *, assemble=True):
        matrices=[];masses=[];bindings=[];topologies=[]
        for body,(o,size,p,z) in zip(self.shell_bodies,self.shell_layout):
            model=body.model
            if (model.boundary_conditions or model.constraint_equations or model.load_cases or
                    model.mesh.point_masses or model.mesh.element_activity is not None or not model.mesh.elements):
                raise ValueError('free elastic shell body without loads/pointmass/activity required')
            used=set();k=np.zeros((size,size)) if assemble else None;m=np.zeros_like(k) if assemble else None;rows=[];skins=[]
            for eid,e in sorted(model.mesh.elements.items()):
                if (type(e) is not NativeParityE4PLS3V2DShellElement or e.shell_section is not None or
                        e.reference_surface_offset!=0. or e.director_polarity!=1):
                    raise ValueError('exact centered positive-polarity elastic S3 V2D body required')
                material=model.materials.get(e.material_name)
                if (type(material) is not IsotropicMaterial or material.yield_stress!=0. or
                        material.hardening_curve is not None or material.density<=0.):
                    raise ValueError('elastic positive-density shell material required')
                e.validate_quadrature_authority();e._validate_configuration()
                skin=S3SkinWorkAdapter.from_model(model,body.body_id,eid,'top');skins.append(skin)
                normal=np.cross(skin.reference_positions[1]-skin.reference_positions[0],skin.reference_positions[2]-skin.reference_positions[0])
                if skin.physical_offset<=0. or normal@skin.director<=0.:raise ValueError('oriented top skin required')
                slots=tuple(int(i) for i in e.get_dof_mapping(model.mesh));used.update(e.node_ids)
                if assemble:
                    k[np.ix_(slots,slots)]+=e.compute_stiffness_matrix(model.mesh,material)
                    m[np.ix_(slots,slots)]+=e.compute_mass_matrix(model.mesh,material)
                rows.append(dict(element=eid,nodes=e.node_ids,slots=slots,formulation=e.formulation_id,
                    coordinates=[model.mesh.nodes[n].coords() for n in e.node_ids],thickness=e.thickness,
                    reference_offset=e.reference_surface_offset,polarity=e.director_polarity,
                    director=skin.director,offset=skin.physical_offset,
                    material=dict(name=material.name,E=material.elastic_modulus,nu=material.poisson_ratio,density=material.density)))
            if used!=set(model.mesh.nodes):raise ValueError('all shell nodes must belong to structural S3 elements')
            if assemble and (not np.isfinite(k).all() or not np.isfinite(m).all() or not np.array_equal(m,m.T) or
                    np.any(m[list(z),:]) or np.any(m[:,list(z)]) or
                    np.linalg.norm(k-k.T)>1e-10*max(1.,np.linalg.norm(k))):
                raise ValueError('actual shell stiffness/mass witness invalid')
            if assemble:np.linalg.cholesky(m[np.ix_(p,p)])
            topology=_topology(model,skins);topologies.append(topology)
            matrices.append(k);masses.append(m);bindings.append(dict(body=body.body_id,size=size,p=p,z=z,rows=rows,topology=topology))
        if not assemble:return None,None,canonical(bindings),tuple(topologies)
        total=sum(x[1] for x in self.shell_layout);k=np.zeros((total,total));m=np.zeros_like(k)
        for (o,size,_,_),kk,mm in zip(self.shell_layout,matrices,masses):
            k[o:o+size,o:o+size]=kk;m[o:o+size,o:o+size]=mm
        return _owned(k),_owned(m),canonical(bindings),tuple(topologies)

    def _definition(self):
        data=super()._definition()
        data.update(shell_contract=self._shell_binding.decode('ascii'),shell_layout=self.shell_layout,
            shell_stiffness=self.shell_stiffness,shell_mass=self.shell_mass,
            native_bodies=[(b.body_id,[d.raw.decode('ascii') for d in b.definitions]) for b in self.native_bodies],
            pairs=self.pairs,contacts=[w._fingerprint() for w in self.works],
            contributors=[(c.mapping,c.error_factor,c.boundaries) for c in getattr(self,'contributors',())],
            maximum_penetration_m=self.maximum_penetration_m,maximum_shell_displacement_m=self.maximum_shell_displacement_m,
            maximum_shell_rotation_rad=self.maximum_shell_rotation_rad,
            aggregate_limits=(self.maximum_geometry_evaluations,self.maximum_integration_intervals,self.maximum_swept_intervals),
            aggregate_limit_basis='ONE_ATOMIC_OWNER_OPERATION_ALL_PAIRS_AND_BOTH_PATH_LEGS',
            error_limits=(self.energy_tolerance_j,self.force_tolerance_n,self.tangent_tolerance_n_per_m),
            error_chart='METRE_TRANSLATIONS_LENGTH_TIMES_ROTATIONS_POTENTIAL_PULLBACK',
            shell_rates='ZERO_ALGEBRAIC_PLACEHOLDERS_NOT_SKIN_VELOCITY',
            force_scale_n=self.force_tolerance_n,moment_scale_nm=self.length*self.force_tolerance_n,
            tangent_scale_n_per_m=self.tangent_tolerance_n_per_m,dimensioned_rotation_length_m=self.length,
            path='NATIVE_AFFINE_EXP_SHELL_ACTUAL_STAGE_AND_PROJECTED_ENDPOINT_TWO_LEGS')
        return data

    def _shell_configuration(self):
        # Bind the complete authoritative inputs of skin/map/topology/K/M
        # preparation. No sparse work-map or derived topology is rebuilt here.
        bodies=[]
        for body,(offset,size,p,z) in zip(self.shell_bodies,self.shell_layout):
            model=body.model
            if (model.boundary_conditions or model.constraint_equations or model.load_cases or
                    model.mesh.point_masses or model.mesh.element_activity is not None or not model.mesh.elements):
                raise ValueError('free elastic shell body without loads/pointmass/activity required')
            nodes=[(n,model.mesh.nodes[n].coords(),tuple(model.mesh.dof_manager.get_node_dofs(n)))
                for n in sorted(model.mesh.nodes)]
            elements=[]
            for eid,e in sorted(model.mesh.elements.items()):
                if (type(e) is not NativeParityE4PLS3V2DShellElement or e.shell_section is not None or
                        e.reference_surface_offset!=0. or e.director_polarity!=1):
                    raise ValueError('exact centered positive-polarity elastic S3 V2D body required')
                e.validate_quadrature_authority();e._validate_configuration()
                material=model.materials.get(e.material_name)
                if (type(material) is not IsotropicMaterial or material.yield_stress!=0. or
                        material.hardening_curve is not None or material.density<=0.):
                    raise ValueError('elastic positive-density shell material required')
                elements.append(dict(id=eid,element_id=e.element_id,nodes=e.node_ids,
                    slots=tuple(int(i) for i in e.get_dof_mapping(model.mesh)),
                    formulation=e.formulation_id,thickness=e.thickness,reference_normal=e.reference_normal,
                    reference_offset=e.reference_surface_offset,polarity=e.director_polarity,
                    director=e.physical_reference_director,offset=e.material_surface_offset_from_reference(1.),
                    material_name=e.material_name,material_direction=e.material_direction,material_angle=e.material_angle_deg,
                    drilling=e.drilling_stabilization,hourglass=e.hourglass_stabilization,reduced=e.reduced_integration,
                    quadrature_authority=e.quadrature_authority_id,gauss_points=e.gauss_points,gauss_weights=e.gauss_weights,
                    shear_points=e.shear_gauss_points,shear_weights=e.shear_gauss_weights,
                    material=(material.name,material.elastic_modulus,material.poisson_ratio,material.density,
                        material.yield_stress,material.hardening_curve)))
            bodies.append((body.body_id,offset,size,p,z,model.mesh.dof_manager.total_dofs,nodes,elements))
        return canonical(bodies)

    def _guard(self,token=None,stage='native_multi_shell.guard'):
        super()._guard(token,stage)
        if any(b.model is not held for b,held in zip(self.shell_bodies,self._captured_models)):
            raise ValueError('exact shell model inventory changed')
        if any(len(b.model.mesh.elements)!=len(held) or any(b.model.mesh.elements.get(eid) is not e for eid,e in held)
               for b,held in zip(self.shell_bodies,self._captured_shell_elements)):
            raise ValueError('exact shell structural object inventory changed')
        if any(vars(cls).keys()!=keys for cls,keys in self._shell_class_keys) or any(vars(cls).get(name) is not value for cls,name,value in self._shell_methods):
            raise ValueError('shell assembly implementation changed')
        if any(self._shell_method_names.intersection(vars(e)) for held in self._captured_shell_elements for _,e in held):
            raise ValueError('shell instance assembly implementation changed')
        if self._shell_configuration()!=self._shell_extra_binding:
            raise ValueError('shell matrix-dependent configuration changed')
        if self.shell_stiffness.flags.writeable or self.shell_mass.flags.writeable:
            raise ValueError('shell cached matrix immutability changed')
        if any(w._fingerprint()!=w.identity or w.contact._fingerprint()!=w.contact.identity for w in self.works):
            raise ValueError('selected contact definition changed')

    def _matrix_witness(self):
        k,m,binding,_=self._shell_contract()
        if binding!=self._shell_binding or not np.array_equal(k,self.shell_stiffness) or not np.array_equal(m,self.shell_mass):
            raise ValueError('shell body assembled matrix witness changed')

    def _new_budget(self):
        return AggregateWorkBudget(self.maximum_geometry_evaluations,self.maximum_integration_intervals,self.maximum_swept_intervals)
    @contextmanager
    def _operation(self,token=None):
        if self._active_budget is not None:raise RuntimeError('nested aggregate operation refused')
        budget=self._new_budget()
        # Set only after the inherited lock is held; no second independent transaction.
        with super()._operation(token):
            self._active_budget=budget
            try:
                self._matrix_witness()
                yield
                self._matrix_witness()
            finally:
                self.last_operation_work=dict(budget.used);self._active_budget=None
    @contextmanager
    def _budget_scope(self):
        if self._active_budget is not None:yield self._active_budget
        else:
            budget=self._new_budget();self._active_budget=budget
            try:yield budget
            finally:self.last_operation_work=dict(budget.used);self._active_budget=None

    def _state(self,data):
        native=super()._state(data)
        q=_array(data.get('shell_displacement',np.zeros(self.shell_size)),(self.shell_size,),'complete global shell displacement')
        if (np.max(np.abs(q[list(i for o,_,p,_ in self.shell_layout for i in (o+x for x in p))]))>self.maximum_shell_displacement_m or
                any(np.linalg.norm(q[o+np.asarray(z)].reshape(-1,3),axis=1).max()>self.maximum_shell_rotation_rad
                    for o,_,_,z in self.shell_layout)):
            raise ValueError('shell small-motion chart limit exceeded')
        return NativeMultiShellState(*(getattr(native,n) for n in ConnectedState.__dataclass_fields__),_owned(q))
    def _advance(self,state,step,factor=1.):
        native=super()._advance(state,step,factor);q=state.shell_displacement.copy()
        q[list(self.shell_p)]+=factor*np.asarray(step)[[self.native_size+i for i in self.shell_p]]
        q[list(self.shell_z)]+=np.asarray(step)[[self.native_size+i for i in self.shell_z]]
        data=self._data(native);data['shell_displacement']=q;return self._state(data)
    def _midpoint_chart(self,increment,u,dt):
        chart=super()._midpoint_chart(increment,u,dt)
        indices=[self.native_size+i for i in self.shell_p];chart[indices,indices]*=.5
        return chart
    def _reference_physical_pose(self,state):
        data=self._data(super()._reference_physical_pose(state));q=state.shell_displacement.copy()
        q[list(self.shell_p)]=0.;data['shell_displacement']=q;return self._state(data)
    def _physical_pose_matches(self,a,b):
        return super()._physical_pose_matches(a,b) and canonical(a.shell_displacement[list(self.shell_p)])==canonical(b.shell_displacement[list(self.shell_p)])

    def _arguments(self,c,state):
        e=self.elements[c.native_index];nodes=[self.node_index[n] for n in e.node_ids]
        o,size,_,_=self.shell_layout[c.shell_index]
        return state.positions[nodes],state.position_low[nodes],state.cell_rotations[c.native_index],state.shell_displacement[o:o+size]
    def _contacts(self,state):
        with self._budget_scope() as budget:
            return tuple(self._contact_one(c,state,budget) for c in self.contributors)
    def _contact_one(self,c,state,budget):
        try:return c.work.evaluate(*self._arguments(c,state),_work_budget=budget)
        except ValueError as exc:
            if 'rim' in str(exc):
                error=ValueError(BOUNDARY);error.boundary_inventory=c.boundaries
                raise error from exc
            raise
    def _evaluate(self,state,token=None):
        with self._budget_scope():return self._evaluate_global(state,token)
    def _evaluate_global(self,state,token):
        r,j,fi,m,dv,dq,account=super()._evaluate(state,token)
        s=slice(self.native_size,self.size);q=state.shell_displacement;v=state.velocity[s];a=state.acceleration[s]
        r[s]+=self.shell_stiffness@q;j[s,s]+=self.shell_stiffness
        m[s,s]+=self.shell_mass;fi[s]+=self.shell_mass@a
        rows=[];errors=[[],[],[]];storage=[]
        for c,work in zip(self.contributors,self._contacts(state)):
            contact=work.contact;idx=list(c.mapping);cj=work.tangent.toarray()
            cj[:42,:42]=spatial_jacobian(work.residual[:42],cj[:42,:42])
            r[idx]+=work.residual;j[np.ix_(idx,idx)]+=cj;storage.append(contact.potential_j)
            for values,value in zip(errors,(contact.energy_error_j,c.error_factor*contact.force_error_n,c.error_factor**2*contact.tangent_error_n_per_m)):
                values.append(nextafter(value,inf) if value>0 else 0.)
            rows.append(dict(pair=c.selection,kernel_identity=c.work.contact.identity,potential_j=contact.potential_j,
                minimum_gap_m=contact.minimum_gap_m,minimum_patch_boundary_clearance_m=contact.minimum_rim_clearance_m,
                selected_face_boundaries=c.boundaries,boundary_semantics='EXTERIOR_RIMS_AND_INTERNAL_TRANSITIONS_EXCLUDED_NO_EDGE_FORCE',beam_force_n=contact.beam_force_n,shell_force_n=contact.shell_force_n,
                beam_moment_nm=contact.beam_moment_nm,shell_moment_nm=contact.shell_moment_nm,
                moment_origin_m=(0.,0.,0.),force_semantics='PHYSICAL_NEGATIVE_POTENTIAL_GRADIENT',
                error_factor=c.error_factor,error_chart='27_PHYSICAL_SAMPLE_METRES_TO_GLOBAL_METRES_LENGTH_TIMES_RAD'))
        if any(not np.isfinite(x) or x<0 for row in errors for x in row):
            raise ValueError('finite nonnegative contact error witnesses required')
        totals=tuple(_outward_sum(x) for x in errors)
        if any(x>t for x,t in zip(totals,(self.energy_tolerance_j,self.force_tolerance_n,self.tangent_tolerance_n_per_m))):
            raise ValueError('global contact potential pullback error gate exceeded')
        account['native_elastic_energy']=account['elastic_energy'];shell_energy=.5*float(q@self.shell_stiffness@q)
        account['shell_elastic_energy']=shell_energy;account['elastic_energy']+=shell_energy
        account['kinetic_energy']+=.5*float(v@self.shell_mass@v)
        account['contact_regularization_storage']=sum(storage)
        account['total_energy']=account['elastic_energy']+account['kinetic_energy']+account['contact_regularization_storage']
        momentum=self.shell_mass@v
        for body,(o,_,_,_) in zip(self.shell_bodies,self.shell_layout):
            slots=np.array([body.model.mesh.dof_manager.get_node_dofs(n)[:3] for n in sorted(body.model.mesh.nodes)])+o
            positions=np.array([body.model.mesh.nodes[n].coords() for n in sorted(body.model.mesh.nodes)])+q[slots]
            extra_p=momentum[slots].sum(axis=0);extra_h=np.cross(positions,momentum[slots]).sum(axis=0)
            for name,extra in (('linear_momentum',extra_p),('angular_momentum',extra_h)):
                high=account[name].copy();low=account[name+'_low'].copy()
                for axis in range(3):high[axis],low[axis]=split_sum((float(high[axis]),float(low[axis]),float(extra[axis])))
                account[name]=high;account[name+'_low']=low
        account['native_multi_shell_contact']=dict(profile=PROFILE,pairs=rows,error_bounds=totals,
            shell_mass_semantics='ACTUAL_ASSEMBLED_TRANSLATION_ONLY_MASS_SHARED_NODES_ONCE',
            shell_rate_semantics='ZERO_ALGEBRAIC_RATE_PLACEHOLDERS_NOT_SKIN_VELOCITY')
        return r,j,fi,m,dv,dq,account

    def _geometry(self,c,state,budget):
        budget.debit('geometry')
        args=self._arguments(c,state);skin=c.work.shell.current_positions(args[3]);reference=c.work.shell.reference_positions
        points=[_line(v,v) for v in skin];refs=[_line(v,v) for v in reference]
        n=_cross(_difference(points[1],points[0]),_difference(points[2],points[0]));nr=_cross(_difference(refs[1],refs[0]),_difference(refs[2],refs[0]))
        if _dot(n,nr)[0]<=0 or _dot(n,n)[0]<=F(c.work.contact.geometry_tolerance)**4:
            raise ValueError('fixed shell physical side/orientation changed')
        try:
            gap,rim=c.work.contact.geometry_gap(*args[:3],skin)
        except ValueError as exc:
            if 'rim' in str(exc):
                error=ValueError(BOUNDARY);error.boundary_inventory=c.boundaries
                raise error from exc
            raise
        return float(gap),float(rim),skin
    def _all_skins(self,state,budget):
        for bi,(body,(o,size,_,_)) in enumerate(zip(self.shell_bodies,self.shell_layout)):
            for eid in sorted(body.model.mesh.elements):
                budget.debit('geometry')
                adapter=S3SkinWorkAdapter.from_model(body.model,body.body_id,eid,'top')
                yield bi,eid,adapter,adapter.current_positions(state.shell_displacement[o:o+size])
    @staticmethod
    def _skin_side(adapter,skin,tolerance):
        points=[_line(v,v) for v in skin];refs=[_line(v,v) for v in adapter.reference_positions]
        n=_cross(_difference(points[1],points[0]),_difference(points[2],points[0]))
        nr=_cross(_difference(refs[1],refs[0]),_difference(refs[2],refs[0]))
        if _dot(n,nr)[0]<=0 or _dot(n,n)[0]<=F(tolerance)**4:
            raise ValueError('fixed shell physical side/orientation changed')
    def _unselected_orientation_leg(self,adapter,a,b,error,budget,token):
        points=[_line(x,y) for x,y in zip(a,b)];refs=[_line(x,x) for x in adapter.reference_positions]
        e1=_difference(points[1],points[0]);e2=_difference(points[2],points[0]);normal=_cross(e1,e2)
        reference=_cross(_difference(refs[1],refs[0]),_difference(refs[2],refs[0]))
        refnorm=_dot(reference,reference)[0];refmax=max(abs(x[0]) for x in reference)
        tolerance=min(w.contact.geometry_tolerance for w in self.works);pending=[(F(0),F(1))]
        while pending:
            cancellation_safe_point(token,'native_multi_shell.unselected.orientation')
            budget.debit('swept');left,right=pending.pop()
            emax=max(_absolute(e1,left,right),_absolute(e2,left,right))
            cerr=8*error*emax+8*error**2
            orientation=_range(_dot(normal,reference),left,right)[0]-3*cerr*refmax
            if orientation>0 and orientation**2>F(tolerance)**4*refnorm:continue
            middle=(left+right)/2
            if float(middle) in (float(left),float(right)):raise ValueError(PATH_FAILURE)
            pending.extend(((left,middle),(middle,right)))

    def _admit(self,state,initial=False):
        restored=self._state(self._data(state))
        if canonical(restored)!=canonical(state):raise ValueError('exact global descriptor state required')
        with self._budget_scope() as budget:
            for _,_,adapter,skin in self._all_skins(state,budget):
                self._skin_side(adapter,skin,min(w.contact.geometry_tolerance for w in self.works))
            for c in self.contributors:
                gap,_,_=self._geometry(c,state,budget)
                if gap < (0. if initial else -self.maximum_penetration_m):raise ValueError('native multi-S3 penetration admission exceeded')
    def _validate_initial_contact(self,state):self._admit(state,True)
    def _admit_saved_state(self,state):self._admit(state)
    def _initialize_state(self,*args,**kwargs):
        state=super()._initialize_state(*args,**kwargs);self._validate_initial_contact(state);return state

    def _swept(self,previous,stage,endpoint,token):
        with self._budget_scope() as budget:
            before=budget.used['swept'];records=[]
            for state in (previous,stage,endpoint):self._admit_saved_state(state)
            eps=F(1,2**52);subnormal=F(1,2**1074)
            for c in self.contributors:
                budget.debit('geometry')
                args=[self._arguments(c,state) for state in (previous,stage,endpoint)]
                beams=[np.array([[float(F(float(h))+F(float(l))) for h,l in zip(row,low)] for row,low in zip(a[0],a[1])]) for a in args]
                skins=[c.work.shell.current_positions(a[3]) for a in args]
                bmag=max(F(1),*(abs(F(float(x))) for a in args for x in np.r_[a[0].ravel(),a[1].ravel()]))
                qmag=max(F(1),*(abs(F(float(x))) for a in args for x in a[3]))
                reference=max(F(1),*(abs(F(float(x))) for x in c.work.shell.reference_positions.ravel()))
                mapping=c.work.shell.work_map
                rowabs=max(sum(abs(F(float(x))) for x in mapping.data[mapping.indptr[i]:mapping.indptr[i+1]]) for i in range(9))
                berr=2*(256*eps*bmag+64*subnormal);serr=2*(256*eps*(reference+rowabs*qmag)+64*subnormal)
                view=SimpleNamespace(contact=c.work,maximum_penetration_m=self.maximum_penetration_m,
                                     maximum_swept_intervals=self.maximum_swept_intervals)
                # The mature certificate debits this shared counter prospectively;
                # propagate every visited interval even when certification fails.
                shared={'visited':budget.used['swept']};legs=[]
                try:
                    for i in (0,1):legs.append(NativeShellContactAnalysis._leg_certificate(view,beams[i],beams[i+1],skins[i],skins[i+1],berr,serr,token,shared))
                finally:budget.used['swept']=shared['visited']
                records.append(dict(pair=c.selection,per_leg_checked_intervals=legs,rounded_beam_envelope_m=float(berr),rounded_shell_envelope_m=float(serr)))
            selected={(c.shell_index,c.selection.shell_element_id) for c in self.contributors}
            for bi,eid,adapter,skin0 in self._all_skins(previous,budget):
                if (bi,eid) in selected:continue
                o,size,_,_=self.shell_layout[bi]
                budget.debit('geometry');budget.debit('geometry')
                skins=[skin0]+[adapter.current_positions(s.shell_displacement[o:o+size]) for s in (stage,endpoint)]
                reference=max(F(1),*(abs(F(float(x))) for x in adapter.reference_positions.ravel()))
                qmag=max(F(1),*(abs(F(float(x))) for state in (previous,stage,endpoint) for x in state.shell_displacement[o:o+size]))
                mapping=adapter.work_map
                rowabs=max(sum(abs(F(float(x))) for x in mapping.data[mapping.indptr[i]:mapping.indptr[i+1]]) for i in range(9))
                error=2*(256*eps*(reference+rowabs*qmag)+64*subnormal)
                for i in (0,1):self._unselected_orientation_leg(adapter,skins[i],skins[i+1],error,budget,token)
            return dict(path='ALL_PAIRS_TWO_LEG_ACTUAL_STAGE_PROJECTED_ENDPOINT',pairs=records,
                checked_intervals=budget.used['swept']-before,minimum_certified_gap_m=-self.maximum_penetration_m,
                minimum_bound_semantics='CERTIFIED_THRESHOLD_FLOOR_NOT_SAMPLED_MINIMUM')
    def _trial_state(self,accepted,dt,token):
        candidate=super()._trial_state(accepted,dt,token)
        self._swept(accepted,candidate.stage,candidate.endpoint,token);return candidate
    def _step_extra(self,previous,unknown,dt,stage,endpoint,token):
        certificate=self._swept(previous,stage,endpoint,token)
        increment=np.zeros(self.size);increment[list(self.p)]=dt*previous.velocity[list(self.p)]+.5*dt*dt*unknown[list(self.p)]
        increment[list(self.z)]=unknown[list(self.z)];increment[self.native_size:]=endpoint.shell_displacement-previous.shell_displacement
        values=[float(work.residual@increment[list(c.mapping)]) for c,work in zip(self.contributors,self._contacts(stage))]
        certificate.update(stage_contact_internal_work_j=sum(values),per_pair_stage_work_j=values,
            work_semantics='COMMON_STAGE_CONTACT_WITH_ACTUAL_ACCEPTED_SHELL_INCREMENT_INCLUDING_PROJECTION',
            shell_endpoint_projection_increment=endpoint.shell_displacement-stage.shell_displacement)
        return dict(native_multi_shell_contact=certificate)
