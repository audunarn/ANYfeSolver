"""Source-only native/S3 small-motion midpoint descriptor DEVELOPMENT owner.

One60 retained system;24 physical and36 algebraic coordinates. No shell
rotary inertia or material-surface velocity is inferred from zero rate slots.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import comb
import numpy as np
from anymaterial import IsotropicMaterial
from ._ge_beam3_connected_dynamic import _RetainedMidpointAssembly, ConnectedState
from ._ge_beam3_generalized_static_boundary import spatial_jacobian
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5.compensated_coordinates import split_sum
from ._ge_beam3_p5.arrays import _array
from ._native_reference_modal import _owned
from .control import cancellation_safe_point
from .fe_core import FEModel
from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from .general_contact.native_retained_line_s3_work import NativeRetainedLineS3Work
from .general_contact.s3_skin_work import S3SkinWorkAdapter

PROFILE='NATIVE_B3_GE_S3_SMALL_MOTION_DESCRIPTOR_MIDPOINT_DEV_V1'
SCHEMA='NATIVE_B3_GE_S3_SMALL_MOTION_NATIVE_CHECKPOINT_DEV_V1'
TRAJECTORY='NATIVE_B3_GE_S3_SMALL_MOTION_TRAJECTORY_DEV_V1'
PATH_FAILURE='native shell actual-stage endpoint-projection path unresolved'
SHELL_P=(0,1,2,6,7,8,12,13,14)
SHELL_Z=(3,4,5,9,10,11,15,16,17)


@dataclass(frozen=True)
class NativeShellState(ConnectedState):
    shell_displacement: np.ndarray


def _plus(a,b):
    return [(a[i] if i<len(a) else F(0))+(b[i] if i<len(b) else F(0)) for i in range(max(len(a),len(b)))]
def _product(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out
def _subtract(a,b):return _plus(a,[-x for x in b])
def _dot(a,b):
    out=[F(0)]
    for x,y in zip(a,b):out=_plus(out,_product(x,y))
    return out
def _cross(a,b):
    return [_subtract(_product(a[1],b[2]),_product(a[2],b[1])),
            _subtract(_product(a[2],b[0]),_product(a[0],b[2])),
            _subtract(_product(a[0],b[1]),_product(a[1],b[0]))]
def _difference(a,b):return [_subtract(x,y) for x,y in zip(a,b)]
def _line(a,b):return [[F(float(x)),F(float(y))-F(float(x))] for x,y in zip(a,b)]
def _bernstein(poly,left,right):
    """Exact polynomial hull on a rational subinterval, never sampled signs."""
    n=len(poly)-1;width=right-left
    shifted=[sum(poly[i]*comb(i,j)*left**(i-j)*width**j for i in range(j,n+1)) for j in range(n+1)]
    return tuple(sum(shifted[j]*F(comb(k,j),comb(n,j)) for j in range(k+1)) for k in range(n+1))
def _range(poly,left,right):
    values=_bernstein(poly,left,right);return min(values),max(values)
def _absolute(vector,left,right):
    return max(abs(x) for poly in vector for x in _bernstein(poly,left,right))


class NativeShellContactAnalysis(_RetainedMidpointAssembly):
    route_profile=PROFILE
    checkpoint_schema=SCHEMA
    state_type=NativeShellState
    trajectory_profile=TRAJECTORY
    trajectory_schema='NATIVE_B3_GE_S3_SMALL_MOTION_TRAJECTORY_CHECKPOINT_DEV_V1'
    contact_result_label='NATIVE_RETAINED_ROUND_LINE_S3_INTERIOR'
    extra_step_keys=('native_shell_contact',)
    retryable=_RetainedMidpointAssembly.retryable+(PATH_FAILURE,)

    def __init__(self,definitions,boundaries,shell_model,contact,*,maximum_penetration_m,
                 maximum_swept_intervals,maximum_shell_displacement_m,
                 maximum_shell_rotation_rad,inertia_order=24,max_iterations=12,max_backtracks=8):
        if type(shell_model) is not FEModel or type(contact) is not NativeRetainedLineS3Work:
            raise TypeError('exact shell FEModel and retained native-S3 work required')
        if type(boundaries) is not tuple or boundaries:
            raise ValueError('free native/S3 descriptor admits no native supports')
        if any(type(x) not in (int,float) or not np.isfinite(x) or x<=0 for x in
               (maximum_penetration_m,maximum_shell_displacement_m,maximum_shell_rotation_rad)):
            raise ValueError('explicit finite positive penetration/small-motion limits required')
        if maximum_penetration_m>=contact.contact.radius or maximum_shell_rotation_rad>.05:
            raise ValueError('penetration must be below radius; shell linear rotation chart<=.05rad')
        if type(maximum_swept_intervals) is not int or not 1<=maximum_swept_intervals<=4096:
            raise ValueError('bounded1..4096 actual path intervals required')
        self.shell_model=shell_model;self.contact=contact
        self.maximum_penetration_m=float(maximum_penetration_m)
        self.maximum_shell_displacement_m=float(maximum_shell_displacement_m)
        self.maximum_shell_rotation_rad=float(maximum_shell_rotation_rad)
        self.maximum_swept_intervals=maximum_swept_intervals
        self.shell_stiffness,self.shell_mass,self._shell_binding=self._shell_contract()
        super().__init__(definitions,boundaries,inertia_order=inertia_order,
            max_iterations=max_iterations,max_backtracks=max_backtracks)
        element=self.elements[0];kernel=contact.contact
        if (element.element_id!=kernel.beam_element_id or element.node_ids!=kernel.beam_node_ids
                or sha(element.operator.reference.canonical_data())!=sha(kernel.beam_reference.canonical_data())):
            raise ValueError('contact must bind complete native definition')
        if any(np.any(element.operator.reference.half_cell_lift(c,.5)) for c in (0,1)):
            raise ValueError('actual retained reconstruction requires exact zero half-cell lifts')
        if set(element.node_ids)&set(contact.shell.node_ids):
            raise ValueError('native and shell material node identities must be disjoint')
        self.size=60;self.physical=self.physical+tuple(42+i for i in SHELL_P)
        self.algebraic=self.algebraic+tuple(42+i for i in SHELL_Z)
        self.p=self.physical;self.z=self.algebraic;self.free=tuple(range(60))
        self.scale=_owned(np.r_[self.scale,np.ones(18)])
        scale=self.scale.copy();scale[[42+i for i in SHELL_Z]]=self.length;self.scale=_owned(scale)
        self._binding=canonical(self._definition());self.identity=sha(self._definition())

    def _admit_element_count(self,count):
        if count!=1:raise ValueError('native/S3 route requires exactly one complete native element')
    def _admit_graph(self):
        if len(self.elements)!=1:raise ValueError('exact single-native component required')

    def _shell_contract(self):
        model=self.shell_model;adapter=self.contact.shell
        if (len(model.mesh.elements)!=1 or len(model.mesh.nodes)!=3 or model.mesh.dof_manager.total_dofs!=18
                or model.boundary_conditions or model.constraint_equations or model.load_cases
                or model.mesh.point_masses or model.mesh.element_activity is not None):
            raise ValueError('one free18-DOF S3 with no loads/supports/pointmass/activity required')
        element=model.mesh.elements.get(adapter.element_id)
        if (type(element) is not NativeParityE4PLS3V2DShellElement or element.shell_section is not None
                or element.reference_surface_offset!=0. or tuple(element.node_ids)!=adapter.node_ids):
            raise ValueError('exact centered isotropic elastic small-strain S3 V2D required')
        normal=np.cross(adapter.reference_positions[1]-adapter.reference_positions[0],
                        adapter.reference_positions[2]-adapter.reference_positions[0])
        if (adapter.side!='top' or element.director_polarity!=1 or adapter.physical_offset<=0.
                or float(normal@adapter.director)<=0.):
            raise ValueError('native/S3 profile admits only top skin positive director polarity and ordered outward normal')
        material=model.materials.get(element.material_name)
        if (type(material) is not IsotropicMaterial or material.yield_stress!=0.
                or material.hardening_curve is not None or material.density<=0.):
            raise ValueError('S3 elastic positive-density material required')
        fresh=S3SkinWorkAdapter.from_model(model,adapter.body_id,adapter.element_id,adapter.side)
        if (not np.array_equal(fresh.reference_positions,adapter.reference_positions)
                or not np.array_equal(fresh.director,adapter.director)
                or fresh.physical_offset!=adapter.physical_offset or (fresh.work_map!=adapter.work_map).nnz):
            raise ValueError('actual formulation physical skin map changed')
        slots=tuple(int(x) for x in element.get_dof_mapping(model.mesh))
        k=np.zeros((18,18));m=np.zeros((18,18))
        k[np.ix_(slots,slots)]=element.compute_stiffness_matrix(model.mesh,material)
        m[np.ix_(slots,slots)]=element.compute_mass_matrix(model.mesh,material)
        if (not np.isfinite(k).all() or not np.isfinite(m).all()
                or np.linalg.norm(k-k.T)>1e-10*max(1.,np.linalg.norm(k))
                or not np.array_equal(m,m.T) or np.any(m[list(SHELL_Z),:]) or np.any(m[:,list(SHELL_Z)])):
            raise ValueError('actual S3 stiffness/mass/zero-rotation witness invalid')
        np.linalg.cholesky(m[np.ix_(SHELL_P,SHELL_P)])
        descriptor=dict(formulation=element.formulation_id,nodes=adapter.node_ids,slots=slots,
            coordinates=[model.mesh.nodes[n].coords() for n in adapter.node_ids],
            thickness=element.thickness,reference_offset=element.reference_surface_offset,
            director_polarity=element.director_polarity,material=dict(name=material.name,
                E=material.elastic_modulus,nu=material.poisson_ratio,density=material.density),
            stiffness=k,mass=m,skin=self.contact._fingerprint(),kernel=self.contact.contact._fingerprint())
        return _owned(k),_owned(m),canonical(descriptor)

    def _definition(self):
        data=super()._definition()
        data.update(shell_contract=self._shell_binding.decode('ascii'),contact=self.contact._fingerprint(),
            state_schema='COMPLETE_NATIVE_PLUS_SHELL_DISPLACEMENT60',shell_physical=SHELL_P,shell_algebraic=SHELL_Z,
            shell_rate_semantics='ZERO_ROTATION_RATE_SLOTS_ARE_NONINERTIAL_PLACEHOLDERS_NOT_SKIN_VELOCITY',
            maximum_penetration_m=self.maximum_penetration_m,maximum_swept_intervals=self.maximum_swept_intervals,
            maximum_shell_displacement_m=self.maximum_shell_displacement_m,
            maximum_shell_rotation_rad=self.maximum_shell_rotation_rad,
            path='NATIVE_AFFINE_EXP_AND_SHELL_TWO_LEGS_THROUGH_ACTUAL_STAGE_PROJECTED_ENDPOINT')
        return data

    def _guard(self,token=None,stage='native_shell.guard'):
        super()._guard(token,stage)
        _,_,binding=self._shell_contract()
        if binding!=self._shell_binding:raise ValueError('shell operator/model/material/contact definition changed')

    def _state(self,data):
        native=super()._state(data)
        q=_array(data.get('shell_displacement',np.zeros(18)),(18,),'complete shell displacement')
        if (np.max(np.abs(q[list(SHELL_P)]))>self.maximum_shell_displacement_m
                or max(np.linalg.norm(q[6*i+3:6*i+6]) for i in range(3))>self.maximum_shell_rotation_rad):
            raise ValueError('shell small-motion chart limit exceeded')
        return NativeShellState(*(getattr(native,n) for n in ConnectedState.__dataclass_fields__),_owned(q))

    def _advance(self,state,step,factor=1.):
        native=super()._advance(state,step,factor);q=state.shell_displacement.copy()
        q[list(SHELL_P)]+=factor*np.asarray(step)[[42+i for i in SHELL_P]]
        # Eta is the ACTUAL stage algebraic increment, matching the descriptor
        # chart. It is full, unlike the half physical endpoint increment.
        q[list(SHELL_Z)]+=np.asarray(step)[[42+i for i in SHELL_Z]]
        data=self._data(native);data['shell_displacement']=q;return self._state(data)

    def _midpoint_chart(self,increment,u,dt):
        chart=super()._midpoint_chart(increment,u,dt)
        chart[[42+i for i in SHELL_P],[42+i for i in SHELL_P]]*=.5
        return chart

    def _contact(self,state):
        return self.contact.evaluate(state.positions,state.position_low,state.cell_rotations[0],state.shell_displacement)

    def _evaluate(self,state,token=None):
        r,j,fi,m,dv,dq,account=super()._evaluate(state,token)
        shell=slice(42,60);q=state.shell_displacement;v=state.velocity[shell];a=state.acceleration[shell]
        r[shell]+=self.shell_stiffness@q;j[shell,shell]+=self.shell_stiffness
        m[shell,shell]+=self.shell_mass;fi[shell]+=self.shell_mass@a
        work=self._contact(state);contact=work.contact;cj=work.tangent.toarray()
        cj[:42,:42]=spatial_jacobian(work.residual[:42],cj[:42,:42])
        # The shell chart is linear. Preserve all native/shell cross blocks.
        r+=work.residual;j+=cj
        account['native_elastic_energy']=account['elastic_energy']
        shell_energy=.5*float(q@self.shell_stiffness@q)
        account['shell_elastic_energy']=shell_energy;account['elastic_energy']+=shell_energy
        account['kinetic_energy']+=.5*float(v@self.shell_mass@v)
        account['contact_regularization_storage']=contact.potential_j
        account['total_energy']=account['elastic_energy']+account['kinetic_energy']+contact.potential_j
        translation_slots=np.asarray([self.shell_model.mesh.dof_manager.get_node_dofs(n)[:3]
                                      for n in self.contact.shell.node_ids],dtype=np.intp)
        momentum=(self.shell_mass@v)[translation_slots]
        coords=np.array([self.shell_model.mesh.nodes[n].coords() for n in self.contact.shell.node_ids])+q[translation_slots]
        shell_p=momentum.sum(axis=0);shell_h=np.cross(coords,momentum).sum(axis=0)
        for name,extra in (('linear_momentum',shell_p),('angular_momentum',shell_h)):
            high=np.array(account[name],copy=True);low=np.array(account[name+'_low'],copy=True)
            for axis in range(3):high[axis],low[axis]=split_sum((float(high[axis]),float(low[axis]),float(extra[axis])))
            account[name]=high;account[name+'_low']=low
        account['shell_mass_semantics']='FORMULATION_TRANSLATION_ONLY_MASS_ZERO_ROTATIONAL_INERTIA'
        account['shell_rate_semantics']='ALGEBRAIC_ZERO_RATE_SLOTS_ARE_NOT_MATERIAL_SURFACE_VELOCITY'
        account['native_shell_contact']=dict(profile=PROFILE,kernel_identity=self.contact.contact.identity,
            potential_j=contact.potential_j,beam_force_n=contact.beam_force_n,shell_force_n=contact.shell_force_n,
            beam_moment_nm=contact.beam_moment_nm,shell_moment_nm=contact.shell_moment_nm,moment_origin_m=(0.,0.,0.),
            force_semantics='PHYSICAL_NEGATIVE_POTENTIAL_GRADIENT',residual_semantics='INTERNAL_POSITIVE_POTENTIAL_GRADIENT',
            minimum_gap_m=contact.minimum_gap_m,minimum_rim_clearance_m=contact.minimum_rim_clearance_m,
            energy_error_j=contact.energy_error_j,force_error_n=contact.force_error_n,
            tangent_error_n_per_m=contact.tangent_error_n_per_m)
        return r,j,fi,m,dv,dq,account

    def _reference_physical_pose(self,state):
        reference=super()._reference_physical_pose(state);data=self._data(reference)
        q=reference.shell_displacement.copy();q[list(SHELL_P)]=0.;data['shell_displacement']=q
        return self._state(data)

    def _physical_pose_matches(self,actual,expected):
        return (super()._physical_pose_matches(actual,expected) and
                canonical(actual.shell_displacement[list(SHELL_P)])==
                canonical(expected.shell_displacement[list(SHELL_P)]))

    def _initialize_state(self,*args,**kwargs):
        state=super()._initialize_state(*args,**kwargs);self._validate_initial_contact(state);return state

    def _geometry(self,state):
        skin=self.contact.shell.current_positions(state.shell_displacement)
        reference=self.contact.shell.reference_positions
        if not np.isfinite(skin).all():raise ValueError('nonfinite actual shell skin')
        exact=[_line(v,v) for v in skin];ref=[_line(v,v) for v in reference]
        ref_normal=_cross(_difference(ref[1],ref[0]),_difference(ref[2],ref[0]))
        current_normal=_cross(_difference(exact[1],exact[0]),_difference(exact[2],exact[0]))
        if (_dot(current_normal,ref_normal)[0]<=0
                or _dot(current_normal,current_normal)[0]<=F(self.contact.contact.geometry_tolerance)**4):
            raise ValueError('native/S3 fixed physical side or orientation changed')
        gap,rim=self.contact.contact.geometry_gap(state.positions,state.position_low,state.cell_rotations[0],skin)
        return float(gap),float(rim),skin

    def _validate_initial_contact(self,state):
        gap,_,_=self._geometry(state)
        if gap<0.:raise ValueError('initial native/S3 penetration or unresolved geometry')
    def _admit_saved_state(self,state):
        restored=self._state(self._data(state))
        if canonical(restored)!=canonical(state):raise ValueError('exact admitted shell state required')
        gap,_,_=self._geometry(restored)
        if gap < -self.maximum_penetration_m:raise ValueError('restored native/S3 penetration budget exceeded')

    def _leg_certificate(self,beam0,beam1,skin0,skin1,beam_error,skin_error,token,budget=None):
        """Exact Bernstein enclosure for signed gap/rim, including rounding.

        N=(S1-S0)x(S2-S0), h=N.(P-S0)>0. Only then may the squared
        positive-height test use (R-Pen)^2|N|^2. Rim numerator is
        N.(edge x (P-Si)), denominator |N|*|edge|, all positive.
        """
        s=[_line(a,b) for a,b in zip(skin0,skin1)];p=[_line(a,b) for a,b in zip(beam0,beam1)]
        edges=[_difference(s[(i+1)%3],s[i]) for i in range(3)]
        e1=_difference(s[1],s[0]);e2=_difference(s[2],s[0]);normal=_cross(e1,e2)
        ref=np.asarray(self.contact.shell.reference_positions);reference=np.cross(ref[1]-ref[0],ref[2]-ref[0])
        ref_polys=[[F(float(x))] for x in reference];ref_norm=sum(x[0]**2 for x in ref_polys)
        required_height=F(self.contact.contact.radius)-F(self.maximum_penetration_m)
        required_rim=F(self.contact.contact.radius)+F(self.contact.contact.rim_clearance)
        if budget is None:budget={'visited':0}
        pending=[(F(0),F(1))];checked=0
        def refusal():
            error=ValueError(PATH_FAILURE);error.path_checked_intervals=budget['visited']
            error.path_interval_cap=self.maximum_swept_intervals;return error
        while pending:
            cancellation_safe_point(token,'native_shell.path.interval')
            if budget['visited']>=self.maximum_swept_intervals:raise refusal()
            left,right=pending.pop();checked+=1;budget['visited']+=1
            emax=max(_absolute(e1,left,right),_absolute(e2,left,right))
            cmax=_absolute(normal,left,right)
            cerr=8*skin_error*emax+8*skin_error**2
            norm_upper=sum((max(abs(v) for v in _bernstein(component,left,right))+cerr)**2 for component in normal)
            orientation=_range(_dot(normal,ref_polys),left,right)[0]-3*cerr*max(abs(x[0]) for x in ref_polys)
            valid=(orientation>0 and orientation**2>F(self.contact.contact.geometry_tolerance)**4*ref_norm)
            for point in p:
                relative=_difference(point,s[0]);rmax=_absolute(relative,left,right);rerr=beam_error+skin_error
                height=_range(_dot(normal,relative),left,right)[0]-3*(cmax*rerr+cerr*rmax+cerr*rerr)
                valid=valid and height>0 and height**2>=required_height**2*norm_upper
                for i,edge in enumerate(edges):
                    rel=_difference(point,s[i]);relmax=_absolute(rel,left,right);edgemax=_absolute(edge,left,right)
                    edgeerr=2*skin_error;verr=2*(edgemax*rerr+relmax*edgeerr+edgeerr*rerr)
                    vmax=2*edgemax*relmax
                    numerator=_range(_dot(normal,_cross(edge,rel)),left,right)[0]-3*(cmax*verr+cerr*vmax+cerr*verr)
                    edge_upper=sum((max(abs(v) for v in _bernstein(component,left,right))+edgeerr)**2 for component in edge)
                    valid=valid and numerator>0 and numerator**2>=required_rim**2*norm_upper*edge_upper
            if not valid:
                middle=(left+right)/2
                if float(middle) in (float(left),float(right)):raise refusal()
                pending.extend(((left,middle),(middle,right)))
        return checked

    def _swept(self,previous,stage,endpoint,token):
        self._guard(token,'native_shell.path.begin')
        for state in (previous,stage,endpoint):
            self._admit_saved_state(state)
        def beam(state):
            return np.array([[float(F(float(h))+F(float(l))) for h,l in zip(row,low)]
                             for row,low in zip(state.positions,state.position_low)])
        beams=[beam(state) for state in (previous,stage,endpoint)]
        skins=[self.contact.shell.current_positions(state.shell_displacement) for state in (previous,stage,endpoint)]
        eps=F(1,2**52);subnormal=F(1,2**1074)
        bmag=max(F(1),*(abs(F(float(x))) for state in (previous,stage,endpoint)
            for x in np.r_[state.positions.ravel(),state.position_low.ravel()]))
        qmag=max(F(1),*(abs(F(float(x))) for state in (previous,stage,endpoint) for x in state.shell_displacement))
        reference=max(F(1),*(abs(F(float(x))) for x in self.contact.shell.reference_positions.ravel()))
        mapping=self.contact.shell.work_map
        rowabs=max(sum(abs(F(float(x))) for x in mapping.data[mapping.indptr[i]:mapping.indptr[i+1]]) for i in range(9))
        # Two uniform envelopes: evaluated endpoints->exact affine family and
        # exact interior->rounded advance/retained-cell/skin evaluations.
        berr=2*(256*eps*bmag+64*subnormal)
        serr=2*(256*eps*(reference+rowabs*qmag)+64*subnormal)
        budget={'visited':0};legs=[]
        for i in (0,1):
            legs.append(self._leg_certificate(beams[i],beams[i+1],skins[i],skins[i+1],berr,serr,token,budget))
        checked=budget['visited']
        self._guard(token,'native_shell.path.end')
        return dict(path='TWO_LEG_ACTUAL_STAGE_PROJECTED_ENDPOINT_SHELL_AFFINE_NATIVE_ZERO_LIFT',
            minimum_certified_gap_m=-self.maximum_penetration_m,maximum_penetration_m=self.maximum_penetration_m,
            checked_intervals=checked,per_leg_checked_intervals=legs,minimum_bound_semantics='PROVEN_THRESHOLD_FLOOR_NOT_SAMPLED_MINIMUM',
            native_reconstruction='BOTH_RETAINED_CELLS_EXACT_ZERO_HALF_CELL_LIFT_AFFINE_PHYSICAL_CURVE',
            rounded_family_error_beam_m=float(berr),rounded_family_error_shell_m=float(serr))

    def _trial_state(self,accepted,dt,token):
        candidate=super()._trial_state(accepted,dt,token)
        self._swept(accepted,candidate.stage,candidate.endpoint,token);return candidate

    def _step_extra(self,previous,unknown,dt,stage,endpoint,token):
        certificate=self._swept(previous,stage,endpoint,token);work=self._contact(stage)
        increment=np.zeros(60);increment[list(self.p)]=dt*previous.velocity[list(self.p)]+.5*dt*dt*unknown[list(self.p)]
        increment[list(self.z)]=unknown[list(self.z)]
        increment[42:]=endpoint.shell_displacement-previous.shell_displacement
        certificate['stage_contact_internal_work_j']=float(work.residual@increment)
        certificate['work_semantics']='STAGE_NATIVE_SPATIAL_CONTACT_WORK_PLUS_ACTUAL_ACCEPTED_SHELL_INCREMENT_INCLUDING_ENDPOINT_PROJECTION'
        certificate['shell_endpoint_projection_increment']=endpoint.shell_displacement-stage.shell_displacement
        return dict(native_shell_contact=certificate)
