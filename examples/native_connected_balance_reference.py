"""Independent straight-reference connected elastic physical balances.

No anyfesolver import, cached station data or reported owner accounting is used.
This reference is deliberately bounded to the registered two straight elements,
identity reference triads and diagonal centered section tensors. No C7 claim.
"""
import math
import numpy as np

PARITY_TOLERANCE=1e-10
BALANCE_TOLERANCE=1e-6
BASE_DT=.00025
BASE_STEPS=4
INITIAL_ANALYTIC={
    False:(.05230416666666667,(.04,.125,0.),(.061,-.012833333333333333,0.)),
    True:(.0516,(.035,.10,.005),(.061,-.008666666666666666,.020833333333333332))}

REFERENCE_POSITIONS=np.array(((-1.,0.,0.),(-.5,0.,0.),(0.,0.,0.),(.5,0.,0.),(1.,0.,0.)))
ELEMENT_NODES=((0,1,2),(2,3,4))
ELASTIC=np.array((1e5,3e4,3e4,30.,20.,10.))
SECTION_INERTIA=np.array((2.,2.,2.,.02,.01,.01))


def _quadratic_mean(left,right,diagonal):
    return float((left@(diagonal*left)+left@(diagonal*right)+right@(diagonal*right))/3.)


def physical_kinetic(state):
    """Exact polynomial integration of the independent straight-cell fields."""
    x=np.asarray(state['positions'],dtype=float)+np.asarray(state['position_low'],dtype=float)
    u=np.asarray(state['cell_rotations'],dtype=float);rates=np.asarray(state['velocity'],dtype=float)
    if x.shape!=(5,3)or u.shape!=(2,2,3,3)or rates.shape!=(78,):raise ValueError('registered connected5node/2element/78mixed reference required')
    kinetic=0.;linear=np.zeros(3);angular=np.zeros(3)
    for e,nodes in enumerate(ELEMENT_NODES):
        for c,(left,right)in enumerate(zip(nodes[:-1],nodes[1:])):
            length=float(np.linalg.norm(REFERENCE_POSITIONS[right]-REFERENCE_POSITIONS[left]))
            vl=rates[6*left:6*left+3];vr=rates[6*right:6*right+3]
            omega=rates[30+6*e+3*c:33+6*e+3*c];rotation=u[e,c]
            body_omega=rotation.T@omega;spin=rotation@(SECTION_INERTIA[3:]*body_omega)
            kinetic+=.5*length*(_quadratic_mean(vl,vr,SECTION_INERTIA[:3])+body_omega@(SECTION_INERTIA[3:]*body_omega))
            linear+=length*SECTION_INERTIA[0]*(vl+vr)/2.
            orbital=(2*np.cross(x[left],vl)+np.cross(x[left],vr)+np.cross(x[right],vl)+2*np.cross(x[right],vr))/6.
            angular+=length*(SECTION_INERTIA[0]*orbital+spin)
    if not np.isfinite(np.r_[kinetic,linear,angular]).all():raise ValueError('nonfinite independent physical balance')
    return dict(kinetic_energy=float(kinetic),linear_momentum=linear,angular_momentum=angular)


def _skew(vector):
    x,y,z=vector
    return np.array(((0.,-z,y),(z,0.,-x),(-y,x,0.)))


def _log_rotation(matrix):
    """Independent principal SO(3) log; this case stays away from pi."""
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(3,3)or not np.isfinite(matrix).all():raise ValueError('finite reference rotation required')
    if np.linalg.norm(matrix.T@matrix-np.eye(3))>1e-9 or np.linalg.det(matrix)<=0:raise ValueError('proper reference rotation required')
    sine_axis=np.array((matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]))/2.
    sine=float(np.linalg.norm(sine_axis));cosine=(float(np.trace(matrix))-1.)/2.
    theta=math.atan2(sine,cosine)
    if theta>=math.pi-1e-5:raise ValueError('independent principal log excludes pi')
    if sine==0.:return np.zeros(3)
    return sine_axis*(theta/sine)


def _left_log_inverse(phi):
    theta=float(np.linalg.norm(phi));cross=_skew(phi)
    if theta<1e-3:
        coefficient=1./12.+theta**2/720.+theta**4/30240.
    else:
        coefficient=(1.-.5*theta/math.tan(.5*theta))/(theta*theta)
    return np.eye(3)-.5*cross+coefficient*(cross@cross)


def elastic_reference(state):
    """Independently integrate the elastic complementary and mixed potentials.

    Straight identity-triad cells have constant section force and linear moment.
    Their force constraint is integral(n/L)dS=F. Diagonal compliance decouples
    forces/moments, so the complementary minimizer is n=F. No material return
    mapping, station cache or owner potential is consulted.
    """
    high=np.asarray(state['positions'],dtype=float);low=np.asarray(state['position_low'],dtype=float)
    frames=np.asarray(state['nodal_frames'],dtype=float)
    rotations=np.asarray(state['cell_rotations'],dtype=float)
    resultants=np.asarray(state['resultants'],dtype=float)
    if high.shape!=(5,3)or low.shape!=(5,3)or frames.shape!=(5,3,3)or rotations.shape!=(2,2,3,3)or resultants.shape!=(2,18):raise ValueError('complete registered reference state required')
    complementary=0.;work_terms=[];compatibility=[];witness_terms=[]
    for e,nodes in enumerate(ELEMENT_NODES):
        p=resultants[e];force_residuals=[];moment_residuals=[]
        for c,(left,right)in enumerate(zip(nodes[:-1],nodes[1:])):
            rotation=rotations[e,c];force=p[3*c:3*c+3]
            ml=p[6+6*c:9+6*c];mr=p[9+6*c:12+6*c]
            chord0=REFERENCE_POSITIONS[right]-REFERENCE_POSITIONS[left]
            length=float(np.linalg.norm(chord0))
            strain=np.array([math.fsum([*(float(rotation[k,i])*float(high[right,k]) for k in range(3)),
                *(float(rotation[k,i])*float(low[right,k]) for k in range(3)),
                *(-float(rotation[k,i])*float(high[left,k]) for k in range(3)),
                *(-float(rotation[k,i])*float(low[left,k]) for k in range(3)),-float(chord0[i])])for i in range(3)])
            ell_left=-_log_rotation(rotation.T@frames[left]);ell_right=_log_rotation(rotation.T@frames[right])
            work_terms.extend((float(force@strain),float(ml@ell_left),float(mr@ell_right)))
            complementary+=.5*length*(force@((1./ELASTIC[:3])*force)+_quadratic_mean(ml,mr,1./ELASTIC[3:]))
            rf=strain-length*force/ELASTIC[:3]
            rl=ell_left-length*(2.*ml+mr)/(6.*ELASTIC[3:])
            rr=ell_right-length*(ml+2.*mr)/(6.*ELASTIC[3:])
            force_residuals.append(rf);moment_residuals.extend((rl,rr))
            witness_terms.extend((float(force@rf),float(ml@rl),float(mr@rr)))
        # Actual native18 order is F0,F1,M0L,M0R,M1L,M1R.
        compatibility.extend(force_residuals+moment_residuals)
    work=math.fsum(work_terms);mixed=work-complementary
    if not np.isfinite((work,complementary,mixed)).all():raise ValueError('finite independent elastic energies required')
    return dict(stored_elastic_energy=float(complementary),mixed_potential=float(mixed),
                compatibility_work_witness=float(work-2.*complementary),
                independent_resultant_residual=np.concatenate(compatibility),
                independent_p_dot_resultant_residual=math.fsum(witness_terms))


def physical_reference(state):
    result=physical_kinetic(state);result.update(elastic_reference(state))
    result['physical_energy']=result['kinetic_energy']+result['stored_elastic_energy']
    return result


def clamp_reference(stage,dt,clamped):
    """Independent left-cell residual work, including consistent nodal inertia.

    Fixed-node external support equals internal plus inertial residual. The
    constrained nodal frame carries a couple but has no invented frame inertia.
    """
    if not clamped:return dict(force_impulse_ns=np.zeros(3),moment_impulse_nms=np.zeros(3),work_j=0.)
    rotations=np.asarray(stage['cell_rotations'],dtype=float);p=np.asarray(stage['resultants'],dtype=float)
    frame=np.asarray(stage['nodal_frames'],dtype=float)[0];rotation=rotations[0,0]
    acceleration=np.asarray(stage['acceleration'],dtype=float)
    force=-rotation@p[0,:3]+SECTION_INERTIA[0]*.5*(2.*acceleration[:3]+acceleration[6:9])/6.
    phi=_log_rotation(rotation.T@frame)
    couple=-rotation@(_left_log_inverse(phi).T@p[0,6:9])
    position=np.asarray(stage['positions'],dtype=float)[0]+np.asarray(stage['position_low'],dtype=float)[0]
    moment=np.cross(position,force)+couple
    if not np.isfinite(np.r_[force,moment]).all():raise ValueError('finite independent support work required')
    return dict(force_impulse_ns=force*dt,moment_impulse_nms=moment*dt,work_j=0.)


def case_velocity(clamped=False):
    """Fixed nonuniform deformation/spin case, not the old single-mode case."""
    x=REFERENCE_POSITIONS[:,0]
    translation=np.column_stack((np.full(5,.01),.02+.03*x*x,.01*x))
    if clamped:translation[0]=0.
    spins=np.array(((1.5,.4,-.3),(1.8,-.2,.5),(1.2,.3,-.4),(1.6,-.4,.2)))
    return np.r_[translation.ravel(),spins.ravel()]


def require_case_definition(owner,clamped):
    """Authenticate raw physical inputs, never a recovered owner quantity."""
    if tuple(owner.nodes)!=(10,20,30,40,50)or owner.n!=5 or owner.e!=2 or owner.size!=78:raise ValueError('registered connected topology required')
    if not np.array_equal(owner.reference_positions,REFERENCE_POSITIONS):raise ValueError('registered straight reference required')
    if tuple(owner.fixed)!=(tuple(range(6))if clamped else ()):raise ValueError('registered homogeneous support required')
    if owner.inertia_order!=24:raise ValueError('original inertia order24 required')
    expected_nodes=((10,20,30),(30,40,50))
    for index,element in enumerate(owner.elements):
        ref=element.operator.reference;expected=REFERENCE_POSITIONS[list(ELEMENT_NODES[index])]
        if tuple(element.node_ids)!=expected_nodes[index]or not np.array_equal(ref.coordinates,expected)or not np.array_equal(ref.nodal_triads,np.tile(np.eye(3),(3,1,1))):raise ValueError('registered physical section frame required')
        if not np.array_equal(element.section.elastic,np.diag(ELASTIC))or not np.array_equal(element.section.metric,np.eye(6))or element.section.yield_force!=1e6 or element.section.hardening!=1.:raise ValueError('registered elastic section required')
        if not np.array_equal(owner.analysis._inertias[element.element_id],np.diag(SECTION_INERTIA)):raise ValueError('registered physical section inertia required')


def _account_vector(account,name):
    return np.asarray(account[name],dtype=float)+np.asarray(account[name+'_low'],dtype=float)


def _finite_errors(errors):
    if not all(math.isfinite(float(value))and value>=0. for value in errors.values()):raise ValueError('nonfinite independent comparison error')
    return max(errors.values())


def assess_trajectory(data,clamped):
    """All accepted public records, with like-energy parity and independent loads.

    Oracle agreement is separate from physical balance. Stored complementary
    energy is used for conservation, mixed potential for reported-energy parity.
    No near-zero floors or amplification normalization are introduced.
    """
    initial_state=data['initial']['state'];initial=physical_reference(initial_state)
    energy_scale,expected_linear,expected_angular=INITIAL_ANALYTIC[bool(clamped)]
    mass=4.;length=2.
    if not math.isfinite(energy_scale)or energy_scale<=0.:raise ValueError('strictly positive independent initial energy required')
    momentum_scale=math.sqrt(2.*mass*energy_scale);angular_scale=length*momentum_scale
    physical=tuple(6*n+a for n in range(5)for a in range(3))+tuple(range(30,42))
    initial_rates=np.asarray(initial_state['velocity'],dtype=float)
    if not np.array_equal(initial_rates[list(physical)],case_velocity(clamped)):raise AssertionError('actual public initial rates differ from the fixed case')
    if not np.array_equal(np.asarray(initial_state['positions']),REFERENCE_POSITIONS)or np.any(np.asarray(initial_state['position_low'])):raise AssertionError('actual public physical reference origin changed')
    initial_errors=dict(kinetic=abs(initial['kinetic_energy']-energy_scale)/energy_scale,
        stored_elastic=abs(initial['stored_elastic_energy'])/energy_scale,
        linear=float(np.linalg.norm(initial['linear_momentum']-expected_linear))/momentum_scale,
        angular=float(np.linalg.norm(initial['angular_momentum']-expected_angular))/angular_scale)
    if _finite_errors(initial_errors)>PARITY_TOLERANCE:raise AssertionError('actual public initializer differs from independent fixed analytic case: '+repr(initial_errors))
    previous=initial;impulse=np.zeros(3);angular_impulse=np.zeros(3);rows=[]
    parity_max=max(initial_errors.values());balance_max=0.
    records=[('initial',data['initial'])]
    for step in data['steps']:records.extend((('stage',step['stage']),('endpoint',step['endpoint'])))
    for kind,record in records:
        ref=physical_reference(record['state']);account=record['accounting']
        errors=dict(kinetic=abs(ref['kinetic_energy']-account['kinetic_energy'])/energy_scale,
                    mixed=abs(ref['mixed_potential']-account['elastic_energy'])/energy_scale,
                    linear=float(np.linalg.norm(ref['linear_momentum']-_account_vector(account,'linear_momentum')))/momentum_scale,
                    angular=float(np.linalg.norm(ref['angular_momentum']-_account_vector(account,'angular_momentum')))/angular_scale)
        maximum=_finite_errors(errors);parity_max=max(parity_max,maximum)
        if maximum>PARITY_TOLERANCE:raise AssertionError('independent reported-quantity parity: '+repr((kind,errors)))
    for step in data['steps']:
        ref=physical_reference(step['endpoint']['state']);support=clamp_reference(step['stage']['state'],step['dt'],clamped)
        actual=step['midpoint_support']
        support_errors=dict(linear=float(np.linalg.norm(np.asarray(actual['force_impulse_ns'])-support['force_impulse_ns']))/momentum_scale,
                            angular=float(np.linalg.norm(np.asarray(actual['moment_impulse_nms'])-support['moment_impulse_nms']))/angular_scale,
                            work=abs(float(actual['work_j'])-support['work_j'])/energy_scale)
        parity_max=max(parity_max,*support_errors.values())
        if _finite_errors(support_errors)>PARITY_TOLERANCE:raise AssertionError('independent clamp reaction parity: '+repr(support_errors))
        impulse+=support['force_impulse_ns'];angular_impulse+=support['moment_impulse_nms']
        perstep=dict(energy=abs(ref['physical_energy']-previous['physical_energy'])/energy_scale,
                     linear=float(np.linalg.norm(ref['linear_momentum']-previous['linear_momentum']-support['force_impulse_ns']))/momentum_scale,
                     angular=float(np.linalg.norm(ref['angular_momentum']-previous['angular_momentum']-support['moment_impulse_nms']))/angular_scale)
        cumulative=dict(energy=abs(ref['physical_energy']-initial['physical_energy'])/energy_scale,
                        linear=float(np.linalg.norm(ref['linear_momentum']-initial['linear_momentum']-impulse))/momentum_scale,
                        angular=float(np.linalg.norm(ref['angular_momentum']-initial['angular_momentum']-angular_impulse))/angular_scale)
        maximum=max(_finite_errors(perstep),_finite_errors(cumulative));balance_max=max(balance_max,maximum)
        rows.append(dict(index=step['index'],perstep=perstep,cumulative=cumulative,support_parity=support_errors,
                         end_time=step['end_time'],physical_energy=ref['physical_energy'],
                         stored_elastic_energy=ref['stored_elastic_energy'],
                         linear_momentum=ref['linear_momentum'].tolist(),angular_momentum=ref['angular_momentum'].tolist(),
                         mixed_minus_stored=ref['compatibility_work_witness'],
                         independent_resultant_residual_norm=float(np.linalg.norm(ref['independent_resultant_residual'])),
                         independent_p_dot_resultant_residual=ref['independent_p_dot_resultant_residual'],
                         reported_stage_residual_norm=step['stage_residual_norm'],
                         reported_endpoint_algebraic_norm=step['endpoint_algebraic_norm']))
        if maximum>BALANCE_TOLERANCE:raise AssertionError('independent physical balance: '+repr(rows[-1]))
        previous=ref
    if not rows or max(row['stored_elastic_energy']for row in rows)<=0.:raise AssertionError('registered deforming connected case must have nonzero stored elastic response')
    return dict(status='PASS_SCOPED_DEVELOPMENT_BALANCES_ONLY',clamped=bool(clamped),scales=dict(energy_j=energy_scale,momentum_ns=momentum_scale,angular_nms=angular_scale,mass_kg=mass,length_m=length),
                parity_tolerance=PARITY_TOLERANCE,balance_tolerance=BALANCE_TOLERANCE,parity_max=parity_max,balance_max=balance_max,initial_fixed_case_errors=initial_errors,steps=rows,engineering_accepted=False)
