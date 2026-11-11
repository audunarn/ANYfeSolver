"""Analytic zero counts independent of the production geometry adapter."""
from math import acos,ceil,cos,floor,pi,sin

import pytest

from anyfesolver.general_contact.native_surface_crossing import GapSlabEnclosure,certify_single_crossing
from anyfesolver.general_contact.native_surface_support import SupportIntervalBudget


def rotating_family(phase,travel,level):
    # Whole spatial angular strip, including every angle between these limits.
    strip=(-.02,.02)
    def extrema(function,left,right,stationary_offset):
        points=[left,right]
        points.extend(stationary_offset+k*pi for k in range(
            ceil((left-stationary_offset)/pi),floor((right-stationary_offset)/pi)+1))
        values=[function(x) for x in points]
        return min(values),max(values)
    def enclose(left,right):
        low=phase+travel*left+strip[0];high=phase+travel*right+strip[1]
        gl,gu=extrema(cos,low,high,0.)
        dl,du=extrema(lambda x:-travel*sin(x),low,high,pi/2)
        # Conservative arithmetic allowance for this bounded analytic fixture.
        return GapSlabEnclosure(gl-level-1e-12,gu-level+1e-12,dl-1e-12,du+1e-12)
    def roots(offset):
        alpha=acos(level)
        return sorted(t for k in range(-3,4) for sign in (-1,1)
            if 0.<(t:=(sign*alpha+2*pi*k-phase-offset)/travel)<1.)
    return enclose,roots


def test_extremum_away_from_contact_allows_one_transverse_crossing():
    enclose,roots=rotating_family(-.4,2.,.7)
    assert all(len(roots(offset))==1 for offset in (-.02,0.,.02))
    # g' changes sign; the old whole-segment monotonicity proof cannot pass.
    assert enclose(0.,1.).derivative_lower<0<enclose(0.,1.).derivative_upper
    budget=SupportIntervalBudget(128)
    assert certify_single_crossing(enclose,budget=budget)
    assert 1<budget.intervals<128


def test_two_crossings_cannot_pass_even_with_same_signed_endpoints():
    enclose,roots=rotating_family(-1.2,2.4,.7)
    assert all(len(roots(offset))==2 for offset in (-.02,0.,.02))
    assert enclose(0.,0.).gap_upper<0 and enclose(1.,1.).gap_upper<0
    assert not certify_single_crossing(enclose,budget=SupportIntervalBudget(128))


def test_unresolved_tangency_consumes_bounded_budget_and_rejects():
    def tangent(left,right):
        minimum=0. if left<=.5<=right else min((left-.5)**2,(right-.5)**2)
        return GapSlabEnclosure(minimum,max((left-.5)**2,(right-.5)**2),2*(left-.5),2*(right-.5))
    budget=SupportIntervalBudget(8)
    assert not certify_single_crossing(tangent,budget=budget)
    assert budget.intervals==8


def test_strictly_open_gap_needs_no_derivative_sign():
    budget=SupportIntervalBudget(1)
    assert certify_single_crossing(lambda l,r:GapSlabEnclosure(1.,2.,-5.,5.),budget=budget)
    assert budget.intervals==1


def test_spatially_mixed_slopes_yield_before_temporal_budget_is_spent():
    budget=SupportIntervalBudget(128)
    def mixed(left,right):
        return GapSlabEnclosure(-1.,1.,-1.,1.,spatial_derivative_width=1.9,temporal_derivative_width=.1)
    assert not certify_single_crossing(mixed,budget=budget,allow_spatial_refinement=True)
    assert budget.intervals==1


def test_correlated_rate_certifies_crossings_despite_mixed_open_point_slopes():
    # g=x*(1+t)-t on x in[0,1.5]. g_t=x-1 has both signs, but for fixed
    # k=2/3,q=g_t-k*g is bilinear with corner range[-1,-1/3]. At each
    # possible zero it equals g_t. Roots t=x/(1-x) exist only for0<x<.5.
    def enclose(left,right):
        values=[x*(1+t)-t for x in (0.,1.5) for t in (left,right)]
        rates=[(x-1)-(2/3)*(x*(1+t)-t) for x in (0.,1.5) for t in (left,right)]
        return GapSlabEnclosure(min(values),max(values),-1.,.5,
            zero_derivative_lower=min(rates)-1e-14,zero_derivative_upper=max(rates)+1e-14)
    budget=SupportIntervalBudget(1)
    assert certify_single_crossing(enclose,budget=budget)
    assert budget.intervals==1


@pytest.mark.parametrize('lower,upper',[(None,1.),(1.,None),(float('nan'),1.),(2.,1.)])
def test_invalid_zero_set_bounds_rejected(lower,upper):
    with pytest.raises(ValueError,match='zero-set'):
        GapSlabEnclosure(-1.,1.,-1.,1.,zero_derivative_lower=lower,zero_derivative_upper=upper)


@pytest.mark.parametrize('values',[(float('nan'),1.,0.,1.),(2.,1.,0.,1.),(0.,1.,2.,1.)])
def test_invalid_enclosures_rejected(values):
    with pytest.raises(ValueError,match='ordered'):
        GapSlabEnclosure(*values)


@pytest.mark.parametrize('phase,travel,expected',[(-.4,2.,True),(-1.2,2.4,False)])
def test_actual_retained_geometry_single_crossing_and_hidden_pulse(phase,travel,expected):
    import numpy as np
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_support import segment_gap_slab_enclosure,history_endpoint_support_certified
    from test_general_contact_native_surface_support import _field
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((travel,0.,0.),(0.,0.,0.))),np.zeros(18))
    press=NativeSurfaceMotionSegment(np.tile((0.,0.,-.029),(3,1)),np.zeros((2,3)),np.zeros(18))
    field=_field(height=.05,rotation=phase,segments=(press,segment))
    angles=field.patch.rule.definition['angle_range']
    assert not history_endpoint_support_certified(field)
    enclose=segment_gap_slab_enclosure(field,0,(-.8,-.2),angles,1)
    # Independent physical formula is .021-.03*cos(phase+travel*t+offset).
    _,roots=rotating_family(phase,travel,.7)
    assert all(len(roots(offset))==(1 if expected else 2) for offset in (-.02,0.,.02))
    for left,right in ((0.,1.),(.125,.25),(.5,.75)):
        bounds=enclose(left,right)
        for t in np.linspace(left,right,13):
            for offset in np.linspace(-.02,.02,9):
                theta=phase+travel*t+offset
                assert bounds.gap_lower<=.021-.03*cos(theta)<=bounds.gap_upper
                assert bounds.derivative_lower<=.03*travel*sin(theta)<=bounds.derivative_upper
    assert certify_single_crossing(enclose,budget=SupportIntervalBudget(128)) is expected


def test_actual_slab_adapter_rejects_wrong_half_cell():
    import numpy as np
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_support import segment_gap_slab_enclosure
    from test_general_contact_native_surface_support import _field
    field=_field(segments=(NativeSurfaceMotionSegment(np.zeros((3,3)),np.zeros((2,3)),np.zeros(18)),))
    with pytest.raises(ValueError,match='half-cell'):
        segment_gap_slab_enclosure(field,1,(-.8,-.2),field.patch.rule.definition['angle_range'],0)


def test_deforming_target_subslab_uses_original_time_derivative():
    import numpy as np
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_support import segment_gap_slab_enclosure
    from test_general_contact_native_surface_support import _field
    shell=np.zeros(18);shell[8]=.01;shell[14]=-.006;shell[12]=.025
    translations=np.tile((.015,0.,-.002),(3,1));omega=.2
    segment=NativeSurfaceMotionSegment(translations,np.array(((omega,0.,0.),(0.,0.,0.))),shell)
    field=_field(height=.08,rotation=.1,segments=(segment,))
    enclose=segment_gap_slab_enclosure(field,0,(-.8,-.2),field.patch.rule.definition['angle_range'],0)
    bounds=enclose(.3125,.375)
    adapter=field.patch.target.adapter
    corners0=adapter.current_positions(field.initial.shell_displacement)
    motion=np.asarray(adapter.work_map@shell).reshape(3,3)
    frame=field.initial.cell_rotations[0]@field.patch.source.reference.nodal_triads[0]
    def independent_gap(t,xi,angle):
        corners=corners0+t*motion
        cross=np.cross(corners[1]-corners[0],corners[2]-corners[0])
        normal=cross/np.sqrt(cross@cross)
        alpha=xi+1
        base=(1-alpha)*(field.initial.positions[0]+field.initial.position_low[0])+alpha*(field.initial.positions[1]+field.initial.position_low[1])+t*translations[0]
        c,s=np.cos(omega*t),np.sin(omega*t)
        rotation=np.array(((1.,0.,0.),(0.,c,-s),(0.,s,c)))
        arm=rotation@frame@np.array((0.,.03*cos(angle),.03*sin(angle)))
        return normal@(base+arm-corners[0])
    for t in np.linspace(.3125,.375,9):
        for xi in (-.8,-.53,-.2):
            for angle in np.linspace(*field.patch.rule.definition['angle_range'],7):
                gap=independent_gap(t,xi,angle)
                derivative=independent_gap(t+1e-25j,xi,angle).imag/1e-25
                assert bounds.gap_lower<=gap<=bounds.gap_upper
                assert bounds.derivative_lower<=derivative<=bounds.derivative_upper


@pytest.mark.parametrize('translation',[0.,float(2**20)])
def test_convex_origin_thin_translated_target_against_high_precision(translation):
    from dataclasses import replace
    from fractions import Fraction
    from decimal import Decimal,localcontext
    import numpy as np
    from scipy.spatial.transform import Rotation
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceHistoryField,NativeSurfaceMotionSegment
    from anyfesolver.general_contact.native_surface_patch import NativeS3MaterialSurfacePatch
    from anyfesolver.general_contact.native_surface_quadrature import NativeRoundSurfaceRule
    from anyfesolver.general_contact.native_surface_support import segment_gap_slab_enclosure,_convex_support_origin
    from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
    from test_general_contact_native_surface_support import _field
    base=_field()
    # Dyadic vertices and translations preserve this represented thin geometry.
    corners=np.array(((-2.,-.0625,-.0625),(2.,-.0625,-.0625),(-2.,.0625,-.0625)))+translation
    adapter=replace(base.patch.target.adapter,reference_positions=corners)
    rule=NativeRoundSurfaceRule(7,base.patch.source.reference,.03,
        xi_range=(-.55,-.5),angle_range=base.patch.rule.definition['angle_range'],axial_order=1,angular_order=2)
    positions=base.initial.positions.copy();positions[:,0]=(-.5625,.4375,1.4375)
    positions[:,1]=0.;positions[:,2]=-.0625+3/128;positions+=translation
    rotations=base.initial.cell_rotations.copy();rotations[0]=Rotation.from_rotvec((-.8,0.,0.)).as_matrix()
    initial=replace(base.initial,positions=positions,cell_rotations=rotations)
    targets=[]
    edges=np.column_stack((corners[1]-corners[0],corners[2]-corners[0]))
    for sample in rule.samples:
        point=sample.station.evaluate(initial.positions,initial.position_low,initial.cell_rotations).surface_position
        uv=np.linalg.lstsq(edges,point-corners[0],rcond=None)[0]
        targets.append(S3SkinMaterialPoint(adapter,(1-uv.sum(),*uv)))
    patch=NativeS3MaterialSurfacePatch(rule,targets,normal_stiffness_Pa_per_m=1000.,
        friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.,activation_policy='transverse_events',gap_check='straight_patch')
    shell=np.zeros(18);shell[8]=2**-12;shell[14]=-2**-13;shell[12]=2**-10
    segment=NativeSurfaceMotionSegment(np.zeros((3,3)),np.array(((1.6,0.,0.),(0.,0.,0.))),shell)
    field=NativeSurfaceHistoryField(patch,initial,(segment,))
    weights=_convex_support_origin(initial,0,rule.definition['xi_range'],corners)
    assert min(weights)>=0 and Fraction(weights[0])+Fraction(weights[1])<=1
    assert sum(weights)>.95  # Exercise a selected origin near the simplex edge.
    enclose=segment_gap_slab_enclosure(field,0,rule.definition['xi_range'],rule.definition['angle_range'],0)
    motion=np.asarray(adapter.work_map@shell).reshape(3,3)
    with localcontext() as context:
        context.prec=70
        def dec(v):return Decimal.from_float(float(v))
        def vec(values):return [dec(v) for v in values]
        def add(a,b):return [x+y for x,y in zip(a,b)]
        def sub(a,b):return [x-y for x,y in zip(a,b)]
        def mul(a,v):return [a*x for x in v]
        def dot(a,b):return sum(x*y for x,y in zip(a,b))
        def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        def trig(x):
            # Independent Decimal Taylor oracle;100terms on |x|<5 suffice
            # far beyond70digits (factorial remainder <1e-230).
            ss=st=x;cc=ct=Decimal(1)
            for k in range(1,100):
                st*=-x*x/Decimal((2*k)*(2*k+1));ss+=st
                ct*=-x*x/Decimal((2*k-1)*(2*k));cc+=ct
            return ss,cc
        frame=[vec(row) for row in initial.cell_rotations[0]@patch.source.reference.nodal_triads[0]]
        def oracle(time,xi,angle):
            t=dec(time);a=dec(xi)+1;theta=dec(angle)
            cs=[add(vec(c),mul(t,vec(v))) for c,v in zip(corners,motion)]
            e1,e2=sub(cs[1],cs[0]),sub(cs[2],cs[0]);cr=cross(e1,e2)
            area=dot(cr,cr).sqrt();n=mul(1/area,cr)
            cp=add(cross(vec(motion[1]-motion[0]),e2),cross(e1,vec(motion[2]-motion[0])))
            nd=mul(1/area,sub(cp,mul(dot(n,cp),n)))
            w=dec(1.6);s,c=trig(w*t);sa,ca=trig(theta)
            arm0=[dot(row,[Decimal(0),dec(.03)*ca,dec(.03)*sa]) for row in frame]
            arm=[arm0[0],c*arm0[1]-s*arm0[2],s*arm0[1]+c*arm0[2]]
            point=add(add(mul(1-a,vec(initial.positions[0])),mul(a,vec(initial.positions[1]))),arm)
            velocity=sub(cross([w,Decimal(0),Decimal(0)],arm),vec(motion[0]))
            return dot(n,sub(point,cs[0])),dot(nd,sub(point,cs[0]))+dot(n,velocity)
        for time in (.1875,.3125,.8125):
            bounds=enclose(time-.015625,time+.015625)
            for xi in (-.55,-.525,-.5):
                for angle in (*rule.definition['angle_range'],1.5*pi):
                    gap,rate=oracle(time,xi,angle)
                    assert dec(bounds.gap_lower)<=gap<=dec(bounds.gap_upper)
                    assert dec(bounds.derivative_lower)<=rate<=dec(bounds.derivative_upper)
        # The actual history has an open/closed/open pulse in both placements.
        assert oracle(0.,-.525,1.5*pi)[0]>0 and oracle(.5,-.525,1.5*pi)[0]<0 and oracle(1.,-.525,1.5*pi)[0]>0
    assert not certify_single_crossing(enclose,budget=SupportIntervalBudget(128))
