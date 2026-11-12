"""Independent scalar geometry checks for cancellation-sensitive time covers."""
from decimal import Decimal
from math import cos, sin

import numpy as np
import pytest

from anyfesolver.general_contact.native_event_polynomial import MaterialTimeNumerator, material_time_numerator
from test_general_contact_native_surface_event_sequence import pulse


def test_rotating_round_point_matches_independent_plane_distance_and_rates():
    field,pair,_history,segment=pulse()
    value=material_time_numerator(pair,field.initial,segment.translations,segment.angular,segment.shell_increment)
    corners=pair.target.adapter.current_positions(field.initial.shell_displacement)
    area=float(np.linalg.norm(np.cross(corners[1]-corners[0],corners[2]-corners[0])))
    for t in np.linspace(0,1,17):
        angle=1.6*t-.8
        expected=(area*(.025-.03*cos(angle)),area*.03*1.6*sin(angle),area*.03*1.6**2*cos(angle))
        for order,reference in enumerate(expected):
            # The independently specified nominal angle and the stored SciPy
            # rotation matrix differ by binary64 construction roundoff.
            roundoff=128*np.finfo(float).eps*area*.03*1.6**2
            assert float(value.value(float(t),order))==pytest.approx(reference,abs=roundoff,rel=2e-13)
    for left,right in ((0.,1.),(.25,.4),(.499999999999,.500000000001)):
        bounds=value.enclosure(left,right)
        for t in np.linspace(left,right,11):
            assert bounds.gap_lower<=float(value.value(float(t)))<=bounds.gap_upper
            assert bounds.derivative_lower<=float(value.value(float(t),1))<=bounds.derivative_upper


def test_tiny_strict_separation_resolves_but_exact_tangency_does_not():
    # The literal coefficients provide an independent rational minimum.
    zero=(Decimal(0),)*3
    for offset in ('1e-25','0','-1e-25'):
        # Avoid default-context rounding of .25 +/- 1e-25.
        constant=Decimal('0.2500000000000000000000001') if offset=='1e-25' else (
            Decimal('0.2499999999999999999999999') if offset=='-1e-25' else Decimal('.25'))
        value=MaterialTimeNumerator((constant,Decimal(-1),Decimal(1)),zero,zero,Decimal(0))
        bounds=value.enclosure(.5-1e-14,.5+1e-14)
        if offset=='1e-25':assert bounds.gap_lower>0
        elif offset=='-1e-25':assert bounds.gap_upper<0
        else:
            assert bounds.gap_lower<=0<=bounds.gap_upper
            assert bounds.derivative_lower<0<bounds.derivative_upper
        assert value.value(.5)==Decimal(offset)
        assert value.sign(.5)==(1 if offset=='1e-25' else -1 if offset=='-1e-25' else 0)


def test_bad_time_range_refuses():
    field,pair,_history,segment=pulse()
    value=material_time_numerator(pair,field.initial,segment.translations,segment.angular,segment.shell_increment)
    for bounds in ((.6,.5),(-.1,.5),(0.,1.1),(.5,.5)):
        with pytest.raises(ValueError,match='ordered'):value.enclosure(*bounds)


def test_affine_deformable_triangle_area_and_translating_round_surface():
    from anyfesolver.general_contact.native_surface_history import NativeSurfaceMotionSegment
    field,pair,_history,_segment=pulse()
    corners=pair.target.adapter.current_positions(field.initial.shell_displacement)
    kx,ky,vz=.11,-.07,.001
    motion=corners*np.array((kx,ky,0.))
    work=pair.target.adapter.work_map
    matrix=work.toarray() if hasattr(work,'toarray') else np.asarray(work)
    shell=np.linalg.lstsq(matrix,motion.ravel(),rcond=None)[0]
    np.testing.assert_allclose(pair.target.adapter.work_map@shell,motion.ravel(),atol=1e-17)
    segment=NativeSurfaceMotionSegment(np.tile((0.,0.,vz),(3,1)),np.array(((1.6,0.,0.),(0.,0.,0.))),shell)
    value=material_time_numerator(pair,field.initial,segment.translations,segment.angular,segment.shell_increment)
    area=float(np.linalg.norm(np.cross(corners[1]-corners[0],corners[2]-corners[0])))
    for t in np.linspace(0,1,13):
        phase=1.6*t-.8
        a=area*(1+kx*t)*(1+ky*t);ad=area*(kx+ky+2*kx*ky*t);add=area*2*kx*ky
        g=.025+vz*t-.03*cos(phase);gd=vz+.03*1.6*sin(phase);gdd=.03*1.6**2*cos(phase)
        expected=(a*g,ad*g+a*gd,add*g+2*ad*gd+a*gdd)
        for order,reference in enumerate(expected):
            assert float(value.value(float(t),order))==pytest.approx(reference,abs=128*np.finfo(float).eps*area*.1,rel=2e-13)


def test_precise_sine_range_is_explicitly_bounded():
    zeros=(Decimal(0),)*3
    for w in (Decimal(-1),Decimal(7),Decimal("Infinity"),Decimal("NaN")):
        with pytest.raises(ValueError,match="angular norm"):
            MaterialTimeNumerator(zeros,zeros,zeros,w)


@pytest.mark.parametrize("perturb_skin", [False, True])
def test_literal_target_material_weights_are_not_replaced_by_a_plane_corner(perturb_skin):
    from decimal import localcontext
    from types import SimpleNamespace
    from dataclasses import replace
    from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
    field,pair,_history,segment=pulse()
    offset=.2
    work=pair.target.adapter.work_map.toarray()
    shell=np.linalg.lstsq(work,np.tile((0.,0.,offset),(3,1)).ravel(),rcond=None)[0]
    if perturb_skin:
        shell[2]=np.nextafter(shell[2],np.inf)
    start=replace(field.initial,shell_displacement=field.initial.shell_displacement+shell)
    baseline=(.25,.25,.5);biased=(.25,.25,.4999999999995)
    def numerator(weights):
        point=S3SkinMaterialPoint(pair.target.adapter,weights)
        return material_time_numerator(SimpleNamespace(source=pair.source,target=point),
            start,segment.translations,segment.angular,segment.shell_increment)
    a,b=numerator(baseline),numerator(biased)
    corners=pair.target.adapter.current_positions(start.shell_displacement)
    if perturb_skin:
        assert np.ptp(corners[:,2])>0.
    with localcontext() as context:
        context.prec=50
        vertices=[[Decimal(float(x)) for x in row] for row in corners]
        e1=[x-y for x,y in zip(vertices[1],vertices[0])]
        e2=[x-y for x,y in zip(vertices[2],vertices[0])]
        cross=(e1[1]*e2[2]-e1[2]*e2[1],
               e1[2]*e2[0]-e1[0]*e2[2],
               e1[0]*e2[1]-e1[1]*e2[0])
        # Only the third literal coefficient changes. Use its full triple
        # product, including a deliberately nonhorizontal one-ulp skin.
        expected=(Decimal(baseline[2])-Decimal(biased[2]))*sum(
            c*x for c,x in zip(cross,vertices[2]))
        assert expected>Decimal('1e-14')
        for time in (0.,.4,1.):
            assert abs((b.value(time)-a.value(time))-expected)<Decimal('1e-40')
