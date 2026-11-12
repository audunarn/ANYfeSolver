"""Development counterexample: endpoint slope is not an H1 energy trace.

This verifies a limitation, not an accepted rim-contact response. Structural
energy is integrated from production virtual work; reference cap geometry and
energy use a separate scalar circle construction. No coupled solve is claimed.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.fe_core import FEModel
from anyfesolver.general_contact.beam_hemisphere_cylinder_work import hemisphere_cylinder_work


def test_fixed_tip_bubble_changes_rim_energy_at_vanishing_structural_cost():
    b=.02; a=-.003; radius=1.; obstacle_radius=2.0003; ke=1e5
    young=1e3; nu=.3; area=1e-3; shear=young/(2*(1+nu))*area*(5/6)
    direction=np.array((b,0.,math.sqrt(1-b*b)))
    transverse=np.array((-direction[2],0.,b))
    tip=np.array((3.,0.,0.)); axis=np.array((0.,0.,1.)); center=np.zeros(3)
    reference_gap0=3.-direction[2]-obstacle_radius
    reference_gap1=3.-(direction[2]+a*b)/math.sqrt(1+a*a)-obstacle_radius
    reference_change=.5*ke*(reference_gap1**2-reference_gap0**2)
    assert reference_gap0<reference_gap1<0 and reference_change<0
    points,weights=np.polynomial.legendre.leggauss(3)
    rows=[]
    for h in (.02,.01,.005):
        model=FEModel('fixed-tip rim energy discriminator')
        material=model.add_material('elastic',young,nu,density=1.)
        positions=np.array((tip-h*direction,tip-.5*h*direction,tip))
        for node,position in enumerate(positions,1): model.add_node(node,*position)
        element=QuadraticBeamElement(1,(1,2,3),'elastic',
            {'area':area,'Iy':1e-4,'Iz':1e-4,'J':1e-4})
        model.add_element(1,element)
        displacement=np.zeros((3,6)); displacement[1,:3]=-a*h*transverse/4
        u=displacement.ravel()
        energy=0.
        for point,weight in zip(points,weights):
            force,_,_=element.compute_nonlinear_response(model.mesh,
                model.get_material('elastic'),(.5+.5*point)*u,tangent=False)
            energy+=.5*weight*float(force@u)
        expected_energy=h*(shear*a*a/6+young*area*a**4/40)
        deformed=positions+displacement[:,:3]
        before=hemisphere_cylinder_work(positions,2,1,radius,center,axis,obstacle_radius)
        after=hemisphere_cylinder_work(deformed,2,1,radius,center,axis,obstacle_radius)
        assert before.branch==after.branch=='rim'
        assert_allclose((before.gap,after.gap),(reference_gap0,reference_gap1),rtol=0,atol=2e-14)
        assert_allclose(energy,expected_energy,rtol=2e-10,atol=1e-22)
        change=.5*ke*(after.gap**2-before.gap**2)
        assert_allclose(change,reference_change,rtol=2e-9,atol=1e-15)
        # Exact curve derivative t+a(2s-1)v has norm >=1. Its x derivative
        # stays positive, and the radial tube gap stays within the explicit
        # 1e-3 radius budget. Neither regularity nor gap admission removes it.
        assert b>abs(a*transverse[0])
        minimum_tube_gap=3.-h*b-radius-obstacle_radius
        assert minimum_tube_gap>-.001
        rows.append(dict(h_m=h,structural_energy_J=energy,reference_energy_J=expected_energy,
            cap_energy_change_J=change,reference_cap_energy_change_J=reference_change,
            initial_rim_gap_m=before.gap,perturbed_rim_gap_m=after.gap,
            minimum_tube_gap_m=minimum_tube_gap,
            cap_change_to_structural_energy=-change/energy))
    assert_allclose([r['structural_energy_J'] for r in rows],
                    rows[0]['structural_energy_J']*np.array((1.,.5,.25)),rtol=2e-10)
    record=os.environ.get('CONTACT_RIM_TRACE_RECORD')
    if record:
        Path(record).write_text(json.dumps(dict(scope='nonadmission counterexample; no coupled solution',
            bubble_amplitude=a,endpoint_stiffness_N_per_m=ke,rows=rows),indent=2),encoding='utf8')
