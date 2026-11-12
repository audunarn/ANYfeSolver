"""Independent continuum bar/line-contact reference for coupled cylinder work."""
from dataclasses import replace
import json, os, math
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.integrate import quad
from anyfesolver import BoundaryCondition, LoadCase, solve_static_nonlinear
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement, QuadraticBeamElement
from test_general_contact_hemisphere_cylinder_static import _support_case, _line_reference

L=.2; EA=1.; A=.112; R=.11; P=.02; KE=1.


def continuum(line_stiffness):
    beta=math.sqrt(line_stiffness/EA); kappa=KE/(EA*beta); D=A+L-R
    def balance(ell):
        return D/(L-ell)*(math.cosh(beta*ell)+kappa*math.sinh(beta*ell))-(1+P/EA)
    ell=brentq(balance,0.,1./beta,xtol=1e-18)
    d=D/(L-ell); gap=-d/beta*math.sinh(beta*ell)
    force_line=EA*d*(math.cosh(beta*ell)-1.)
    # Independent quadrature of the closed continuous solution, not FE fields.
    line_energy=.5*line_stiffness*quad(lambda s:(d/beta*math.sinh(beta*(s-ell)))**2,
                                     0.,ell,epsabs=1e-20,epsrel=1e-12)[0]
    return dict(tip=R-A+gap,gap=gap,reaction=EA*(d-1),
        line_force=force_line,contact_force=force_line-KE*gap,
        line_energy=line_energy,cap_energy=.5*KE*gap**2,active_length=ell)


def fixture(quadratic,cells,line_stiffness):
    _,definition,settings=_support_case(quadratic,KE)
    model=FEModel('distributed axial cylinder convergence')
    model.add_material('elastic',1e3,.3,density=1.)
    count=(2*cells+1) if quadratic else cells+1
    for i,s in enumerate(np.linspace(0.,L,count),1):model.add_node(i,A+s,0.,0.)
    element_type=QuadraticBeamElement if quadratic else BeamElement
    for i in range(cells):
        nodes=(2*i+1,2*i+2,2*i+3) if quadratic else (i+1,i+2)
        model.add_element(i+1,element_type(i+1,nodes,'elastic',
            {'area':1e-3,'Iy':1e-4,'Iz':1e-4,'J':1e-4}))
    model.add_boundary_condition(BoundaryCondition('root',(count,),
        {key:0. for key in ('ux','uy','uz','rx','ry','rz')}))
    model.add_boundary_condition(BoundaryCondition('guide',tuple(range(1,count)),
        {key:0. for key in ('uy','uz','rx','ry','rz')}))
    definition=replace(definition,surfaces=(replace(definition.surfaces[0],
        element_ids=tuple(range(1,cells+1))),),interactions=(replace(definition.interactions[0],
        line_stiffness=line_stiffness,gap_tolerance=1e-5),))
    load=LoadCase('distributed contact end force');load.add_nodal_load(1,forces=(-P,0.,0.))
    return model,definition,settings,load


@pytest.mark.parametrize('quadratic',(False,True))
def test_spatial_and_enforcement_against_independent_continuum(quadratic):
    rows=[]
    for cells,k in ((1,1e9),(2,1e9),(4,1e9),(4,4e9),(4,16e9)):
        model,definition,settings,load=fixture(quadratic,cells,k)
        result=solve_static_nonlinear(model,load,contact=definition,contact_settings=settings,
            num_steps=4,max_iterations=20,tolerance=1e-10)
        assert result.status=='completed',result.failure_reason
        expected=continuum(k);info=result.info['general_contact']
        tip=result.displacements[model.mesh.dof_manager.get_node_dofs(1)[0]]
        reaction=result.steps[-1].support_reactions['root'][0]
        contact=np.asarray(info['beam_force'])[0]
        assert tip==pytest.approx(expected['tip'],rel=1e-3,abs=1e-12)
        assert reaction==pytest.approx(expected['reaction'],rel=1e-3,abs=1e-12)
        assert contact==pytest.approx(expected['contact_force'],rel=1e-3,abs=1e-12)
        assert abs(reaction+contact-P)/P<=1e-6
        independent_gap=float('inf'); independent_energy=0.; independent_line_force=0.
        for element in model.mesh.elements.values():
            positions=np.array([model.mesh.nodes[n].coords()+result.displacements[
                model.mesh.dof_manager.get_node_dofs(n)[:3]] for n in element.node_ids])
            xs=positions[:,0]
            # Direct Lagrange polynomial extrema, independent of swept/search bounds.
            if len(xs)==2:
                polynomial=np.polynomial.Polynomial([xs[0],xs[1]-xs[0]])
            else:
                polynomial=np.polynomial.Polynomial([xs[0],4*xs[1]-3*xs[0]-xs[2],
                                                      2*xs[0]-4*xs[1]+2*xs[2]])
            points=[0.,1.]+[float(t.real) for t in polynomial.deriv().roots()
                            if abs(t.imag)<1e-12 and 0.<t.real<1.]
            independent_gap=min(independent_gap,min(float(polynomial(t))-R for t in points))
            energy,forces=_line_reference(positions,R,np.zeros(3),np.array((0.,0.,1.)),k,L/cells)
            independent_energy+=energy; independent_line_force-=forces[:,0].sum()
        assert independent_gap>=-1e-5
        assert info['minimum_gap']<=independent_gap+1e-12
        assert info['line_contact_energy']==pytest.approx(independent_energy,rel=1e-4,abs=1e-14)
        assert contact==pytest.approx(independent_line_force-KE*independent_gap,rel=1e-4,abs=1e-12)
        assert info['candidate_endpoints']==2 and info['active_endpoints']==1
        assert info['gap_tolerance_met'] and abs(info['minimum_gap'])<=1e-5
        assert info['maximum_integration_relative_error']<=1e-4
        assert expected['line_force']>.4*P
        rows.append(dict(cells=cells,line_stiffness=k,reference=expected,tip=tip,
            reaction=reaction,contact_force=contact,info=info,
            states=len(result.info['force_displacement_history']),independent_gap=independent_gap,
            independent_line_energy=independent_energy,independent_line_force=independent_line_force))
    assert abs(rows[-1]['info']['minimum_gap'])<abs(rows[-2]['info']['minimum_gap'])<abs(rows[-3]['info']['minimum_gap'])
    record=os.environ.get('CONTACT_APP_RESULT_RECORD')
    if record:Path(record).with_name('continuum_B'+('3' if quadratic else '2')+'.json').write_text(json.dumps(rows,indent=2))
