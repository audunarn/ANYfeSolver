"""Continuum strong/weak agreement and current native spatial modal convergence."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from docs.reference_cases import ge_beam3_continuum_frequency_shooting as reference
from docs.reference_cases.ge_beam3_curved_p5_modal_reference import parabolic_modal_reference
from anyfesolver import _ge_beam3_native_generalized_modal as modal
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_generalized_modal import make,inertia
from test_ge_beam3_schur_line_program import save

CASES=('straight-diagonal','straight-coupled','curved-coupled')
REFINEMENTS=(1,2,4,8)

def data(coupled):
    c=np.diag([1000.,400.,350.,.8,1.,1.2])
    if coupled:c[0,3]=c[3,0]=.04;c[1,5]=c[5,1]=.03
    return c,inertia() if coupled else np.diag([2.,2.,2.,.07,.09,.11])

@pytest.mark.parametrize('family',('axial','torsion'))
def test_analytic_uniform_rod_endpoint(family,tmp_path):
    c,j=data(False)
    index=6 if family=='axial' else 9
    omega=float(np.pi/4*np.sqrt(c[index-6,index-6]/j[index-6,index-6]))
    matrix=reference.generator(0.,0.,np.linalg.inv(c),j,omega*omega)
    endpoint=expm(2*matrix)[:,index]
    assert abs(endpoint[index])<1e-11
    save(tmp_path/'analytic.json',dict(family=family,omega=omega,endpoint=endpoint))

@pytest.mark.parametrize('height',(0.,.2))
def test_hamiltonian_and_rigid_linear_fields(height,tmp_path):
    c,j=data(True);errors=[]
    symplectic=np.block([[np.zeros((6,6)),np.eye(6)],[-np.eye(6),np.zeros((6,6))]])
    for t in (-.73,.17):
        a=reference.generator(t,height,np.linalg.inv(c),j,2.3)
        h=a.T@symplectic+symplectic@a
        errors.append(float(np.linalg.norm(h)/max(1.,np.linalg.norm(a))))
        spin=np.array([.2,-.3,.4]);point=np.array([t,height*(1-t*t),0.])
        field=np.r_[np.array([.7,-.1,.2])+np.cross(spin,point),spin,np.zeros(6)]
        expected=np.r_[np.cross(spin,[1.,-2*height*t,0.]),np.zeros(9)]
        errors.append(float(np.linalg.norm(reference.generator(t,height,np.linalg.inv(c),j,0.)@field-expected)))
    assert max(errors)<=1e-11
    save(tmp_path/'identities.json',dict(height=height,errors=errors))

@pytest.mark.parametrize('kind',('height','profile','asymmetry','mass','seeds','sites','nonfinite'))
def test_reference_rejects_bad_authority(kind):
    c,j=data(True);height=.2;profile='ODE13';seeds=np.arange(1.,7.);sites=np.linspace(-1.,1.,9)
    if kind=='height':height=True
    if kind=='profile':profile='OTHER'
    if kind=='asymmetry':c[1,0]=1.
    if kind=='mass':j[:]=0.
    if kind=='seeds':seeds=seeds[::-1]
    if kind=='sites':sites[0]=-.99
    if kind=='nonfinite':c[0,0]=np.inf
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        reference.solve(height,c,j,seeds,profile=profile,sites=sites)

def test_reference_has_no_native_imports():
    tree=ast.parse(Path(reference.__file__).read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('eval','exec','__import__')
    assert set(imports)<={'dataclasses','time','numpy','scipy.integrate','scipy.optimize'}

def stations(macros):
    g,w=np.polynomial.legendre.leggauss(32);rows=[]
    for index in range(macros):
        for cell in (0,1):
            for a,b in zip(g,w):
                weight=float((a+1)/2)
                t=float(-1+(2*index+cell+weight)/macros)
                rows.append((index+1,cell,weight,t,float(b/(2*macros))))
    return rows

def compare_shapes(model,packet,modes,continuum,height,inertia,macros):
    physical=[];analytic=[];measures=[]
    lookup={float(t):index for index,t in enumerate(continuum.sites)}
    layout=dict(packet.internal_layout)
    for eid,cell,weight,t,measure in stations(macros):
        element=model.mesh.elements[eid];coords=element.operator.reference.coordinates
        b=coords[0]-2*coords[1]+coords[2];lift=.5*weight*(weight-1)*b
        ids=element.node_ids
        left=model.mesh.dof_manager.get_node_dofs(ids[cell])
        right=model.mesh.dof_manager.get_node_dofs(ids[cell+1])
        cell_modes=modes.full_modes[list(layout[eid][3*cell:3*cell+3])]
        velocity=(1-weight)*modes.full_modes[list(left[:3])]+weight*modes.full_modes[list(right[:3])]
        velocity+=np.cross(cell_modes.T,lift).T
        q=np.vstack((velocity,cell_modes))
        jac,_,rotation=reference.geometry(t,height)
        factor=np.linalg.cholesky(inertia).T@rotation.T
        physical.append(factor@q);analytic.append(factor@continuum.fields[:,lookup[t],:6].T)
        measures.append(measure*jac)
    gram=np.zeros((6,6));native=np.zeros((6,6));exact=np.zeros((6,6))
    for p,a,w in zip(physical,analytic,measures):
        gram+=w*(p.T@a);native+=w*(p.T@p);exact+=w*(a.T@a)
    native_error=float(np.linalg.norm(native-np.eye(6)))
    reference_error=float(np.linalg.norm(exact-np.eye(6)))
    assert native_error<=1e-11 and reference_error<=1e-8
    mac=gram*gram/(np.diag(native)[:,None]*np.diag(exact)[None,:])
    assert np.isfinite(mac).all() and np.max(mac)<=1+1e-10
    return dict(mac=mac,native_normalization_error=native_error,reference_orthogonality_error=reference_error)

@pytest.mark.parametrize('case',CASES)
def test_native_frequency_convergence(case,tmp_path):
    height=.2 if case=='curved-coupled' else 0.;coupled=case!='straight-diagonal'
    c,j=data(coupled)
    coarse=parabolic_modal_reference(height,c,j,terms=12,order=96)
    fine=parabolic_modal_reference(height,c,j,terms=16,order=96)
    alternate=parabolic_modal_reference(height,c,j,terms=16,order=64)
    seeds=np.sqrt(fine.squared_frequencies[:6])
    ritz_error=float(np.max(np.abs(np.sqrt(coarse.squared_frequencies[:6])/seeds-1)))
    quadrature_error=float(np.max(np.abs(np.sqrt(alternate.squared_frequencies[:6])/seeds-1)))
    assert ritz_error<1e-7 and quadrature_error<1e-11
    sites=np.unique(np.r_[[-1.,1.],[r[3] for n in REFINEMENTS for r in stations(n)]])
    low=reference.solve(height,c,j,seeds,profile='ODE11',sites=sites)
    high=reference.solve(height,c,j,seeds,profile='ODE13',sites=sites)
    omega=np.sqrt(high.eigenvalues)
    profile_error=float(np.max(np.abs(np.sqrt(low.eigenvalues)/omega-1)))
    method_error=float(np.max(np.abs(seeds/omega-1)))
    assert profile_error<1e-8 and method_error<1e-7
    save(tmp_path/'reference-low.json',low);save(tmp_path/'reference-high.json',high)
    save(tmp_path/'reference-agreement.json',dict(case=case,ritz_error=ritz_error,
        quadrature_error=quadrature_error,profile_error=profile_error,method_error=method_error,seeds=seeds))
    rows=[]
    for macros in REFINEMENTS:
        print(dict(stage='native-modal',case=case,macros=macros),flush=True)
        model,states,masses=make(bool(height),macros,clamped=True,coupled=coupled)
        assert all(np.array_equal(e.section.elastic,c) for e in model.mesh.elements.values())
        assert all(np.array_equal(m,j) for m in masses.values())
        n=model.mesh.dof_manager.total_dofs;before=canonical(states)
        packet,modes=modal.solve_modes(model,states,np.zeros(n),masses,np.zeros(n),num_modes=6)
        assert np.min(modes.eigenvalues)>0 and canonical(states)==before
        errors=np.abs(np.sqrt(modes.eigenvalues)/omega-1)
        shape=compare_shapes(model,packet,modes,high,height,j,macros)
        row=dict(macros=macros,frequency_errors=errors,max_frequency_error=float(np.max(errors)),
            diagonal_mac=np.diag(shape['mac']),minimum_mac=float(np.min(np.diag(shape['mac']))))
        rows.append(row)
        save(tmp_path/('native-n%d.json'%macros),dict(row=row,shape=shape,packet=packet,modes=modes,
            accepted_history_unchanged=True))
        print(dict(stage='comparison',case=case,**{k:v for k,v in row.items() if k in ('macros','max_frequency_error','minimum_mac')}),flush=True)
    assert all(b['max_frequency_error']<a['max_frequency_error'] for a,b in zip(rows,rows[1:]))
    assert rows[-1]['max_frequency_error']<.02 and rows[-1]['minimum_mac']>=.95
    save(tmp_path/'assessment.json',dict(case=case,rows=rows,reference_frequencies=omega,
        production_qualified=False,independent_review='PENDING',full_slenderness_qualified=False))
