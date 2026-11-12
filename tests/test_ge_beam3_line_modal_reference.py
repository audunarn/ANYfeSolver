"""Small continuum modal checks, independent of native beam operators."""
import ast
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_line_modal_reference as modal
from docs.reference_cases import ge_beam3_dead_line_reference as equilibrium
from docs.reference_cases import ge_beam3_curved_linear_ritz_reference as unloaded
from docs.reference_cases.ge_beam3_curved_moment_fixture import reference_section

INERTIA=np.diag([1.,1.,1.,.02,.01,.01]).tolist()


def samples():
    return sorted({-1.,1.,*map(float,modal.sites(64)[0]),*map(float,modal.sites(80)[0])})


@pytest.mark.parametrize('height',[0.,.15])
def test_zero_load_matches_separate_reference_linear_operator(height):
    c=reference_section().tolist()
    base=equilibrium.solve(c,[0.,0.,0.],samples(),height=height)
    k,m,material,geometric,_=modal.build(base,INERTIA)
    old_k,old_m=unloaded.build(c,INERTIA,height=height)
    assert np.linalg.norm(k-old_k)/np.linalg.norm(old_k)<1e-11
    assert np.linalg.norm(m-old_m)/np.linalg.norm(old_m)<1e-11
    assert np.linalg.norm(geometric)<1e-9


@pytest.fixture(scope='module')
def loaded():
    return equilibrium.solve(reference_section().tolist(),[.3,-.2,.1],samples())


def test_loaded_reference_two_polynomial_profiles_and_signed_residuals(loaded):
    a=modal.solve(loaded,INERTIA,degree=14,quadrature=64)
    b=modal.solve(loaded,INERTIA,degree=18,quadrature=80)
    assert np.max(abs(a.eigenvalues-b.eigenvalues)/np.maximum(1.,abs(b.eigenvalues)))<1e-7
    assert max(a.residual,b.residual)<1e-8 and np.min(b.eigenvalues)>0
    assert np.linalg.norm(b.geometric)>1e-3
    np.testing.assert_allclose(b.modes.T@b.mass@b.modes,np.eye(6),atol=1e-11,rtol=1e-11)
    assert not b.production_qualified
    assert b.external_hessian_policy=='ZERO_FOR_ADDITIVE_SPATIAL_POSITION_VARIATION'


def test_balanced_prestress_matches_finite_potential_formula_and_rejects_mismatch():
    from docs.reference_cases.ge_beam3_loaded_ritz_reference import skew,station_tangent
    spin=np.array([.4,-.3,.2]); angle=np.linalg.norm(spin); s=skew(spin)
    q=np.eye(3)+np.sin(angle)/angle*s+(1-np.cos(angle))/angle**2*(s@s)
    c=reference_section(); strain=np.array([.01,.02,-.03,.02,-.01,.03]); stress=c@strain
    n=q@stress[:3]; m=q@stress[3:]
    a,b=modal.balanced_station(q,strain,c,n,m)
    _,_,expected_a,expected_b=station_tangent(q,q@(np.array([1.,0.,0.])+strain[:3]),[1.,0.,0.],strain[3:],c)
    np.testing.assert_allclose(a,expected_a,rtol=1e-11,atol=1e-14)
    np.testing.assert_allclose(b,expected_b,rtol=1e-11,atol=1e-14)
    with pytest.raises(ValueError,match='prestress'): modal.balanced_station(q,strain,c,n+[.01,0.,0.],m)


@pytest.mark.parametrize('mutation',['sample','profile','inertia','degree','quadrature','count'])
def test_invalid_modal_inputs_fail_closed(loaded,mutation):
    state=loaded; mass=INERTIA; kw={}
    if mutation=='sample': state=replace(loaded,parameter=loaded.parameter+.000001)
    if mutation=='profile': state=replace(loaded,profile='BVP7')
    if mutation=='inertia': mass=np.diag([-1.,1.,1.,.02,.01,.01]).tolist()
    if mutation=='degree': kw['degree']=True
    if mutation=='quadrature': kw['quadrature']=12
    if mutation=='count': kw['count']=False
    with pytest.raises((ValueError,np.linalg.LinAlgError)): modal.solve(state,mass,**kw)


def test_reference_imports_no_producer_or_native_recovery():
    tree=ast.parse(Path(modal.__file__).read_text()); imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom): imports.append(node.module)
    assert not any(x.startswith(('anyfesolver','tests')) for x in imports)
    assert set(x for x in imports if x.startswith('docs.'))=={
        'docs.reference_cases.ge_beam3_dead_line_reference','docs.reference_cases.ge_beam3_loaded_ritz_reference'}
