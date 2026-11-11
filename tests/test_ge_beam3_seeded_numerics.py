"""Small correctness checks for private V5 numerical successors."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import eigvalsh
from anyfesolver._native_signed_kinetic_factor_modes import solve_signed_kinetic_factor_modes as solve
from anyfesolver._ge_beam3_lifted_kinetic_factor import current_lifted_kinetic_factor
from anyfesolver._ge_beam3_p5_centered.mass import current_rest_mass
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_conservative_backtrack import acceptance
from test_ge_beam3_curved_contrast_probe import make_curved


def test_backtrack_is_exact_extraction_of_preserved_research_function():
    root=Path(__file__).parents[1]
    nodes=[]
    for path in ['docs/reference_cases/ge_beam3_curved_p5_energy_seed_probe.py',
                 'src/anyfesolver/_ge_beam3_conservative_backtrack.py']:
        tree=ast.parse((root/path).read_text())
        nodes.append(ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='acceptance')))
    assert nodes[0]==nodes[1]
    assert acceptance(1.,.9,-.2,1.,2.,.1,.2)=='ACCEPT_ENERGY'
    assert acceptance(1.,1.,-1e-20,1.,2.,.1,.05)=='ACCEPT_ROUNDOFF_RESIDUAL'
    assert acceptance(1.,1.,-1e-20,1.,2.,.1,.1)=='REJECT_ROUNDOFF'
    assert acceptance(1.,.9,.2,1.,2.,.1,.05)=='REJECT_NON_DESCENT'


@pytest.mark.parametrize('angles',[[0.,0.,0.],[.4,-.3,.2]])
def test_current_kinetic_factor_matches_full_lifted_inertia(angles):
    model,inertias=make_curved(100.)
    e=model.mesh.elements[1]; ref=e.core.reference
    q=np.array([rotation(angles),rotation(-np.array(angles)/2)])
    j=inertias[1].copy(); j[0,4]=j[4,0]=1e-4
    b=current_lifted_kinetic_factor(ref,j,q,order=e.core.order)
    m=current_rest_mass(ref,j,e.core.order,ref.coordinates,np.zeros((3,3)),q)
    np.testing.assert_allclose(b.T@b,m,rtol=1e-11,atol=1e-14)
    assert not np.any(b[:,[3,4,5,9,10,11,15,16,17]])
    with pytest.raises(ValueError): b.setflags(write=True)


def test_signed_kinetic_reduction_retains_negative_mode():
    f=np.diag([1.,1e12]); g=np.array([[-3.,1e12],[1e12,0.]])
    result=solve(f,g,np.eye(2),(0,1),(),bounds=(-5.,0.),num_modes=1)
    assert abs(result.eigenvalues[0]+3.)<1e-10
    assert abs(result.full_modes[1,0]/result.full_modes[0,0]+1e-12)<1e-23
    assert result.negative_eigenvalues_retained and not result.production_qualified


def test_signed_trace_with_coupled_kinetic_factor_matches_dense_small_system():
    rng=np.random.default_rng(2026); f=rng.normal(size=(15,6))
    b=np.zeros((8,6)); b[:,:4]=rng.normal(size=(8,4))
    raw=rng.normal(size=(6,6)); g=(raw+raw.T)*.05
    k=f.T@f+g; m=b.T@b
    condensed=k[:4,:4]-k[:4,4:]@np.linalg.solve(k[4:,4:],k[4:,:4])
    result=solve(f,g,b,tuple(range(6)),(4,5),bounds=(-10.,100.),num_modes=4)
    np.testing.assert_allclose(result.eigenvalues,eigvalsh(condensed,m[:4,:4]),rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(result.full_modes.T@m@result.full_modes,np.eye(4),rtol=1e-11,atol=1e-11)


def test_underflow_cannot_hide_algebraic_inertia():
    b=np.diag([1.,1.,1e-200])
    with pytest.raises(ValueError,match='kinetic factor must be exactly zero'):
        solve(np.eye(3),np.zeros((3,3)),b,(0,1,2),(2,),bounds=(-1.,2.),num_modes=2)


@pytest.mark.parametrize('kind',['nonfinite','empty','wide','rank'])
def test_kinetic_factor_bad_inputs_fail_closed(kind):
    b=np.eye(2)
    if kind=='nonfinite': b[0,0]=np.nan
    elif kind=='empty': b=np.zeros((0,2))
    elif kind=='wide': b=np.ones((2,3))
    else: b[0]=0.
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        solve(np.eye(2),np.zeros((2,2)),b,(0,1),(),bounds=(-1.,2.),num_modes=2)
