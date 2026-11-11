"""Equation checks independent of beam FE operators; no candidate solve."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from docs.reference_cases import ge_beam3_spatial_continuum as p
from docs.reference_cases.ge_beam3_curved_p5_arch_reference import equations as planar

def state(x):
    r,q=p.reference(x);return np.vstack((r,q,np.zeros((6,len(x)))))

def test_curved_stress_free_identity():
    x=np.linspace(-1.,1.,31);y=state(x);actual=p.equations(x,y)
    theta=np.arctan(-.2*x);rate=-.2/(1+.04*x*x);c=np.cos(theta/2);s=np.sin(theta/2)
    expected=np.vstack((np.ones_like(x),-.2*x,np.zeros_like(x),
        rate*np.array([-s,-s,c,c])/(2*np.sqrt(2.)),np.zeros((6,len(x)))))
    assert np.max(abs(actual-expected))<1e-14

@pytest.mark.parametrize('theta',(-.7,0.,.4))
def test_planar_specialization(theta):
    x=np.array([-.8,-.2]);y=state(x)
    q=np.array([np.cos(theta/2),np.cos(theta/2),np.sin(theta/2),np.sin(theta/2)])/np.sqrt(2.)
    y[3:7]=q[:,None];y[7:10]=np.array([-.2,-.015,0.])[:,None];y[10:13]=np.array([0.,0.,.004])[:,None]
    actual=p.equations(x,y);flat=planar(x,np.array([y[0],y[1],np.full(2,theta),y[12]]),[-.2,-.015],.1,1000.,400.,.01)
    assert np.max(abs(actual[:2]-flat[:2]))<1e-14
    assert np.max(abs(actual[12]-flat[3]))<1e-14
    dq=.5*flat[2]*np.array([-q[2],-q[2],q[0],q[0]])[:,None]
    assert np.max(abs(actual[3:7]-dq))<1e-14

def test_force_moment_first_integral():
    x=np.array([-.3,.4]);y=state(x);y[7:13]=np.arange(12).reshape(6,2)/100.
    rhs=p.equations(x,y)
    assert np.max(abs(rhs[10:13]+np.cross(rhs[:3],y[7:10],axis=0)+np.cross(y[:3],rhs[7:10],axis=0)))<1e-14

def test_rigid_spatial_covariance():
    x=np.array([-.7,.1]);y=state(x);y[7:13]=np.arange(12).reshape(6,2)/100.
    q=Rotation.from_rotvec([.7,-.9,1.2]).as_quat()[[3,0,1,2]];s=p.matrix(q);changed=y.copy()
    for slots in (slice(0,3),slice(7,10),slice(10,13)):changed[slots]=s@y[slots]
    changed[3:7]=p.product(q[:,None],y[3:7])
    rhs=p.equations(x,y);expected=rhs.copy()
    for slots in (slice(0,3),slice(7,10),slice(10,13)):expected[slots]=s@rhs[slots]
    expected[3:7]=p.product(q[:,None],rhs[3:7])
    assert np.max(abs(p.equations(x,changed)-expected))<1e-12

def test_boundary_count_and_no_smeared_point_force():
    a=np.concatenate([state(np.array([x]))[:,0] for x in p.BREAKS[:-1]])
    b=np.concatenate([state(np.array([x]))[:,0] for x in p.BREAKS[1:]])
    assert p.boundary(a,b,np.array([0.]),0.).shape==(53,)
    assert np.max(abs(p.boundary(a,b,np.array([0.]),0.)))<1e-14
    altered=p.boundary(a,b,np.array([.02]),0.)
    assert np.count_nonzero(abs(altered)>1e-14)==1 and np.min(altered)==-.02

def test_no_mechanics_import():
    tree=ast.parse(Path(p.__file__).read_bytes())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('anyfesolver','tests','docs'))
        if isinstance(node,ast.Import):assert all(not n.name.startswith(('anyfesolver','tests','docs')) for n in node.names)

@pytest.mark.parametrize('knots',(np.linspace(0.,1.,65),np.linspace(0.,1.,385),np.array([0.,.03,.21,.39,.9,1.])))
def test_validation_points_not_collocation_nodes_or_midpoints(knots):
    sites=p.validation_grid(knots).reshape(-1,2)
    assert len(sites)==len(knots)-1
    for i,row in enumerate(sites):
        fractions=(row-knots[i])/(knots[i+1]-knots[i])
        assert min(abs(fractions[:,None]-np.array([0.,.5,1.])[None]).ravel())>.2
    # The previous fixed257-point check is exactly nodes/midpoints at384cells.
    if len(knots)==385:assert np.max(abs(np.linspace(0.,1.,257)*768-np.round(np.linspace(0.,1.,257)*768)))<1e-10
