"""Supported-only allocation repair; frozen-wheel numerical comparator."""
import ast
import os
from pathlib import Path
import zipfile
import numpy as np
import pytest
from scipy import sparse
from anyfesolver import BoundaryCondition,FixedSupport
from anyfesolver.fe_core import FEModel
from anyfesolver.elements import BeamElement
import anyfesolver.assembly as assembly

def model(count, supported):
    result=FEModel('supported-mode-storage')
    result.add_material('m',210.e9,.3,density=7850.)
    for i in range(count):
        a,b=2*i+1,2*i+2
        result.add_node(a,0.,float(i),0.)
        result.add_node(b,1.,float(i),0.)
        result.add_element(i+1,BeamElement(i+1,[a,b],'m',{'area':.01,'Iy':1.e-5,'Iz':1.e-5,'J':2.e-5}))
    roots=[2*i+1 for i in supported];tips=[2*i+2 for i in supported]
    if roots:
        result.add_boundary_condition(FixedSupport('bases',roots))
        result.add_boundary_condition(BoundaryCondition('axial',tips,{'uy':0.,'uz':0.,'rx':0.,'ry':0.,'rz':0.}))
    result.apply_boundary_conditions()
    return result

@pytest.fixture
def legacy():
    path=os.environ.get('RIGID_LEGACY_WHEEL')
    if not path:pytest.skip('Independent archived-wheel comparator needs RIGID_LEGACY_WHEEL')
    with zipfile.ZipFile(path) as archive:source=archive.read('anyfesolver/assembly.py').decode()
    tree=ast.parse(source)
    names={'_rigid_body_modes_full','_orthonormalize_columns','build_reduced_rigid_body_modes'}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ImportFrom)) and (getattr(node,'name','') in names or isinstance(node,ast.ImportFrom) and node.module=='__future__')]
    assert len([node for node in selected if isinstance(node,ast.FunctionDef)])==3
    namespace=dict(vars(assembly))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(Path(path)/'anyfesolver/assembly.py'),'exec'),namespace)
    return namespace

@pytest.mark.parametrize('supported',[(),(0,),(1,),(0,1)])
@pytest.mark.parametrize('transformed',[False,True])
def test_frozen_wheel_bits_diagnostics_and_model_observation_order(legacy,supported,transformed,monkeypatch):
    actual=model(2,supported);n=actual.mesh.dof_manager.total_dofs
    _,_,transform,_,independent,_=assembly.build_constraint_transformation(sparse.eye(n,format='csr'),np.zeros(n),actual)
    if not transformed:transform=None
    original_get=actual.mesh.get_node;observations=[]
    def observed(node_id):observations.append(node_id);return original_get(node_id)
    monkeypatch.setattr(actual.mesh,'get_node',observed)
    old,old_info=legacy['build_reduced_rigid_body_modes'](actual,independent,n,transformation=transform)
    prior_calls=list(observations);observations.clear()
    new,new_info=assembly.build_reduced_rigid_body_modes(actual,independent,n,transformation=transform)
    assert new.dtype==old.dtype and new.shape==old.shape and new.tobytes()==old.tobytes()
    assert new_info==old_info and observations==prior_calls
    if len(supported)==2:
        assert new.shape[1]==0
        assert new_info['constraint_compatibility_rank']==(0 if transformed else 12)
    else:
        new_full,new_components=assembly._rigid_body_modes_full(actual,n)
        old_full,old_components=legacy['_rigid_body_modes_full'](actual,n)
        assert new_full.shape==old_full.shape and new_full.tobytes()==old_full.tobytes()
        assert new_components==old_components

def test_supported_10000_topology_has_no_dense_component_grid(monkeypatch):
    # Actual physical topology only: no stiffness assembly, recovery, load
    # patterns, solver or large numerical arrays are executed here.
    actual=model(10000,range(10000));n=actual.mesh.dof_manager.total_dofs
    original=np.zeros;allocations=[]
    def bounded(shape,*args,**kwargs):
        allocations.append(shape)
        if isinstance(shape,tuple) and len(shape)==2 and shape[0]>=120000:
            assert shape[1]==0,'dense supported component grid attempted'
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(np,'zeros',bounded)
    independent=np.asarray(actual.mesh.dof_manager.get_free_dofs(),dtype=np.intp)
    modes,info=assembly.build_reduced_rigid_body_modes(actual,independent,n)
    assert modes.shape==(10000,0) and info['rank']==0
    assert info['component_count']==10000 and info['constraint_compatibility_rank']==60000
    assert all(row['supported'] and row['candidate_modes']==0 for row in info['components'])
    assert (120000,0) in allocations


def test_original_affine_rotated_constraints_match_frozen_wheel(legacy):
    actual=FEModel('rotated-affine');actual.add_node(1,0.,0.,0.)
    ux,uy=actual.mesh.get_node(1).dofs[:2];c=float(np.sqrt(.5))
    actual.add_constraint_equation(terms=((ux,c),(uy,c)),rhs=1.,source_id='local-x')
    actual.add_constraint_equation(terms=((uy,c),(ux,-c)),rhs=2.,source_id='local-y')
    reduced,_,transform,_,independent,_=assembly.build_constraint_transformation(sparse.eye(6,format='csr'),np.zeros(6),actual)
    old,old_info=legacy['build_reduced_rigid_body_modes'](actual,independent,6,transformation=transform)
    new,new_info=assembly.build_reduced_rigid_body_modes(actual,independent,6,transformation=transform)
    assert old.shape==new.shape and old.tobytes()==new.tobytes() and old_info==new_info
    assert new.shape[1]==4


def test_fully_constrained_early_return_matches_original(legacy):
    actual=model(1,(0,));actual.add_boundary_condition(FixedSupport('tip',[2]));actual.apply_boundary_conditions()
    independent=np.asarray(actual.mesh.dof_manager.get_free_dofs(),dtype=np.intp)
    assert independent.size==0
    old,old_info=legacy['build_reduced_rigid_body_modes'](actual,independent,12)
    new,new_info=assembly.build_reduced_rigid_body_modes(actual,independent,12)
    assert old.shape==new.shape and old.tobytes()==new.tobytes() and old_info==new_info
