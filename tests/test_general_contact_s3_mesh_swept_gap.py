"""Independent squared-gap proof and connected public whole-chord consumer."""
import copy,json,os
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import test_general_contact_s3_mesh_weak_pressure_friction as weak
import examples.general_contact_s3_mesh_weak_pressure_friction as endpoint_example
from examples.general_contact_s3_mesh_swept_gap import build_case
from anyfesolver.general_contact import S3MeshWholeChordGapFrictionSettings
from anyfesolver.general_contact import s3_friction_swept_gap as gap

def _bind(monkeypatch):
    monkeypatch.setattr(weak,'build_case',build_case)
    monkeypatch.setattr(endpoint_example,'build_case',build_case)

def _v(a,b):return tuple((F(x),F(x)) for x in (a,b))
def _p(x,y,z):return (_v(*x),_v(*y),_v(*z))

def test_squared_chord_proof(monkeypatch):
    import time
    started=time.perf_counter()
    progress_root=os.environ.get('AUTOMATIC_RUNTIME_RESULT')
    progress=Path(progress_root).parent/'proof-progress.json' if progress_root else None
    mark=lambda label:progress.write_text(json.dumps(dict(label=label,elapsed_s=time.perf_counter()-started))) if progress else None
    # Target z=t*x; material point x=t,z=t. Endpoints touch, midpoint opens.
    target=(_p((0,0),(0,0),(0,0)),_p((1,0),(0,0),(0,1)),_p((0,0),(1,0),(0,0)))
    point=_p((0,1),(F(1,4),0),(0,1));budget=F(float(5e-5))
    from anyfesolver.general_contact.s3_friction_mesh_coverage import restrict
    at=lambda p,t:tuple(restrict(v,t,t) for v in p)
    assert gap.point_facet_certified(at(point,F(0)),tuple(at(p,F(0)) for p in target),budget)
    assert gap.point_facet_certified(at(point,F(1)),tuple(at(p,F(1)) for p in target),budget)
    numerator=F(1,4);norm2=F(5,4)
    assert numerator**2>budget**2*norm2
    assert not gap.point_facet_certified(point,target,budget)
    assert not gap.point_facet_certified(at(point,F(1,2)),tuple(at(p,F(1,2)) for p in target),budget)
    mark('algebraic interior control complete')
    # Strict full-patch witness: affine skins remain parallel/coplanar with
    # positive chart/union margins while the interior absolute gap exceeds G.
    case=build_case();m,d,s,f,p,r=case
    owner=weak.create_s3_friction_manager(m,weak.GeneralS3PatchManager(m,d,s),f)
    first=np.zeros(owner.n_dof);last=first.copy();a=2**-10;drift=2**-3
    source_ids=owner.normal.law.source_node_ids;target_ids=owner.normal.law.target_node_ids
    for node in (*source_ids,*target_ids):
        x=m.mesh.get_node(node).x;dd=m.mesh.dof_manager.get_node_dofs(node)
        first[dd[2]]=-a*x
        last[dd[2]]=a*(x+drift if node in source_ids else x)
        if node in source_ids:last[dd[0]]=drift
    from anyfesolver.general_contact.s3_friction_mesh_coverage import mesh_moving_coverage
    from anyfesolver.general_contact.s3_friction_dyadic_interval import outward_dyadic_160
    with outward_dyadic_160():assert mesh_moving_coverage(owner,first,last)
    mark('strict patch union certified')
    # 2^-14 is drift's ideal opening contribution; bind actual reference
    # skins and binary work maps independently, including existing tilt.
    assert F(1,2**14)>budget
    def exact_points(adapter,t):
        q=[(1-t)*F(float(x))+t*F(float(y)) for x,y in zip(first,last)]
        A=adapter.work_map.tocsr();coords=[]
        for row,ref in enumerate(adapter.reference_positions.reshape(-1)):
            lo,hi=A.indptr[row:row+2]
            coords.append(F(float(ref))+sum(F(float(v))*q[j] for j,v in zip(A.indices[lo:hi],A.data[lo:hi])))
        return [coords[k:k+3] for k in range(0,len(coords),3)]
    sub=lambda x,y:[a-b for a,b in zip(x,y)]
    cross=lambda x,y:[x[i]*y[j]-x[j]*y[i] for i,j in ((1,2),(2,0),(0,1))]
    midpoint=[]
    for t in (F(0),F(1,2),F(1)):
        for target_adapter in owner.target_adapters.values():
            vertices=exact_points(target_adapter,t);normal=cross(sub(vertices[1],vertices[0]),sub(vertices[2],vertices[0]));nn=sum(v*v for v in normal)
            for source_adapter in owner.source_adapters:
                for pt in exact_points(source_adapter,t):
                    h=sum(x*n for x,n in zip(sub(pt,vertices[0]),normal))
                    if t==F(1,2):
                        assert h>0 and h*h>budget*budget*nn
                        midpoint.append(dict(numerator=str(h),normal_squared=str(nn)))
                    else:assert h*h<=budget*budget*nn
    gap.certify_material_gap(owner,first,first);gap.certify_material_gap(owner,last,last)
    mark('both strict patch endpoints certified')
    with pytest.raises(ValueError,match='material-gap'):owner.require_friction_path(first,last)
    mark('strict patch unsafe interior refused')
    safe=_p((F(1,4),0),(F(1,4),0),(0,0))
    flat=tuple(at(p,F(0)) for p in target)
    assert gap.point_facet_certified(safe,flat,budget)
    with pytest.raises(ValueError,match='normal'):gap.point_facet_certified(safe,(flat[0],)*3,budget)
    # All-part completeness: later failure on every level must not promote first pass.
    fake=SimpleNamespace(n_dof=1,source_adapters=[object()],target_adapters={1:object()},
        normal=SimpleNamespace(settings=SimpleNamespace(maximum_swept_subdivisions=2)),
        friction_settings=SimpleNamespace(finite_material_gap_error_m=5e-5,maximum_material_gap_evaluations=65536),_require_prepared_geometry=lambda:None)
    monkeypatch.setattr(gap,'mapped_path',lambda *a:(flat,None))
    monkeypatch.setattr(gap,'material_point_paths',lambda *a:[safe]);calls=[]
    def partial(*a,**k):
        calls.append(1)
        if len(calls)%2==0:raise ValueError('later unresolved part')
        return True
    monkeypatch.setattr(gap,'point_facet_certified',partial)
    with pytest.raises(ValueError,match='unsafe or unresolved'):gap.certify_material_gap(fake,np.zeros(1),np.zeros(1))
    assert len(calls)==6
    calls.clear();fake.friction_settings.maximum_material_gap_evaluations=1
    with pytest.raises(ValueError,match='cap exhausted'):gap.certify_material_gap(fake,np.zeros(1),np.zeros(1))
    assert len(calls)==1
    result_root=os.environ.get('AUTOMATIC_DOMAIN_RESULT')
    if result_root:Path(result_root).write_text(json.dumps(dict(safe_endpoints=True,
        unsafe_interior_exact_numerator=str(numerator),exact_normal_square=str(norm2),
        strict_patch_drift_opening_contribution_m=str(F(1,2**14)),actual_midpoint_exact_predicates=midpoint,strict_patch_union_prerequisite=True,
        partial_failure_refused=True,point_facet_evaluations=len(calls))))

@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='installed swept stage inputs FRICTION_STAGE/AUTOMATIC_WORKFLOW_EVIDENCE absent: staged whole-chord solve requires the original workflow stage selection')
def test_swept_public(monkeypatch):
    _bind(monkeypatch);weak.test_weak_public(monkeypatch)
    evidence=Path(os.environ['AUTOMATIC_WORKFLOW_EVIDENCE']);stage=os.environ['FRICTION_STAGE']
    saved=json.loads((evidence/(stage+'.json')).read_text())
    assert saved['info']['contact_path']['finite_material_gap_error_scope']=='whole_original_dof_chord'
    for phase in saved['info']['force_displacement_history']:
        cert=phase['friction']['whole_chord_geometry_certificate']
        assert cert['method']==gap.METHOD and cert['history_event_scope']=='not_certified'
        assert cert['certified_segments']<=cert['maximum_segments']

@pytest.mark.skipif(not os.environ.get('FRICTION_STAGE') or not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='saved swept inputs FRICTION_STAGE/AUTOMATIC_WORKFLOW_EVIDENCE absent: saved-pose proof requires the original staged evidence root')
def test_swept_saved_proof(monkeypatch):
    _bind(monkeypatch);weak.test_weak_saved_proof()

def test_swept_safety(monkeypatch):
    from dataclasses import asdict,replace
    m,d,s,f,p,r=build_case();owner=weak.create_s3_friction_manager(m,weak.GeneralS3PatchManager(m,d,s),f)
    with pytest.raises(TypeError,match='finite_material_gap_error_m'):
        S3MeshWholeChordGapFrictionSettings(**{k:v for k,v in asdict(f).items() if k!='finite_material_gap_error_m'})
    for cap in (True,0,-1):
        with pytest.raises(ValueError,match='positive integer'):replace(f,maximum_material_gap_evaluations=cap)
    zero=np.zeros(owner.n_dof);owner.require_friction_path(zero,zero)
    unchanged=copy.deepcopy(owner.checkpoint_state())
    for changed in (replace(f,maximum_material_gap_evaluations=2),replace(f,finite_material_gap_error_m=4e-5)):
        owner.friction_settings=changed
        with pytest.raises(ValueError,match='prepared mesh friction geometry'):owner.require_friction_path(zero,zero)
        assert owner.checkpoint_state()==unchanged
        owner.friction_settings=f
    before=copy.deepcopy(owner.checkpoint_state());u=r['manufactured_pose'].copy()
    for node in (2,3,4,5):u[m.mesh.dof_manager.get_node_dofs(node)[2]]+=1e-3
    with pytest.raises(ValueError,match='material-gap'):owner.require_friction_path(np.zeros(owner.n_dof),u)
    assert owner.checkpoint_state()==before


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE') or not os.environ.get('ENDPOINT_PROFILE_CHECKPOINT') or not Path(os.environ['ENDPOINT_PROFILE_CHECKPOINT']).is_file(),
    reason='archived swept inputs AUTOMATIC_WORKFLOW_EVIDENCE and ENDPOINT_PROFILE_CHECKPOINT absent: safety replay requires the original staged checkpoint')
def test_swept_archive_safety_replay(monkeypatch):
    _bind(monkeypatch);weak.test_weak_safety(monkeypatch)
    # No endpoint-profile checkpoint transplant or reseal.
    old=Path(os.environ['ENDPOINT_PROFILE_CHECKPOINT'])
    incoming=json.loads(old.read_text())['checkpoint']
    from examples.general_contact_s3_mesh_moving_friction import solve_case
    with pytest.raises(ValueError,match='contract|restart|definition|incompat'):solve_case(build_case(),2,incoming)

@pytest.mark.skipif(not os.environ.get('AUTOMATIC_WORKFLOW_EVIDENCE'),
    reason='Archived staged inputs AUTOMATIC_WORKFLOW_EVIDENCE absent; this replay requires its original workflow evidence.')
def test_swept_rollback(monkeypatch):
    _bind(monkeypatch);weak.test_weak_rollback(monkeypatch)
