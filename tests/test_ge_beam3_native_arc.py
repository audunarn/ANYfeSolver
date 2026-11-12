"""Actual objective native arc continuation and full predictor/path replay."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from scipy import sparse
from anyfesolver._ge_beam3_native_arc import ArcProgram,solve_arc,direction,capture,_ARC
from anyfesolver._ge_beam3_native_arc_restart import encode_checkpoint,decode_checkpoint
from anyfesolver._ge_beam3_native_translation import _PATH
from anyfesolver._ge_beam3_native_generalized_program import _PROGRAM
from anyfesolver._ge_beam3_native_generalized_combined_couples import _RUN
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern,_ACTIVE
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver._ge_beam3_p5.algebra import rotation
from test_ge_beam3_native_translation import bar as source_bar
from test_ge_beam3_native_generalized_restart import make,pattern
from test_ge_beam3_schur_line_program import save


def bar(sign=1.):
    m,p=source_bar();return m,ArcProgram((.15,.2),1.,p.distributed,initial_sign=sign)


def packet(result):
    return dict(status=result.status,cursor=result.completed_steps,parameter=result.parameter,u=result.displacements,
        reaction=result.physical_imbalance,checkpoint_sha256=sha256(result.checkpoint).hexdigest(),failure=result.failure,events=result.events)


@pytest.fixture(scope='module')
def saved_bar(tmp_path_factory):
    root=tmp_path_factory.mktemp('arc-bar');m,p=bar();r=solve_arc(m,p)
    save(root/'result.json',packet(r));save(root/'checkpoint.json',r.checkpoint)
    assert r.status=='completed',r.failure
    return root,p,r


def test_analytical_bar_and_prefix(saved_bar):
    root,p,r=saved_bar;m,_=bar();chain,records=decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert encode_checkpoint(m,p,chain,records)==r.checkpoint
    unit=np.zeros(18);unit[6]=.09375;unit[12]=.125
    norm=np.sqrt(1.+np.dot(unit,unit)/3);errors=[]
    for index,state in enumerate(chain[1:]):
        expected=sum(p.steps[:index+1])/norm
        errors.extend((abs(records[index]['parameter']-expected),float(np.max(np.abs(state['displacements']-unit*expected)))))
    assert max(errors)<=1e-11
    prefix=encode_checkpoint(m,p,chain[:2],records[:1]);fresh,_=bar()
    resumed=solve_arc(fresh,p,checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    assert resumed.status=='completed' and resumed.checkpoint==r.checkpoint,resumed.failure
    save(root/'analytical.json',dict(errors=errors,parameters=[v['parameter'] for v in records],prefix_resume_byte_identical=True))


def test_compression_bar(tmp_path):
    m,p=bar(-1.);r=solve_arc(m,p);save(tmp_path/'result.json',packet(r));assert r.status=='completed',r.failure
    fresh,_=bar(-1.);chain,records=decode_checkpoint(fresh,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    expected=-sum(p.steps)/np.sqrt(1.+(.09375**2+.125**2)/3)
    assert abs(r.parameter-expected)<=1e-11 and all(v['parameter']<0. for v in records)
    save(tmp_path/'compression.json',dict(expected=expected,actual=r.parameter,checkpoint=r.checkpoint.decode('ascii')))


@pytest.mark.parametrize('kind',['hash','duplicate','nonfinite','schema','metric','direction','orientation','parameter','step','arc-residual','program','history','missing'])
def test_checkpoint_mutation(saved_bar,kind):
    root,p,r=saved_bar;m,_=bar();v=json.loads(r.checkpoint);raw=r.checkpoint
    if kind=='hash':pass
    elif kind=='duplicate':raw=b'{"schema":"bad",'+raw[1:]
    elif kind=='nonfinite':raw=b'{"bad":NaN}'
    else:
        if kind=='schema':v['schema']='GE_BEAM3_NATIVE_GENERALIZED_TRANSLATION_RESTART_V1'
        elif kind=='metric':v['metric'][0]*=2
        elif kind=='direction':
            a=np.array(v['records'][0]['direction']);a[7]+=.001;a/=np.sqrt(np.dot(np.array(v['metric']),a*a));v['records'][0]['direction']=a.tolist()
        elif kind=='orientation':v['records'][0]['direction']=[-x for x in v['records'][0]['direction']]
        elif kind=='parameter':v['records'][0]['parameter']+=.1
        elif kind=='step':v['records'][0]['step_size']+=.01
        elif kind=='arc-residual':v['records'][0]['arc_residual']+=1e-10
        elif kind=='program':v['program']['parameter_scale']=2.;v['program_sha256']=sha(v['program'])
        elif kind=='history':v['accepted_chain']['snapshots'][1]['states'][0]['state']['epoch']+=1
        elif kind=='missing':v['records'].pop()
        v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'});raw=canonical(v)
    with pytest.raises(ValueError):decode_checkpoint(m,p,raw,expected_sha256='0'*64 if kind=='hash' else sha256(raw).hexdigest())
    save(root/(kind+'-rejected.json'),dict(kind=kind,rejected=True))


def test_fold_predictor_orientation(tmp_path):
    # F(x,lambda)=x*x+lambda-1, not a beam qualification fixture.
    previous=np.array([-1.,.2]);previous/=np.linalg.norm(previous);rows=[]
    for x in (.1,0.,-.1):
        tangent=direction(sparse.csr_matrix([[2*x]]),np.array([1.]),np.array([0]),previous,np.ones(2))
        assert abs(2*x*tangent[0]+tangent[1])<=1e-15 and tangent[0]<0 and previous@tangent>0
        rows.append(dict(x=x,tangent=tangent));previous=tangent
    assert rows[0]['tangent'][1]>0 and rows[1]['tangent'][1]==0 and rows[2]['tangent'][1]<0
    save(tmp_path/'fold.json',dict(rows=rows,beam_postbuckling_claim=False))


def test_rigid_frame_covariance(tmp_path):
    from anyfesolver.fe_core import FEModel
    from anyfesolver._ge_beam3_native_generalized_element import NativeGeneralizedStaticElement
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
    m,_=bar();load=DistributedPattern(LinePattern(((1,.4,-.1,.15),)),());moments=SpatialNodalMoments(((3,.1,.15,-.12),))
    p=ArcProgram((.15,.2),1.,load,moments);r=solve_arc(m,p);assert r.status=='completed',r.failure
    s=rotation([.4,-.2,.7]);shift=np.array([2.,-3.,1.]);moved=FEModel('native-arc-rotated')
    for i,node in m.mesh.nodes.items():moved.add_node(i,*(s@node.coords()+shift))
    for i,e in m.mesh.elements.items():
        ref=Reference(np.array([s@x+shift for x in e.operator.reference.coordinates]),np.array([s@q for q in e.operator.reference.nodal_triads]))
        made=NativeGeneralizedStaticElement(i,e.node_ids,ref,e.section,order=4);moved.add_element(i,made);moved.materials[made.material_name]=made.section
    for bc in m.boundary_conditions:moved.add_boundary_condition(bc)
    load2=DistributedPattern(LinePattern(((1,*map(float,s@np.array([.4,-.1,.15]))),)),())
    p2=ArcProgram(p.steps,1.,load2,SpatialNodalMoments(((3,*map(float,s@np.array([.1,.15,-.12]))),)))
    changed=solve_arc(moved,p2);assert changed.status=='completed',changed.failure
    expected=np.array([s@v for v in r.displacements.reshape(-1,3)]).ravel()
    reaction=np.array([s@v for v in r.physical_imbalance.reshape(-1,3)]).ravel()
    errors=dict(displacement=float(np.max(np.abs(changed.displacements-expected))),reaction=float(np.max(np.abs(changed.physical_imbalance-reaction))),parameter=abs(changed.parameter-r.parameter))
    assert max(errors.values())<=1e-11
    chain,_=decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    a=chain[1]['displacements'].reshape(-1,6)[:,3:];b=(chain[2]['displacements']-chain[1]['displacements']).reshape(-1,6)[:,3:]
    noncommuting=max(np.linalg.norm(np.cross(x,y)) for x,y in zip(a,b));assert noncommuting>1e-12
    save(tmp_path/'covariance.json',dict(base=packet(r),moved=packet(changed),errors=errors,noncommuting_increment_cross=float(noncommuting)))


@pytest.mark.parametrize('case',['curved-plastic','connected-plastic'])
def test_geometry_arc(case,tmp_path):
    m=make(case);p=ArcProgram((.2,.2),1.,pattern(m),SpatialNodalMoments(((3,.015,-.01,.008),)))
    result=solve_arc(m,p);save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    fresh=make(case);chain,records=decode_checkpoint(fresh,p,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert encode_checkpoint(fresh,p,chain,records)==result.checkpoint
    last=chain[-1];positive=sum(sum(s.accumulated)>0 for state in last['states'].values() for s in state['response'].history.stations);assert positive>0
    recovery={i:recover_native_fields(e,fresh.mesh,last['states'][i],expected_committed_total_u=last['displacements'][list(e.get_dof_mapping(fresh.mesh))]) for i,e in fresh.mesh.elements.items()}
    prefix=encode_checkpoint(fresh,p,chain[:2],records[:1]);again=make(case)
    resumed=solve_arc(again,p,checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    assert resumed.status=='completed' and resumed.checkpoint==result.checkpoint,resumed.failure
    save(tmp_path/'recovery.json',dict(fields=recovery,positive_plastic_stations=positive,byte_identical_resume=True,parameters=[row['parameter'] for row in records],arc_residuals=[row['arc_residual'] for row in records]))
