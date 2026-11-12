from copy import deepcopy
from hashlib import sha256
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_next_spatial_native as gate
from docs.reference_cases.ge_beam3_retained_prestress_protocol import canonical,strict_bytes


@pytest.mark.parametrize('sign',('plus','minus'))
def test_exact_endpoint_and_trial_binding(sign,tmp_path):
    seed,checkpoint,trial=gate.source(sign)
    assert sha256(checkpoint).hexdigest()==gate.CHECKPOINT[sign]
    assert strict_bytes(checkpoint)['program']['targets']==([.0065] if sign=='plus' else [-.0065])
    assert strict_bytes(trial)['numerical_negative_direction'] is True
    relative='wave/'+sign+'-a/output/checkpoint.json'
    (tmp_path/'manifest.json').write_bytes((gate.ENDPOINT/'manifest.json').read_bytes())
    destination=tmp_path/relative;destination.parent.mkdir(parents=True);destination.write_bytes(checkpoint)
    assert gate.member(tmp_path,gate.ENDPOINT_SHA,relative,gate.CHECKPOINT[sign])==checkpoint
    destination.write_bytes(checkpoint+b' ')
    with pytest.raises(ValueError):gate.member(tmp_path,gate.ENDPOINT_SHA,relative,gate.CHECKPOINT[sign])


def specimen():
    # Schema-only fixture; NOT mechanics evidence.
    z=lambda r,c:[[0.]*c for _ in range(r)]
    p=dict(left=z(432,432),right=z(432,438),geometric=z(438,438),kinetic=[],stiffness=z(438,438),mass=z(438,438),
        net_residual=[0.]*438,free_dofs=list(range(6,288))+list(range(294,438)),
        algebraic_dofs=[6*i+j for i in range(1,48) for j in (3,4,5)],
        internal_layout=[[i+1,list(range(294+6*i,300+6*i))] for i in range(24)],compliance_errors=[],
        seed_sha256=sha256(b'seed').hexdigest(),checkpoint_sha256=sha256(b'checkpoint').hexdigest(),
        model_sha256='0'*64,parameter=.0273,completed_targets=1,displacement_target=.0065,
        policy='schema-only',control_constraint_in_physical_stiffness=False,
        physical_loading_path_from_rest=False,production_qualified=False)
    return p
def seal(p):p['identity']=sha256(canonical(dict(policy=p['policy'],**{k:p[k] for k in gate.BODY}))).hexdigest()


@pytest.mark.parametrize('kind',('control','target','cursor','claim','seed','identity','internal','shape','float'))
def test_resealed_extent_policy_mutations(kind):
    p=specimen();seal(p);gate.validate(p,'plus',b'seed',b'checkpoint')
    if kind=='control':p['free_dofs'].remove(74)
    elif kind=='target':p['displacement_target']=.006
    elif kind=='cursor':p['completed_targets']=True
    elif kind=='claim':p['production_qualified']=True
    elif kind=='seed':p['seed_sha256']='0'*64
    elif kind=='internal':p['internal_layout'][0][1][0]=0
    elif kind=='shape':p['right'][0].pop()
    elif kind=='float':p['right'][0][0]=1
    else:p['identity']='0'*64
    if kind!='identity':seal(p)
    with pytest.raises(ValueError):gate.validate(p,'plus',b'seed',b'checkpoint')


# Actual small precise-policy owner test, separate from the large research wave.
from test_ge_beam3_elastic_seed_continuation import seed,programme
from test_ge_beam3_precise_work_enrollment import original,precise_model


def test_actual_precise_endpoint_capture_schur_and_custody(original):
    from anyfesolver import _ge_beam3_elastic_seed_continuation as owner
    from anyfesolver import _ge_beam3_elastic_seed_modal as modal
    from anyfesolver._ge_beam3_p5_seeded.core import canonical as native
    from docs.reference_cases.ge_beam3_precise_work_enrollment import enroll
    program=programme((.0003,));context,enrolled=enroll(precise_model(),program,original,sha256(original).hexdigest())
    result=owner.solve(precise_model(),program,enrolled,expected_seed_sha256=sha256(enrolled).hexdigest())
    assert result.status=='completed',result.failure
    state,records=context.restore(result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    before=native(state);made=precise_model();masses={i:np.diag([1.,1.,1.,3e-5,1e-5,2e-5]) for i in made.mesh.elements}
    packet,check=modal.prepare(made,program,enrolled,result.checkpoint,masses,
        expected_seed_sha256=sha256(enrolled).hexdigest(),expected_sha256=sha256(result.checkpoint).hexdigest())
    physical=context.physical
    _,j,metrics,_,_=context.assemble(state.mechanical,state.parameter,state.histories,.0003)
    coordinates=list(range(physical.nodal_count));resultants=[]
    for i in range(len(physical.elements)):
        coordinates.extend(range(physical.nodal_count+24*i,physical.nodal_count+24*i+6))
        resultants.extend(range(physical.nodal_count+24*i+6,physical.nodal_count+24*(i+1)))
    k=j[np.ix_(coordinates,coordinates)]-j[np.ix_(coordinates,resultants)]@np.linalg.solve(j[np.ix_(resultants,resultants)],j[np.ix_(resultants,coordinates)])
    assert np.linalg.norm(k-packet.stiffness)/max(1.,np.linalg.norm(k))<1e-11
    assert 12 in packet.free_dofs and packet.completed_targets==1
    assert not packet.control_constraint_in_physical_stiffness
    assert context.checkpoint(records)==result.checkpoint and native(state)==before
    check()
