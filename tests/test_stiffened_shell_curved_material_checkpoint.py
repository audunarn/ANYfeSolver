"""Predecessor-aware curved point snapshots and truthful cached station recovery."""
from dataclasses import replace
import json
from math import fsum
import numpy as np
import pytest
from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_ge_checkpoint_binary import MAGIC,decode,encode
from anyfesolver._stiffened_shell_ge_finite_experimental import _position_pair
from test_stiffened_shell_curved_material import (
    curved_material_panel,_actions,make_owner,assert_generalized_records_equal,
)


def decoded(raw,model):
    # Mutation oracle starts from our trusted bounded fixture; production restore
    # independently derives its exact shapes from the compiled model.
    size=int.from_bytes(raw[len(MAGIC):len(MAGIC)+4],'little')
    header=json.loads(raw[len(MAGIC)+4:len(MAGIC)+4+size])
    return decode(raw,model,lambda kind,generation:tuple((n,tuple(v)) for n,v in header['arrays']))


@pytest.fixture(scope='module')
def accepted():
    d=curved_material_panel();owner=ss.create_analysis(definition=d)
    try:
        result=owner.solve(_actions(),(.5,1.),max_iterations=20,max_backtracks=12)
        yield owner,result,owner.checkpoint()
    finally:owner.close()


def test_curved_station_recovery_native_update_work_and_no_replay(accepted,monkeypatch):
    owner,result,_=accepted;driver=owner._finite_driver;state=driver._state
    xyz,low=_position_pair(driver._reference_positions,state.q[owner._shell_count:owner._shell_count+driver._beam_nodal].reshape(-1,6)[:,:3])
    dissipated=[]
    for i,(part,record) in enumerate(zip(owner._beam_parts,result.beam_generalized_material)):
        element,_,_=part;nodes=[owner._beam_index[n] for n in element.node_ids]
        raw=element.operator.evaluate(xyz[nodes],low[nodes],state.frames[nodes],
            state.cells[i],state.resultants[i],origin=driver._beam_predecessors[i],check=element.operator.guard)
        stations=json.loads(raw.material)['stations']
        for station,data in zip(record.stations,stations):
            np.testing.assert_array_equal(station.local_strain,data['strain'][0])
            np.testing.assert_array_equal(station.local_resultants,data['resultants'][0])
            assert station.branch==data['branch']
            assert station.dissipation_per_length==tuple(v[0] for v in data['dissipation'])
            assert station.plastic_increment==tuple(v[0] for v in data['plastic_increment'])
            dissipated.append(fsum(station.dissipation_per_length)*station.measure)
    assert max(dissipated)>0. and min(dissipated)>=0.
    def forbidden(*args,**kwargs):raise AssertionError('recovery must use cached accepted records')
    with monkeypatch.context() as patch:
        patch.setattr(driver,'_assemble',forbidden)
        for element,_,_ in owner._beam_parts:patch.setattr(type(element.operator),'evaluate',forbidden)
        current=owner.recover()
        assert_generalized_records_equal(current.beam_generalized_material,result.beam_generalized_material)
        current.beam_generalized_material[0].stations[0].local_resultants.setflags(write=True)
        current.beam_generalized_material[0].stations[0].local_resultants[:]=0.
        assert_generalized_records_equal(owner.recover().beam_generalized_material,result.beam_generalized_material)


@pytest.mark.parametrize('field',('beam_origin_plastic','beam_origin_accumulated','shell_origin_plastic','shell_origin_alpha'))
def test_altered_predecessors_with_valid_digest_reject_atomically(accepted,request,field):
    owner,_,raw=accepted;kind,generation,arrays=decoded(raw,owner._identity)
    assert kind=='finite_curved_points_coupled'
    arrays={k:v.copy() for k,v in arrays.items()};arrays[field].flat[0]+=.01
    changed=encode(kind,owner._identity,generation,tuple(arrays.items()))
    fresh=make_owner(request,owner._definition)
    with pytest.raises(ValueError,match='mechanically inconsistent|nonphysical'):
        fresh.restore(changed)
    assert fresh._generation==0 and fresh._finite_driver is None


def test_old_snapshot_omits_unknown_increment_then_new_acceptance_upgrades(accepted,request):
    owner,result,raw=accepted;_,generation,arrays=decoded(raw,owner._identity)
    old=encode('finite_shell_material',owner._identity,generation,
        tuple((n,v) for n,v in arrays.items() if '_origin_' not in n))
    legacy=make_owner(request,owner._definition);restored=legacy.restore(old)
    assert legacy.checkpoint()==old and restored.beam_generalized_material==()
    np.testing.assert_array_equal(restored.beam_station_accumulated,result.beam_station_accumulated)
    new=make_owner(request,owner._definition);new.restore(raw)
    left=legacy.solve(_actions(),(.9,),max_iterations=20,max_backtracks=12)
    right=new.solve(_actions(),(.9,),max_iterations=20,max_backtracks=12)
    assert legacy.checkpoint()==new.checkpoint()
    assert left.beam_generalized_material
    assert_generalized_records_equal(left.beam_generalized_material,right.beam_generalized_material)


def test_predecessor_checkpoint_requires_matching_policy_and_material(accepted,request):
    owner,_,raw=accepted;kind,generation,arrays=decoded(raw,owner._identity)
    plain=replace(owner._definition,shell=replace(owner._definition.shell,j2_curve=None),
        shell_reference_policy=ss.CURVED_Q4_POINT_REFERENCE_V1)
    elastic=make_owner(request,plain)
    changed=encode(kind,elastic._identity,generation,tuple(arrays.items()))
    with pytest.raises(ValueError,match='matching policy'):elastic.restore(changed)
    fresh=make_owner(request,owner._definition)
    changed=encode('finite_curved_points_material',owner._identity,generation,
        tuple((n,v) for n,v in arrays.items() if not n.startswith('shell_')))
    with pytest.raises(ValueError,match='matching policy'):fresh.restore(changed)
    assert elastic._finite_driver is None and fresh._finite_driver is None


def test_restored_fixed_predecessor_residual_and_tangent(accepted,request):
    from scipy.sparse.linalg import norm
    owner,_,raw=accepted;fresh=make_owner(request,owner._definition);fresh.restore(raw)
    a=owner._finite_driver;b=fresh._finite_driver
    original=a._assemble(a._state,a._load_factor,a._shell_predecessors,a._beam_predecessors)
    replay=b._assemble(b._state,b._load_factor,b._shell_predecessors,b._beam_predecessors)
    np.testing.assert_allclose(original[0],replay[0],rtol=1e-10,atol=1e-8)
    assert norm(original[1]-replay[1])/norm(original[1])<=1e-10


def test_elastic_shell_generalized_history_uses_separate_schema(request):
    d=curved_material_panel();d=replace(d,shell=replace(d.shell,j2_curve=None))
    owner=make_owner(request,d)
    result=owner.solve(_actions(),(.5,),max_iterations=20,max_backtracks=16)
    raw=owner.checkpoint();kind,_,arrays=decoded(raw,owner._identity)
    assert kind=='finite_curved_points_material' and 'shell_origin_alpha' not in arrays
    fresh=make_owner(request,d);restored=fresh.restore(raw)
    assert fresh.checkpoint()==raw and restored.beam_generalized_material
    assert_generalized_records_equal(result.beam_generalized_material,restored.beam_generalized_material)
