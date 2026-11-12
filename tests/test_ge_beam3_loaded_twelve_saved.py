"""Read-only twelve-macro factor, refinement and evidence checks."""
from copy import deepcopy
from hashlib import sha256
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_loaded_arch_refinement as probe

OUTPUT=probe.EXTERNAL/'ge-beam3-loaded-twelve-20260907-v1'
REVISION='0ec814beac4369c16bc70368fcdbd90bf3b6e951'
DIGEST='cf8ef1e5309fdcec5a203aa6451f6ebca59ae9541f5a11b697b5073d161501b1'


@pytest.fixture
def saved():
    if not OUTPUT.exists(): pytest.skip('external twelve-macro evidence absent')
    raw=probe.read(OUTPUT/'comparison.json')
    assert len(raw)==3538 and sha256(raw).hexdigest()==DIGEST
    return raw,probe.validate(OUTPUT,raw,REVISION,12)


def test_complete_dag_and_committed_copy(saved):
    raw,value=saved
    assert (probe.ROOT/'docs/reference_cases/ge_beam3_loaded_twelve_development_result.json').read_bytes()==raw
    assert value['new_native_nonlinear_solves']==value['new_reference_solves']==0
    assert len(value['artifacts'])==10 and value['new_native_spectra']==2
    assert probe.read(OUTPUT/'checkpoint-4.json')==probe.read(probe.configuration(12)['input'])


def test_observed_refinement_and_signs_not_full_qualification(saved):
    for row in saved[1]['rows']:
        a=np.asarray(row['eigenvalue_relative_errors']); b=np.asarray(row['coarse_eigenvalue_relative_errors'])
        assert np.all(a<b) and row['every_error_decreased'] is True
        assert max(a)<.02  # recorded observation on these two states, not a qualification terminal
        assert row['native_negative_modes']==row['continuum_negative_modes']==(0 if row['cursor']==1 else 2)
        assert row['same_equilibrium_branch_proved'] is False and row['buckling_factor_authorized'] is False
    assert saved[1]['production_qualified'] is False and saved[1]['independent_review']=='PENDING'


def test_saved_original_factor_ritz_and_zero_inertia_traces(saved):
    from anyfesolver._dyadic_factor_chain import reassemble_chain_exact_binary64
    for cursor in (1,4):
        value=probe.parse(probe.read(OUTPUT/f'native-{cursor}.json')); p=value['packet']; modes=value['modes']
        assert p['identity']==modes['operator_identity']
        assert p['material_policy']==modes['material_policy']=='FROZEN_PLASTIC_COORDINATES_ELASTIC_PERTURBATION'
        assert modes['negative_eigenvalues_retained'] is True and modes['certified_intervals'] is False
        assert modes['state_advanced'] is False and modes['checkpoint_converted'] is False
        left,right,g,b=(np.asarray(p[k]) for k in ('left','right','geometric','kinetic'))
        v=np.asarray(modes['full_modes']); roots=np.asarray(modes['eigenvalues'])
        assert v.shape==(222,6) and g.shape==(222,222)
        assert len(p['free'])-len(p['algebraic'])==141
        assert np.count_nonzero(b[:,p['algebraic']])==0
        h,m=reassemble_chain_exact_binary64(left,right,g,b,v)
        scale=np.maximum(1.,np.sqrt(abs(roots))[:,None]*np.sqrt(abs(roots))[None,:])
        assert np.max(abs(h-m*roots[None,:])/scale)<1e-11
        assert np.linalg.norm(m-np.eye(6))<1e-11


@pytest.mark.parametrize('mutation',['status','revision','qualification','budget','replay_budget','artifact','negative','eigenvalue','improvement'])
def test_scope_and_summary_mutations_reject(saved,mutation):
    value=deepcopy(saved[1])
    if mutation=='status': value['status']='PASS'
    elif mutation=='revision': value['revision']='0'*40
    elif mutation=='qualification': value['production_qualified']=True
    elif mutation=='budget': value['coordinate_limit']=512
    elif mutation=='replay_budget': value['retained_coordinate_limit']=256
    elif mutation=='artifact': value['artifacts'][0]['sha256']='0'*64
    elif mutation=='negative': value['rows'][1]['native_negative_modes']=0
    elif mutation=='improvement': value['rows'][0]['every_error_decreased']=False
    else: value['rows'][0]['native_eigenvalues'][0]+=1.
    with pytest.raises(ValueError): probe.validate(OUTPUT,probe.canonical(value),REVISION,12)


def test_changed_native_factor_packet_rejects(saved,monkeypatch):
    original=probe.read
    def read(path):
        raw=original(path)
        if path.name=='native-4.json':
            value=probe.parse(raw); value['packet']['geometric'][0][0]+=1.; return probe.canonical(value)
        return raw
    monkeypatch.setattr(probe,'read',read)
    with pytest.raises(ValueError,match='hash DAG'): probe.validate(OUTPUT,saved[0],REVISION,12)
