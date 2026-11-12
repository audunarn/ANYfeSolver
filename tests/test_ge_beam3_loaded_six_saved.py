"""Saved six-macro hash DAG, factor identities and mutation tests; no solves."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from docs.reference_cases import ge_beam3_loaded_arch_refinement as probe

OUTPUT=probe.EXTERNAL/'ge-beam3-loaded-six-20260907-v1'
REVISION='b66ce8d825708014c352449dd009af23a6f6b1f0'
DIGEST='93b6507bce49e3e1022fc7792f2c9018af89bf55681ed485cc328972ed8e79bc'


@pytest.fixture
def saved():
    if not OUTPUT.exists(): pytest.skip('external development evidence absent')
    raw=(OUTPUT/'comparison.json').read_bytes()
    assert len(raw)==3490 and sha256(raw).hexdigest()==DIGEST
    return raw,probe.validate(OUTPUT,raw,REVISION)


def test_complete_dag_and_exact_committed_copy(saved):
    raw,value=saved
    assert (probe.ROOT/'docs/reference_cases/ge_beam3_loaded_six_development_result.json').read_bytes()==raw
    assert value['new_native_nonlinear_solves']==value['new_reference_solves']==0
    assert len(value['artifacts'])==10 and value['new_native_spectra']==2


def test_refinement_and_signed_stability_not_qualification(saved):
    for value in saved[1]['rows']:
        a=np.asarray(value['eigenvalue_relative_errors']); b=np.asarray(value['coarse_eigenvalue_relative_errors'])
        assert np.all(a<b) and value['every_error_decreased'] is True
        assert value['native_negative_modes']==value['continuum_negative_modes']==(0 if value['cursor']==1 else 2)
        assert max(a)>.06 and value['same_equilibrium_branch_proved'] is False
        assert value['buckling_factor_authorized'] is False
    assert saved[1]['production_qualified'] is False and saved[1]['independent_review']=='PENDING'


def test_original_factor_ritz_identity_and_state_policy(saved):
    from anyfesolver._dyadic_factor_chain import reassemble_chain_exact_binary64
    for cursor in (1,4):
        value=probe.parse(probe.read(OUTPUT/f'native-{cursor}.json')); p=value['packet']; modes=value['modes']
        assert p['material_policy']==modes['material_policy']=='FROZEN_PLASTIC_COORDINATES_ELASTIC_PERTURBATION'
        assert p['identity']==modes['operator_identity']
        assert modes['negative_eigenvalues_retained'] is True and modes['certified_intervals'] is False
        assert modes['mass_policy']=='CURRENT_LIFTED_KINETIC_FIELD_LINEARIZED_AT_REST'
        left,right,g,b=(np.asarray(p[k]) for k in ('left','right','geometric','kinetic'))
        v=np.asarray(modes['full_modes']); roots=np.asarray(modes['eigenvalues'])
        assert v.shape==(114,6) and g.shape==(114,114)
        assert np.count_nonzero(b[:,p['algebraic']])==0
        h,m=reassemble_chain_exact_binary64(left,right,g,b,v)
        scale=np.maximum(1.,np.sqrt(abs(roots))[:,None]*np.sqrt(abs(roots))[None,:])
        assert np.max(abs(h-m*roots[None,:])/scale)<1e-11
        assert np.linalg.norm(m-np.eye(6))<1e-11


@pytest.mark.parametrize('mutation',['status','revision','qualification','budget','artifact','negative','eigenvalue','improvement'])
def test_summary_mutations_rejected(saved,mutation):
    value=deepcopy(saved[1])
    if mutation=='status': value['status']='PASS'
    elif mutation=='revision': value['revision']='0'*40
    elif mutation=='qualification': value['production_qualified']=True
    elif mutation=='budget': value['coordinate_limit']=256
    elif mutation=='artifact': value['artifacts'][0]['sha256']='0'*64
    elif mutation=='negative': value['rows'][1]['native_negative_modes']=0
    elif mutation=='improvement': value['rows'][0]['every_error_decreased']=False
    else: value['rows'][0]['native_eigenvalues'][0]+=1.
    with pytest.raises(ValueError): probe.validate(OUTPUT,probe.canonical(value),REVISION)


def test_native_packet_mutation_rejected(saved,monkeypatch):
    original=probe.read
    def read(path):
        raw=original(path)
        if path.name=='native-4.json':
            value=probe.parse(raw); value['packet']['geometric'][0][0]+=1.; return probe.canonical(value)
        return raw
    monkeypatch.setattr(probe,'read',read)
    with pytest.raises(ValueError,match='hash DAG'): probe.validate(OUTPUT,saved[0],REVISION)
