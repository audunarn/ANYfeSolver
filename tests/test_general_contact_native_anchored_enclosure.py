"""Anchored root gradients preserve mass difference and signed variation."""
import numpy as np
import pytest
from anyfesolver.general_contact.native_curved_s3_work import (
    _anchored_root_leading_difference_bound,_root_component_gradient_bounds,
    NativeCurvedMaterialS3Work)
from test_general_contact_native_curved_s3_work import _case


def test_constant_gradient_unequal_masses_and_opposite_variation():
    v=np.array([-2.,.3,1.]);F=3.
    exact=F*np.outer(v,v)
    assert np.linalg.norm(exact,np.inf)<=_anchored_root_leading_difference_bound(F,v,np.zeros(3))
    e=np.array([.2,-.1,.4])
    difference=F*(np.outer(v+e,v+e)-np.outer(v-e,v-e))
    assert np.linalg.norm(difference,np.inf)<=_anchored_root_leading_difference_bound(F,v,np.abs(e))


def test_positive_mass_signed_variations_and_fallback_saturation():
    rng=np.random.default_rng(178)
    for _ in range(50):
        v=rng.normal(size=6);variation=rng.uniform(0.,.3,6);F=2.7
        matrices=[]
        for mass in (F,F*rng.uniform()):
            g=v+rng.uniform(-1.,1.,(8,6))*variation
            weights=rng.uniform(0.,1.,8);weights*=mass/weights.sum()
            matrices.append(np.einsum('q,qi,qj->ij',weights,g,g))
        assert np.linalg.norm(matrices[0]-matrices[1],np.inf)<=_anchored_root_leading_difference_bound(F,v,variation)


def test_direct_anchor_difference_with_axial_extension_and_cancellation():
    base=np.array([-1.,0.,5.,0.]);B=np.array([[1.,0.],[0.,0.],[-4.,2.],[0.,0.]])
    gradients=[base,base+B[:,0],base+B[:,1]];anchor=base+B@np.array([1.,0.])
    supported=np.array([True,True,True,False]);extension=.2
    bounds=_root_component_gradient_bounds(gradients,(-.3,.3),extension,supported,anchor=anchor)
    assert bounds[3]==0. and bounds[1]>=extension
    for alpha in np.linspace(-.3,.3,51):
        for sign in (-1.,1.):
            difference=base+B@np.array([np.cos(alpha),np.sin(alpha)])-anchor+sign*extension*supported
            assert np.all(np.abs(difference)<=bounds)


def test_anchor_identity_mutation_and_partial_work(monkeypatch):
    import test_general_contact_native_curved_s3_work as checks
    work,p,low,U,u=_case()
    old=NativeCurvedMaterialS3Work(work.contact,work.shell,root_tangent_enclosure_policy='componentwise')
    new=NativeCurvedMaterialS3Work(work.contact,work.shell,integration_budget_policy='certified_active',
        root_tangent_enclosure_policy='anchored')
    assert new.identity!=work.identity and new.identity!=old.identity
    monkeypatch.setattr(checks,'_case',lambda:(new,p,low,U,u))
    checks.test_curved_s3_axial_front_complete_target_work_and_independent_partition()
    new.root_tangent_enclosure_policy='unknown'
    with pytest.raises(ValueError,match='changed'):new._configuration(p,low,U,u)


def test_anchored_owner_restart_identity_and_empty_roundtrip():
    from hashlib import sha256
    from test_general_contact_native_curved_s3_static import _analysis
    common=dict(physical_fibres=True,plastic_fibres=True,
        integration_budget_policy='certified_active')
    anchored=_analysis(**common,root_tangent_enclosure_policy='anchored')
    for policy in ('spectral','componentwise'):
        previous=_analysis(**common,root_tangent_enclosure_policy=policy)
        assert previous.identity!=anchored.identity
        for writer,reader in ((previous,anchored),(anchored,previous)):
            raw=writer._checkpoint(())
            with pytest.raises(ValueError,match='definition/settings binding'):
                reader._restore(raw,sha256(raw).hexdigest())
    raw=anchored._checkpoint(())
    _,records=anchored._restore(raw,sha256(raw).hexdigest())
    assert anchored._checkpoint(records)==raw


@pytest.mark.parametrize('name,completed,digest',(
    ('active',1,'98d22bd2aff840df5d7b1034298891b179ed438ec9246a9f76b89445a7a15347'),
    ('open',2,'0ccdfa3cf637d12ca36051e315bd092cc61c53ac9b084676f8f99fac7e092d6e'),
    ('recontact',3,'5620d6a34a9e6f593a46de11e9c7b908804e2a6c3d7101b451719fde1321a44a')))
def test_anchored_accepted_fixture_exact_replay(name,completed,digest):
    from pathlib import Path
    from hashlib import sha256
    from test_general_contact_native_curved_s3_static import _analysis
    raw=(Path(__file__).parent/f'data/general_contact_native_fibre_s3_plastic_k8e7_anchored_{name}_v1.checkpoint').read_bytes()
    assert sha256(raw).hexdigest()==digest
    owner=_analysis(physical_fibres=True,plastic_fibres=True,normal_stiffness_Pa_per_m=8e7,
        integration_budget_policy='certified_active',root_tangent_enclosure_policy='anchored')
    state,records=owner._restore(raw,digest)
    assert state.completed_targets==completed
    assert owner._checkpoint(records)==raw
