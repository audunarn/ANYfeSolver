"""Whole-band component envelopes retain complete mapped gradient work."""
import numpy as np
import pytest
from anyfesolver.general_contact.native_curved_s3_work import (
    _component_root_leading_difference_bound,_root_component_gradient_bounds,
    NativeCurvedMaterialS3Work)
from test_general_contact_native_curved_s3_work import _case


def test_component_leading_difference_signed_products_and_positive_mass():
    rng=np.random.default_rng(318)
    for bounds in (np.array([3.,0.,0.]),np.array([1.,2.,.01]),np.ones(9)):
        factor=2.3
        for _ in range(20):
            # Distinct signed gradients and quadrature supports; each positive
            # measure has at most the common mass, including fallback mass.
            matrices=[]
            for __ in range(2):
                gradients=rng.uniform(-1.,1.,(7,len(bounds)))*bounds
                weights=rng.uniform(0.,1.,7);weights*=factor*rng.uniform()/weights.sum()
                matrices.append(np.einsum('q,qi,qj->ij',weights,gradients,gradients))
            assert np.linalg.norm(matrices[0]-matrices[1],np.inf)<=_component_root_leading_difference_bound(factor,bounds)


def test_whole_band_components_cancellation_extension_and_structural_zeros():
    base=np.array([-1.,0.,5.,0.]);circular=np.array([[1.,0.],[0.,0.],[-4.,2.],[0.,0.]])
    gradients=[base,base+circular[:,0],base+circular[:,1]]
    extension=.2;angles=(-.3,.3);supported=np.array([True,True,True,False])
    bounds=_root_component_gradient_bounds(gradients,angles,extension,supported)
    assert bounds[3]==0. and bounds[1]>=extension # station zero is not band zero
    for angle in np.linspace(*angles,51):
        value=base+circular@np.array([np.cos(angle),np.sin(angle)])
        for sign in (-1.,1.):
            value_with_axial_change=value+sign*extension*supported
            assert np.all(np.abs(value_with_axial_change)<=bounds)


def test_component_policy_identity_and_guard():
    work,p,low,U,u=_case()
    spectral=NativeCurvedMaterialS3Work(work.contact,work.shell,root_tangent_enclosure_policy='spectral')
    assert work.identity==spectral.identity
    component=NativeCurvedMaterialS3Work(work.contact,work.shell,root_tangent_enclosure_policy='componentwise')
    assert component.identity!=work.identity
    component.root_tangent_enclosure_policy='spectral'
    with pytest.raises(ValueError,match='changed'):component._configuration(p,low,U,u)
    component.root_tangent_enclosure_policy='unknown'
    with pytest.raises(ValueError,match='changed'):component._configuration(p,low,U,u)
    with pytest.raises(ValueError):NativeCurvedMaterialS3Work(work.contact,work.shell,root_tangent_enclosure_policy='sampled')


def test_component_policy_partial_front_independent_work(monkeypatch):
    import test_general_contact_native_curved_s3_work as checks
    work,p,low,U,u=_case()
    component=NativeCurvedMaterialS3Work(work.contact,work.shell,
        integration_budget_policy='certified_active',root_tangent_enclosure_policy='componentwise')
    monkeypatch.setattr(checks,'_case',lambda:(component,p,low,U,u))
    checks.test_curved_s3_axial_front_complete_target_work_and_independent_partition()


def test_component_owner_restart_identity_rejects_spectral_checkpoint():
    from hashlib import sha256
    from test_general_contact_native_curved_s3_static import _analysis
    common=dict(physical_fibres=True,plastic_fibres=True,integration_budget_policy='certified_active')
    spectral=_analysis(**common)
    component=_analysis(**common,root_tangent_enclosure_policy='componentwise')
    assert component.identity!=spectral.identity
    for writer,reader in ((spectral,component),(component,spectral)):
        raw=writer._checkpoint(())
        with pytest.raises(ValueError,match='definition/settings binding'):
            reader._restore(raw,sha256(raw).hexdigest())
    raw=component._checkpoint(())
    _,records=component._restore(raw,sha256(raw).hexdigest())
    assert component._checkpoint(records)==raw
