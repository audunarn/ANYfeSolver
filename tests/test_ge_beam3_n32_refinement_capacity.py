from types import SimpleNamespace
from contextvars import Context
import numpy as np
import pytest
from anyfesolver._ge_beam3_refinement_capacity import retained_limits,n32_refinement_capacity
from anyfesolver._ge_beam3_native_generalized_program import model_identity,retained_model_identity
from anyfesolver._ge_beam3_retained_generalized_state import _retained_size
from anyfesolver._ge_beam3_elastic_seed_modal import _factor_size
from anyfesolver._ge_beam3_native_generalized_restart import _model
from anyfesolver._ge_beam3_precise_geometric_work import POLICY
from docs.reference_cases.ge_beam3_n32_controlled_case import model
from docs.reference_cases.ge_beam3_n32_equilibrium import sample,guess_descriptor,write_native
from docs.reference_cases.ge_beam3_refined_controlled_case import model as old_model

def test_limits_scoped_restored_and_not_propagated_to_fresh_context():
    assert retained_limits()==(24,1024,512)
    with n32_refinement_capacity():
        assert retained_limits()==(32,1280,640)
        assert Context().run(retained_limits)==(24,1024,512)
        assert _retained_size(390,32)==1158 and _factor_size(390,32)==582
        with pytest.raises(ValueError): 
            with n32_refinement_capacity():pass
    assert retained_limits()==(24,1024,512)
    with pytest.raises(ValueError):_retained_size(390,32)
    with pytest.raises(ValueError):_factor_size(390,32)
def test_scope_restores_on_exception():
    with pytest.raises(RuntimeError):
        with n32_refinement_capacity():raise RuntimeError('cancel')
    assert retained_limits()==(24,1024,512)
@pytest.mark.parametrize('args',[(True,32),(390,True),(390,33),(516,32),(391,32),(512,32)])
def test_invalid_capacity_still_rejected(args):
    with n32_refinement_capacity():
        with pytest.raises(ValueError):_retained_size(*args)
def test_native_model_full_audit_and_exit_rejection():
    m=model(32,arithmetic_policy=POLICY)
    with pytest.raises(ValueError):retained_model_identity(m)
    with n32_refinement_capacity():
        elements,nodal,free,fixed,identity=_model(m,retained=True)
        assert len(elements)==32 and nodal==390 and len(free)==378 and len(fixed)==12
        assert identity==retained_model_identity(m)
        assert [m.mesh.nodes[n].x for n in (1,17,33,65)]==[-1.,-.5,0.,1.]
        assert all(e.operator.arithmetic_policy==POLICY for _,e in elements)
        with pytest.raises(ValueError):model_identity(m)
    with pytest.raises(ValueError):retained_model_identity(m)
def test_old_identity_and_caps_unchanged():
    m=old_model(4,arithmetic_policy=POLICY);before=retained_model_identity(m)
    with n32_refinement_capacity():
        assert retained_model_identity(m)==before and _retained_size(54,4)==150
    assert retained_model_identity(m)==before
    with pytest.raises(ValueError):model(24,arithmetic_policy=POLICY)
def test_one_sided_polynomial_sampling():
    poly=lambda u:np.arange(52,dtype=float)+100*u
    assert sample(poly,0.,1)[7]==120.
    assert sample(poly,0.,2)[7]==33.
    np.testing.assert_array_equal(sample(poly,-.75),np.arange(13)+50.)
    for args in ((poly,2.),(poly,0.,0),(poly,0.,True)):
        with pytest.raises(ValueError):sample(*args)

@pytest.mark.parametrize('sign',('plus','minus'))
def test_actual_bound_continuum_guess_native_shapes_frames_and_clamps(sign):
    from docs.reference_cases.ge_beam3_spatial_ritz_worker import source
    from anyfesolver._ge_beam3_retained_generalized_state import Context as Physical,Mechanical
    m=model(32,arithmetic_policy=POLICY);elements=tuple(sorted(m.mesh.elements.items()))
    positions=np.array([m.mesh.nodes[i].coords() for i in range(1,66)])
    frames=np.empty((65,3,3))
    for i,(_,e) in enumerate(elements):frames[2*i:2*i+3]=e.operator.reference.nodal_triads
    initial=Mechanical(positions.copy(),np.zeros_like(positions),frames.copy(),np.tile(np.eye(3),(32,2,1,1)),np.zeros((32,18)))
    p=SimpleNamespace(node_ids=tuple(range(1,66)),elements=elements,reference_positions=positions,reference_frames=frames,
        fixed=tuple(range(6))+tuple(range(384,390)),initial=SimpleNamespace(mechanical=initial))
    p.make=lambda data:Physical.make(p,data)
    value=source(sign);guess=guess_descriptor(p,value)
    np.testing.assert_array_equal(guess.positions[[0,64]],positions[[0,64]])
    np.testing.assert_array_equal(guess.nodal_frames[[0,64]],frames[[0,64]])
    assert abs(guess.positions[16,2]-(.0065 if sign=='plus' else -.0065))<1e-11
    assert guess.resultants.shape==(32,18) and guess.cell_rotations.shape==(32,2,3,3)
    assert np.isfinite(guess.resultants).all() and np.max(abs(guess.resultants))>0
    assert not initial.resultants.any() and not initial.position_low.any()

def test_native_array_history_writer_roundtrip_exclusive_and_no_partial(tmp_path):
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    from docs.reference_cases.ge_beam3_retained_prestress_protocol import strict_bytes
    m=old_model(2,arithmetic_policy=POLICY);history=m.mesh.elements[1].operator.cell.virgin()
    value=dict(mechanical=dict(cell_rotations=np.tile(np.eye(3),(2,2,1,1))),recovery=[dict(history=history,field=np.zeros(6))])
    path=tmp_path/'native.json';write_native(path,value)
    raw=path.read_bytes();assert raw==canonical(value) and strict_bytes(raw)['mechanical']['cell_rotations'][0][0][0][0]==1.
    with pytest.raises(FileExistsError):write_native(path,value)
    assert path.read_bytes()==raw
    invalid=tmp_path/'invalid.json'
    with pytest.raises(ValueError):write_native(invalid,dict(bad=np.array([float('nan')])))
    assert not invalid.exists()
