"""Actual small private-Q4 lifecycle controls; no large memory prediction."""
import gc
import weakref
import numpy as np
import pytest
from anyfesolver import stiffened_shell as api
from anyfesolver.control import CancellationToken
import anyfesolver._stiffened_shell_b2_q4_panel_experimental as product
from anyfesolver.e4_pl_element import _retire_q4_private_mesh_caches
from anyfesolver._stiffened_shell_b2_assembly_diagnostics import Capture,DiagnosticRefusal,capture_assembly,active_capture
from examples.stiffened_shell_b2_physical_rigid_footprint import definition,preload


def refs(owner):
    return [weakref.ref(owner._buckling_model.mesh)]+[weakref.ref(x[1]) for x in owner._buckling_shells]


def collected(references):
    gc.collect()
    assert references and all(x() is None for x in references)


@pytest.mark.parametrize('captured',(False,True))
def test_close_collects_guard_and_snapshot_roots(captured,monkeypatch):
    if captured:
        original=Capture.prefactor
        def prefactor(c,persistent,*args):
            assert type(persistent['private_q4_elements']) is list and len(persistent['private_q4_elements'])==2
            return original(c,persistent,*args)
        monkeypatch.setattr(Capture,'prefactor',prefactor)
    for unused in range(3):
        if captured:
            with capture_assembly() as c:owner=api.create_analysis(definition=definition())
            assert c.status=='COMPLETE'
        else:owner=api.create_analysis(definition=definition())
        references=refs(owner)
        owner.close();owner.close()
        assert owner._private_q4_mesh is None and owner._private_q4_elements==[]
        assert owner._buckling_model is None and owner._buckling_shells==()
        del owner
        collected(references)


def test_other_open_owner_and_recovery_are_unchanged():
    a=api.create_analysis(definition=definition());b=api.create_analysis(definition=definition())
    try:
        references=refs(a);before=b.solve_linear(preload());k=b._K.copy()
        a.close();del a;collected(references)
        mesh=b._buckling_model.mesh
        for unused,element,material_id,slots in b._buckling_shells:
            element.compute_stiffness_matrix(mesh,b._buckling_model.get_material(material_id))
        after=b.solve_linear(preload())
        assert before.status==after.status=='completed' and before.displacements==after.displacements
        assert (b._K!=k).nnz==0 and b._handle.factorization_count==1
    finally:b.close()


def test_wrong_mesh_refuses_without_evicting_live_external_record():
    a=api.create_analysis(definition=definition());b=api.create_analysis(definition=definition())
    try:
        element=b._buckling_shells[0][1];mesh=b._buckling_model.mesh
        guard=element._qualified_component_guard
        with pytest.raises(ValueError,match='does not own'):
            _retire_q4_private_mesh_caches(element,a._buckling_model.mesh)
        assert element._qualified_component_guard is guard
        element.compute_stiffness_matrix(mesh,b._buckling_model.get_material(b._buckling_shells[0][2]))
        a.close()
        assert b.solve_linear(preload()).status=='completed'
    finally:a.close();b.close()


def test_tamper_refusal_and_retirement_use_closure_owned_guard():
    owner=api.create_analysis(definition=definition());element=owner._buckling_shells[0][1]
    mesh=owner._buckling_model.mesh;material=owner._buckling_model.get_material(owner._buckling_shells[0][2])
    references=refs(owner);tampered=(object(),)+element._qualified_component_guard[1:]
    with pytest.raises(AttributeError,match='internally managed'):element._qualified_component_guard=tampered
    object.__setattr__(element,'_qualified_component_guard',tampered)
    with pytest.raises(RuntimeError):element.compute_stiffness_matrix(mesh,material)
    owner.close();del owner,element,mesh,material
    collected(references)


def test_live_input_mutation_rebinds_before_retirement():
    owner=api.create_analysis(definition=definition())
    try:
        element=owner._buckling_shells[0][1];mesh=owner._buckling_model.mesh
        material=owner._buckling_model.get_material(owner._buckling_shells[0][2])
        before=element.compute_stiffness_matrix(mesh,material).copy();element.thickness*=1.1
        after=element.compute_stiffness_matrix(mesh,material)
        assert not np.array_equal(before,after)
    finally:owner.close()


@pytest.mark.parametrize('stage',('first_shell','later_shell','scatter','prefactor','factor','finish','cancel'))
def test_constructor_unwind_collects_private_mesh(monkeypatch,stage):
    references=[];models=product.FEModel;factory=product.create_shell_element
    def model(*a,**k):
        m=models(*a,**k);references.append(weakref.ref(m.mesh));return m
    def create(*a,**k):
        e=factory(*a,**k);references.append(weakref.ref(e));return e
    monkeypatch.setattr(product,'FEModel',model);monkeypatch.setattr(product,'create_shell_element',create)
    def fail(*a,**k):raise (KeyboardInterrupt if stage=='cancel' else RuntimeError)('original '+stage)
    if stage in ('first_shell','later_shell','scatter'):
        original=product._scatter;calls=[]
        def scatter(*a,**k):
            calls.append(1)
            if len(calls)==(2 if stage=='later_shell' else 1):fail()
            return original(*a,**k)
        monkeypatch.setattr(product,'_scatter',scatter)
    elif stage=='prefactor':monkeypatch.setattr(Capture,'prefactor',fail)
    elif stage in ('factor','cancel'):monkeypatch.setattr(product,'factorize_cached',fail)
    else:monkeypatch.setattr(Capture,'finish',fail)
    with pytest.raises(KeyboardInterrupt if stage=='cancel' else RuntimeError,match='original '+stage):
        with capture_assembly() as c:api.create_analysis(definition=definition())
    assert c.status=='FAILED' and active_capture() is None
    collected(references)


def test_unwind_retirement_failure_keeps_original_exception(monkeypatch):
    def fail(*a,**k):raise KeyboardInterrupt('original cancellation')
    def cleanup(*a,**k):raise RuntimeError('secondary retirement')
    monkeypatch.setattr(product,'factorize_cached',fail)
    # Exercise the actual wrapper with a secondary cleanup failure. Restore and
    # retire the captured private owner after checking the original exception.
    owners=[];retire=product._retire_private_q4_mesh
    def capture(owner):owners.append(owner);cleanup()
    monkeypatch.setattr(product,'_retire_private_q4_mesh',capture)
    with pytest.raises(KeyboardInterrupt,match='original cancellation') as error:
        api.create_analysis(definition=definition())
    assert any('secondary retirement' in x for x in error.value.__notes__)
    for owner in owners:retire(owner)


def test_close_failure_still_drops_factor_handles(monkeypatch):
    owner=api.create_analysis(definition=definition());retire=product._retire_private_q4_mesh
    def fail(*a):raise RuntimeError('retirement failed')
    monkeypatch.setattr(product,'_retire_private_q4_mesh',fail)
    with pytest.raises(RuntimeError,match='retirement failed'):owner.close()
    assert owner._closed and owner._handle is None and owner._attachment_elimination is None
    assert owner._physical_scaled_kkt is None and owner._buckling_model is None
    monkeypatch.setattr(product,'_retire_private_q4_mesh',retire);owner.close()


def test_cancelled_actual_batch_remains_live_until_close():
    owner=api.create_analysis(definition=definition());references=refs(owner)
    token=CancellationToken();token.cancel('before first RHS')
    result=owner.solve_linear_many((('case-1',preload()),),selected_dofs=(12,),cancellation_token=token)
    assert result.status=='cancelled' and result.cases==() and owner._handle.solve_count==0
    assert all(x() is not None for x in references)
    owner.close();del owner;collected(references)


@pytest.mark.parametrize('unwind',(False,True))
def test_actual_mixed_q4_s3_inherited_lifetime(monkeypatch,unwind):
    from examples.stiffened_shell_b2_mixed_panel import definition as mixed_definition
    references=[];models=product.FEModel;factory=product.create_shell_element
    def model(*a,**k):
        m=models(*a,**k);references.append(weakref.ref(m.mesh));return m
    def create(*a,**k):
        e=factory(*a,**k);references.append(weakref.ref(e));return e
    monkeypatch.setattr(product,'FEModel',model);monkeypatch.setattr(product,'create_shell_element',create)
    if unwind:
        def fail(*a,**k):raise RuntimeError('mixed factor refusal')
        monkeypatch.setattr(product,'factorize_cached',fail)
        with pytest.raises(RuntimeError,match='mixed factor refusal'):
            api.create_analysis(definition=mixed_definition())
    else:
        owner=api.create_analysis(definition=mixed_definition())
        assert len(owner._private_q4_elements)==1 and len(owner._buckling_shells)==3
        load=np.zeros(54);load[49],load[50],load[53]=.01,-.005,.001
        result=owner.solve_linear_many((('mixed',load),),selected_dofs=(49,50,53),recover_members={})
        assert result.status=='completed' and len(result.cases)==1 and result.factorization_count==1
        owner.close();del owner
    collected(references)
