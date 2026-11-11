"""Full input checking without repeated sparse support construction."""
from dataclasses import replace
import pytest
from anyfesolver import _ge_beam3_retained_generalized_state as retained
from anyfesolver import _ge_beam3_native_generalized_restart as restart
from anyfesolver._ge_beam3_native_generalized_loading import DistributedPattern
from anyfesolver._ge_beam3_native_line_loading import LinePattern
from test_ge_beam3_native_generalized_modal import make

EMPTY=DistributedPattern(LinePattern(()),())


def test_support_transform_built_once(monkeypatch):
    calls=[];original=restart.build_constraint_transformation
    def counted(*args):
        calls.append(1);return original(*args)
    monkeypatch.setattr(restart,'build_constraint_transformation',counted)
    m,_,_=make(False,2,clamped=True)
    c=retained.Context(m,retained.Program((1.,),EMPTY))
    for _ in range(100):c.guard()
    assert len(calls)==1
    before=retained.canonical(c.initial)
    c.recover(c.initial)
    assert retained.canonical(c.initial)==before and len(calls)==1


@pytest.mark.parametrize('mutation',('support-value','support-node','support-map','support-remove',
    'dof-order','dof-count','dof-count-type','node','material-owner','constraint','element-mpc','program','deadline'))
def test_full_guard_mutations(mutation,monkeypatch):
    m,_,_=make(False,1,clamped=True)
    c=retained.Context(m,retained.Program((1.,),EMPTY))
    bc=m.boundary_conditions[0];dm=m.mesh.dof_manager
    if mutation=='support-value':bc.dof_constraints['ux']=.1
    elif mutation=='support-node':bc.node_ids[:]=[3]
    elif mutation=='support-map':bc._dof_indices['ux']=1
    elif mutation=='support-remove':m.boundary_conditions.clear()
    elif mutation=='dof-order':dm._node_to_dof[3][0],dm._node_to_dof[3][1]=dm._node_to_dof[3][1],dm._node_to_dof[3][0]
    elif mutation=='dof-count':dm._total_dofs+=6
    elif mutation=='dof-count-type':
        dm._total_dofs=float(dm._total_dofs)
        with pytest.raises((ValueError,TypeError)):restart._model(m)
    elif mutation=='node':m.mesh.nodes[3].x+=.1
    elif mutation=='material-owner':m.materials.clear()
    elif mutation=='constraint':m.constraint_equations.append(object())
    elif mutation=='element-mpc':
        # Not represented in element.to_dict: the complete support audit must
        # still notice a changed element-provided equation after capture.
        monkeypatch.setattr(m.mesh.elements[1],'get_mpc_constraints',lambda mesh:[
            dict(slave=12,masters={6:1.},value=0.,label='injected')],raising=False)
    elif mutation=='program':c.program=replace(c.program,targets=(.5,))
    else:
        from anyfesolver._ge_beam3_effective_deadline import effective_deadline_seconds
        c.started-=effective_deadline_seconds(120)+1
    with pytest.raises((ValueError,RuntimeError)):c.guard()


def test_constraint_cache_not_solver_authority():
    # The generic assembler obtains its supports from the constraint audit,
    # not DOFManager's legacy bookkeeping set. Prove the same policy here.
    m,_,_=make(False,1,clamped=True)
    c=retained.Context(m,retained.Program((1.,),EMPTY))
    prior=restart._model(m)
    m.mesh.dof_manager._constrained_dofs.add(12)
    assert restart._model(m)[2:]==prior[2:]
    c.guard()
