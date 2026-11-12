"""Owned-control factors, physical free space and fail-closed custody."""
from hashlib import sha256
import json
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_translation_modal as modal
from anyfesolver import _ge_beam3_retained_translation_control as control
from anyfesolver import _ge_beam3_retained_nodal_loading as nodal
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver.control import CancellationToken,SolveCancelled
from docs.reference_cases import ge_beam3_retained_arch_case as case


def inputs():
    m=case.model(2);p=control.Program((.003,),3,(0.,-1.,0.),case.program(2).nodal_forces)
    c=control.Context(m,p);raw=c.checkpoint(())
    masses={i:np.diag([1.,1.,1.,3e-5,1e-5,2e-5]) for i in m.mesh.elements}
    return m,p,raw,masses


def capture(m,p,raw,masses,**kw):return modal.prepare(m,p,raw,masses,expected_sha256=sha256(raw).hexdigest(),**kw)


def test_virgin_factors_equal_force_owned_and_control_stays_free(tmp_path):
    m,p,raw,masses=inputs();packet,check=capture(m,p,raw,masses)
    c=control.Context(m,p).physical;old_raw=c.checkpoint(())
    old,guard=nodal.prepare(m,c.program,old_raw,masses,expected_sha256=sha256(old_raw).hexdigest())
    for key in ('left','right','geometric','kinetic','mass','stiffness','net_residual'):
        np.testing.assert_array_equal(getattr(packet,key),getattr(old,key))
    assert packet.free_dofs==old.free_dofs and packet.algebraic_dofs==old.algebraic_dofs
    assert 13 in packet.free_dofs and packet.completed_targets==0 and packet.parameter==0.
    assert packet.displacement_target==0. and not packet.control_constraint_in_physical_stiffness
    assert not packet.production_qualified and packet.policy==modal.POLICY
    check();guard();(tmp_path/'virgin.json').write_bytes(canonical(packet))


def test_actual_controlled_state_full_schur_and_recovery(tmp_path):
    m,p,_,masses=inputs();result=control.solve(m,p)
    assert result.status=='completed' and result.completed_targets==1,result.failure
    raw=result.checkpoint;packet,check=capture(m,p,raw,masses)
    c=control.Context(m,p);state,records=c.restore(raw,expected_sha256=sha256(raw).hexdigest());phys=c.physical
    assert c.checkpoint(records)==raw and packet.parameter==state.parameter and packet.displacement_target==.003
    assert abs(c.value(state.mechanical)-.003)<1e-11 and 13 in packet.free_dofs
    _,j,metrics,_,_=c.assemble(state.mechanical,state.parameter,state.histories,.003)
    coordinates=list(range(phys.nodal_count));resultants=[]
    for i in range(len(phys.elements)):
        coordinates.extend(range(phys.nodal_count+24*i,phys.nodal_count+24*i+6))
        resultants.extend(range(phys.nodal_count+24*i+6,phys.nodal_count+24*(i+1)))
    expected=j[np.ix_(coordinates,coordinates)]-j[np.ix_(coordinates,resultants)]@np.linalg.solve(
        j[np.ix_(resultants,resultants)],j[np.ix_(resultants,coordinates)])
    free=packet.free_dofs
    error=np.linalg.norm((expected-packet.stiffness)[np.ix_(free,free)])/max(1.,np.linalg.norm(expected[np.ix_(free,free)]))
    assert error<=1e-11 and max(metrics)<=1e-11
    rotations=[6*i+k for i in range(5) for k in (3,4,5)]
    assert not np.any(packet.mass[:,rotations]) and not np.any(packet.mass[rotations,:])
    assert len(c.recover(state))==2
    check();(tmp_path/'controlled.json').write_bytes(canonical(dict(packet=packet,checkpoint_sha256=sha256(raw).hexdigest(),
        full_stationary_schur_error=float(error),production_qualified=False)))


@pytest.mark.parametrize('mutation',('hash','program','arc-checkpoint','force-checkpoint','missing-mass','indefinite-mass',
    'cancel','model','caller-mass','packet','control-row','control-target','control-claim','resealed-parameter'))
def test_fail_closed(mutation):
    m,p,raw,masses=inputs()
    if mutation=='hash':
        with pytest.raises(ValueError,match='authority'):modal.prepare(m,p,raw,masses,expected_sha256='0'*64)
    elif mutation=='program':
        with pytest.raises(ValueError,match='exact translation'):capture(m,case.program(2),raw,masses)
    elif mutation=='arc-checkpoint':
        c=case.arc.Context(m,case.program(2))
        with pytest.raises(ValueError):capture(m,p,c.checkpoint(()),masses)
    elif mutation=='force-checkpoint':
        c=control.Context(m,p).physical
        with pytest.raises(ValueError):capture(m,p,c.checkpoint(()),masses)
    elif mutation=='missing-mass':
        with pytest.raises(ValueError,match='inertia map'):capture(m,p,raw,{})
    elif mutation=='indefinite-mass':
        masses[1][0,0]=-1.
        with pytest.raises(np.linalg.LinAlgError):capture(m,p,raw,masses)
    elif mutation=='cancel':
        t=CancellationToken();t.cancel()
        with pytest.raises(SolveCancelled):capture(m,p,raw,masses,cancellation_token=t)
    elif mutation=='resealed-parameter':
        data=json.loads(raw);data['initial']['parameter']=.1
        body={k:v for k,v in data['initial'].items() if k!='record_sha256'};data['initial']['record_sha256']=sha(body)
        data['checkpoint_sha256']=sha({k:v for k,v in data.items() if k!='checkpoint_sha256'})
        with pytest.raises(ValueError):capture(m,p,canonical(data),masses)
    else:
        packet,check=capture(m,p,raw,masses)
        if mutation=='model':m.materials.clear()
        elif mutation=='caller-mass':masses[1][0,0]*=2
        elif mutation=='packet':object.__setattr__(packet,'parameter',.1)
        elif mutation=='control-row':object.__setattr__(p,'direction',(0.,1.,0.))
        elif mutation=='control-target':object.__setattr__(p,'targets',(.004,))
        else:object.__setattr__(packet,'control_constraint_in_physical_stiffness',True)
        with pytest.raises(ValueError):check()
