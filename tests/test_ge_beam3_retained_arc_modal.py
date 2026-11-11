"""Arc-owned spectral authority and exact zero-family guards."""
from hashlib import sha256
from types import SimpleNamespace
import numpy as np
import pytest
from anyfesolver import _ge_beam3_retained_arc_modal as modal
from anyfesolver import _ge_beam3_retained_nodal_loading as nodal
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from docs.reference_cases import ge_beam3_retained_arch_case as case
from docs.reference_cases.ge_beam3_retained_arc_spectrum import planar_partition


def inputs():
    m=case.model(2);p=case.program(2);c=case.arc.Context(m,p);raw=c.checkpoint(())
    masses={i:np.diag([1.,1.,1.,3e-5,1e-5,2e-5]) for i in m.mesh.elements}
    return m,p,raw,masses


def capture(m,p,raw,masses,**kw):
    return modal.prepare(m,p,raw,masses,expected_sha256=sha256(raw).hexdigest(),**kw)


def test_virgin_force_and_arc_factors_identical():
    m,p,raw,masses=inputs();packet,guard=capture(m,p,raw,masses)
    c=case.arc.Context(m,p).physical;old_raw=c.checkpoint(())
    old,check=nodal.prepare(m,c.program,old_raw,masses,expected_sha256=sha256(old_raw).hexdigest())
    for key in ('left','right','geometric','kinetic','mass','stiffness','net_residual'):
        np.testing.assert_array_equal(getattr(packet,key),getattr(old,key))
    assert packet.free_dofs==old.free_dofs and packet.algebraic_dofs==old.algebraic_dofs
    assert packet.completed_steps==0 and packet.parameter==0. and packet.policy==modal.POLICY
    assert not packet.production_qualified and not packet.arc_constraint_in_physical_stiffness
    assert packet.checkpoint_sha256==sha256(raw).hexdigest()
    guard();check()


@pytest.mark.parametrize('mutation',('hash','program','missing-mass','indefinite-mass','cancel','force-checkpoint',
                                   'model','caller-mass','packet','arc-row'))
def test_authority_fail_closed(mutation):
    m,p,raw,masses=inputs()
    if mutation=='hash':
        with pytest.raises(ValueError,match='authority'):modal.prepare(m,p,raw,masses,expected_sha256='0'*64)
    elif mutation=='program':
        with pytest.raises(ValueError,match='exact arc'):capture(m,case.arc.Context(m,p).physical.program,raw,masses)
    elif mutation=='missing-mass':
        with pytest.raises(ValueError,match='inertia map'):capture(m,p,raw,{})
    elif mutation=='indefinite-mass':
        masses[1][0,0]=-1.
        with pytest.raises(np.linalg.LinAlgError):capture(m,p,raw,masses)
    elif mutation=='cancel':
        token=CancellationToken();token.cancel()
        with pytest.raises(SolveCancelled):capture(m,p,raw,masses,cancellation_token=token)
    elif mutation=='force-checkpoint':
        c=case.arc.Context(m,p).physical
        with pytest.raises(ValueError):capture(m,p,c.checkpoint(()),masses)
    else:
        packet,guard=capture(m,p,raw,masses)
        if mutation=='model':m.materials.clear()
        elif mutation=='caller-mass':masses[1][0,0]*=2
        elif mutation=='packet':object.__setattr__(packet,'parameter',.1)
        else:object.__setattr__(p,'parameter_scale',2.)
        with pytest.raises(ValueError):guard()


@pytest.mark.parametrize('mutation',('material','kinetic','geometric'))
def test_cross_family_coupling_is_not_thresholded(mutation):
    # Synthetic uncoupled diagnostic; tiny nonzeros must reject partition.
    p=SimpleNamespace(left=np.eye(12),right=np.eye(12),geometric=np.eye(12),kinetic=np.eye(12),
                      free_dofs=tuple(range(12)),algebraic_dofs=())
    planar_partition(p,12)
    if mutation=='material':p.left[0,2]=1e-30
    elif mutation=='kinetic':p.kinetic[0,2]=1e-30
    else:p.geometric[0,2]=p.geometric[2,0]=1e-30
    with pytest.raises(ValueError,match='couple'):planar_partition(p,12)
