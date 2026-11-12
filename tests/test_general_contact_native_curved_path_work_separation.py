"""Curved path checks own geometry; equilibrium owns endpoint integration."""
import numpy as np
import pytest
from anyfesolver.control import CancellationToken,SolveCancelled
from test_general_contact_native_curved_s3_static import _analysis

def test_curved_zero_path_uses_complete_geometry_control_not_force_quadrature(monkeypatch):
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    start=owner.initial.mechanical;u=owner.initial.shell_displacement
    step=np.zeros(owner.context.count+len(u));work=owner.pairs[0].work;nodes=owner.context.nodes[0]
    pose=work._configuration(start.positions[nodes],start.position_low[nodes],start.cell_rotations[0],u)
    expected=work.certify_path(pose,np.zeros((3,3)),np.zeros((2,3)),np.zeros_like(u),
        gap_tolerance_m=owner.pairs[0].gap_tolerance_m,maximum_intervals=owner.pairs[0].maximum_swept_intervals)
    assert expected.certified
    def forbidden(*args,**kwargs):raise AssertionError('path certification must not integrate endpoint forces')
    monkeypatch.setattr(owner,'_contact',forbidden)
    monkeypatch.setattr(work,'evaluate',forbidden)
    assert owner._swept_admissible(start,u,step,start,u)==expected.certified

def test_curved_path_retains_gap_rejection_and_cancellation():
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    start=owner.initial.mechanical;u=owner.initial.shell_displacement
    step=np.zeros(owner.context.count+len(u))
    # A translated source penetrates the physical target by far more than30um.
    step[8]=step[14]=-.001
    end=owner.context.advance(start,step[:owner.context.count])
    assert not owner._swept_admissible(start,u,step,end,u)
    token=CancellationToken();token.cancel()
    with pytest.raises(SolveCancelled):
        owner._swept_cancellable(start,u,np.zeros_like(step),start,u,cancellation_token=token)

def test_equilibrium_still_requires_endpoint_integration_and_path_guards(monkeypatch):
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    start=owner.initial.mechanical;u=owner.initial.shell_displacement;work=owner.pairs[0].work
    def reject(*args,**kwargs):raise MemoryError('owned endpoint integration failure')
    monkeypatch.setattr(work,'evaluate',reject)
    with pytest.raises(MemoryError,match='owned endpoint integration failure'):
        owner._assemble(start,u,1.,owner.initial.native_histories)
    work.contact.rim_clearance*=2
    with pytest.raises(ValueError,match='changed'):
        owner._swept_admissible(start,u,np.zeros(owner.context.count+len(u)),start,u)


def test_authoritative_caller_endpoint_matches_certified_path(monkeypatch):
    from anyfesolver.general_contact.native_surface_path import material_path_pose
    owner=_analysis(physical_fibres=True,plastic_fibres=True)
    start=owner.initial.mechanical;u=owner.initial.shell_displacement;work=owner.pairs[0].work
    step=np.zeros(owner.context.count+len(u));step[8]=1e-6;step[14]=2e-6
    step[owner.context.nodal_count+3:owner.context.nodal_count+6]=(1e-5,-2e-5,3e-5)
    step[owner.context.count+owner.shell_free[0]]=1e-6
    end=owner.context.advance(start,step[:owner.context.count]);last=u+step[owner.context.count:]
    real=work.certify_path;calls=[]
    def captured(pose,translations,angular,shell_increment,**kwargs):
        represented=material_path_pose(pose,translations,angular,shell_increment,1.)
        nodes=owner.context.nodes[0]
        np.testing.assert_allclose(represented.positions+represented.position_low,
            end.positions[nodes]+end.position_low[nodes],rtol=0.,atol=1e-14)
        np.testing.assert_allclose(represented.cell_rotations,end.cell_rotations[0],rtol=0.,atol=1e-13)
        np.testing.assert_array_equal(represented.shell_displacement,last)
        calls.append(1)
        return real(pose,translations,angular,shell_increment,**kwargs)
    monkeypatch.setattr(work,'certify_path',captured)
    assert owner._swept_admissible(start,u,step,end,last)
    assert calls==[1]
