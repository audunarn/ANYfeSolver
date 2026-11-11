"""Full retained/native + deformable S3 material contribution, not acceptance."""
from dataclasses import replace
import numpy as np
import pytest

from anyfesolver.general_contact.friction_point import initial_friction_point
from anyfesolver.general_contact.native_retained_station import NativeRetainedRoundStation
from anyfesolver.general_contact.native_surface_s3_friction import NativeS3MaterialPointFriction
from anyfesolver.general_contact.native_surface_contributor import NativeS3MaterialContributor
from anyfesolver.general_contact.s3_friction_point import S3SkinMaterialPoint
from test_general_contact_native_retained_s3_static import _analysis


def _contributor(*,radius=.0515,targets=(1.,),activation_policy='reject'):
    owner=_analysis(targets=targets)
    reference=owner.context.elements[0][1].operator.reference
    source=NativeRetainedRoundStation(1,reference,.5,radius,np.pi)
    corners=owner.contact.shell.reference_positions
    uv=np.linalg.lstsq(np.column_stack((corners[1]-corners[0],corners[2]-corners[0])),
                       np.array((1.5,0.,-.051))-corners[0],rcond=None)[0]
    target=S3SkinMaterialPoint(owner.contact.shell,tuple((1-uv.sum(),*uv)))
    pair=NativeS3MaterialPointFriction(source,target,reference_weight_m2=.02,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.,
        activation_policy=activation_policy)
    history=initial_friction_point(target.physical_pose(owner.initial.shell_displacement).material_frame)
    return owner,NativeS3MaterialContributor(owner,pair),history


@pytest.mark.parametrize('slide,branch',[(.00001,'stick'),(.001,'slip')])
def test_coupled_material_contributor_full_retained_tangent_and_no_line_double_count(slide,branch):
    owner,bridge,history=_contributor()
    increment=np.zeros(owner.context.count); increment[6]=increment[12]=slide
    shell=np.zeros(owner.shell_stiffness.shape[0]); shell[owner.shell_free[0]]=-1e-6
    origin=owner.initial
    candidate=bridge.evaluate(origin,history,increment,shell,1.)
    assert candidate.material.contribution.trial.branch==branch
    # Compare assembled contact against the local mapped work, independently of
    # the old line-contact assembly. No second normal law is added.
    structural_r,structural_j,_=owner.context.assemble(candidate.mechanical,1.,origin.native_histories)[:3]
    expected=np.r_[structural_r,owner.shell_stiffness@candidate.shell_displacement]
    slots=np.r_[owner.context.slots[0],owner.context.count+np.arange(len(shell))]
    np.add.at(expected,slots,candidate.material.contribution.residual)
    np.testing.assert_allclose(candidate.residual,expected,rtol=0.,atol=1e-13)
    assert candidate.spatial_tangent.shape==(60,60)
    assert np.linalg.norm(candidate.spatial_tangent[18:24,42:])>0.
    assert np.linalg.norm(candidate.spatial_tangent[42:,18:24])>0.
    # Physical retained rotations and algebraic resultant blocks are both kept.
    np.testing.assert_allclose(candidate.spatial_tangent[:42,24:42],structural_j[:,24:42],rtol=0.,atol=1e-13)
    direction=np.random.default_rng(135).normal(size=60)*.01
    direction[list(owner.context.fixed)]=0.
    direction[owner.context.count+owner.shell_fixed]=0.
    for h in (1e-5,2e-6):
        plus=bridge.evaluate(origin,history,increment+h*direction[:42],shell+h*direction[42:],1.)
        minus=bridge.evaluate(origin,history,increment-h*direction[:42],shell-h*direction[42:],1.)
        fd=(plus.residual-minus.residual)/(2*h)
        assert np.linalg.norm(candidate.origin_chart_tangent@direction-fd)/np.linalg.norm(fd)<=1e-5
    assert history.partner_key is None
    np.testing.assert_array_equal(origin.shell_displacement,np.zeros(18))


def test_material_contributor_rejects_foreign_geometry_and_supported_state_changes():
    owner,bridge,history=_contributor()
    origin=owner.initial
    bad=np.zeros(18); bad[owner.shell_fixed[0]]=1e-4
    with pytest.raises(ValueError,match='supported increments'):
        bridge.evaluate(origin,history,np.zeros(42),bad,1.)
    with pytest.raises(ValueError,match='origin violates shell supports'):
        bridge.evaluate(replace(origin,shell_displacement=bad),history,np.zeros(42),np.zeros(18),1.)
    pair=bridge.pair
    foreign=NativeRetainedRoundStation(2,pair.source.reference,.5,.0515,np.pi)
    other=NativeS3MaterialPointFriction(foreign,pair.target,reference_weight_m2=.02,
        normal_stiffness_Pa_per_m=1000.,friction_coefficient=.3,tangential_stiffness_Pa_per_m=1000.)
    with pytest.raises(ValueError,match='native reference'):
        NativeS3MaterialContributor(owner,other)
    pair.source.radius*=1.01
    with pytest.raises(ValueError,match='definition changed'):
        bridge.evaluate(origin,history,np.zeros(42),np.zeros(18),1.)


def test_finite_nonaxial_rotation_origin_chart_tangent():
    owner,bridge,history=_contributor()
    increment=np.zeros(42)
    increment[9:12]=(.04,-.01,.03)
    increment[21:24]=(.01,.02,.08)
    shell=np.zeros(18)
    candidate=bridge.evaluate(owner.initial,history,increment,shell,1.)
    direction=np.zeros(60)
    direction[9:12]=(.3,-.2,.5)
    direction[21:24]=(-.4,.3,.2)
    # Nonzero finite rotations distinguish the two coordinate conventions.
    assert np.linalg.norm((candidate.spatial_tangent-candidate.origin_chart_tangent)@direction)>1e-5
    for h in (1e-5,2e-6):
        plus=bridge.evaluate(owner.initial,history,increment+h*direction[:42],shell,1.)
        minus=bridge.evaluate(owner.initial,history,increment-h*direction[:42],shell,1.)
        fd=(plus.residual-minus.residual)/(2*h)
        assert np.linalg.norm(candidate.origin_chart_tangent@direction-fd)/np.linalg.norm(fd)<=1e-5
