"""Independent positive-gap path with unrelated clipped support."""
import numpy as np
from anyfesolver.general_contact.s3_pressure_hermite_bounds import s3_pressure_hermite_bounds


def test_temporal_gap_proof_does_not_spend_space_on_unrelated_clipped_modes(monkeypatch):
    # First face is entirely inside; the second crosses the x+y=2 target edge.
    xy=np.array(((.1,.1),(.3,.1),(.1,.3),(1.5,.1),(2.2,.1),(1.5,.8)))
    reference=np.c_[xy,np.zeros(6)]
    target=np.array(((0.,0.,0.),(2.,0.,0.),(0.,2.,0.)))
    start=reference.copy();end=reference.copy()
    # Independent exact gap: 4e-4*(t-.5)**2 + 1e-8 >= 1e-8.
    start[:3,2]=end[:3,2]=.00010001
    start[3:,2]=end[3:,2]=.01
    v0=np.zeros((6,3));v1=v0.copy()
    v0[:3,2]=-.0004;v1[:3,2]=.0004
    def evaluate():
        return s3_pressure_hermite_bounds(reference,target,((0,1,2),(3,4,5)),
        start,end,target,target,v0,v1,np.zeros((3,3)),np.zeros((3,3)),1.,
        active=(False,)*6,gap_tolerance_m=1e-10,
        maximum_time_panels=8,maximum_spatial_cells=64,maximum_depth=4)
    from anyfesolver.general_contact.s3_pressure_hermite_bounds import _SpatialPartition
    selected=_SpatialPartition.pop
    with monkeypatch.context() as patch:
        patch.setattr(_SpatialPartition,"pop",lambda self,violations=None:selected(self))
        legacy=evaluate()
    assert not legacy.success and legacy.status=="spatial_budget_exhausted"
    result=evaluate()
    assert result.success,result
    assert result.visited_time_panels>1
    assert result.visited_spatial_cells<=64
    assert all(value is None or value>=0 for value in result.open_joint_lower)
