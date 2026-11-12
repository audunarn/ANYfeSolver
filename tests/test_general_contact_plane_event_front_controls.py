"""Batched-front controls/restart identities, without a second whole solve."""
from dataclasses import replace
from pathlib import Path
import runpy
import numpy as np
import pytest
from anyfesolver import ContactTransientSolverControls,TransientConfig
from anyfesolver.general_contact.transient import _restart_analysis_contract
from anyfesolver.general_contact.multiple_shell_planes import MultipleShellPlaneManager
from anyfesolver.general_contact.core import ContactSolverSettings
from anyfesolver.general_contact.plane_newmark_events import localize_endpoint_event
from anyfesolver.general_contact.event_subdivision import EventNeedsCutback


@pytest.mark.parametrize('controls',[
    dict(batch_plane_event_fronts_dev=1),dict(batch_plane_event_fronts_dev=True),
    dict(batch_plane_event_fronts_dev=True,q4_plane_event_subdivision_dev=True)])
def test_front_admission_rejects_invalid_and_nonmultiple_profiles(controls):
    with pytest.raises(ValueError,match='batched plane event'):
        ContactTransientSolverControls(**controls)


def test_front_selection_and_revision_are_restart_bound_without_changing_old_identity():
    example=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_deforming_q4_planes_dev.py'))
    model,definition,modes=example['build_model']()
    manager=MultipleShellPlaneManager(model,definition,ContactSolverSettings())
    controls=example['CONTROLS']
    transient=TransientConfig(1.e-5,.00002,initial_velocity=modes@np.array([.002,.05]))
    def contract(c):return _restart_analysis_contract(transient,c,manager,
        'MULTIPLE_Q4_FIXED_PLANES_NEWMARK_EVENT_DEV_V1',None,np.zeros(modes.shape[0]),None)
    off=contract(replace(controls,batch_plane_event_fronts_dev=False));on=contract(controls)
    assert 'batch_plane_event_fronts_dev' not in off['controls']
    assert 'plane_event_front_revision' not in off['controls']
    assert on['controls']['batch_plane_event_fronts_dev'] is True
    assert on['controls']['plane_event_front_revision']=='SHARED_MINIMUM_ORIENTED_GAP_V1'
    stripped=dict(on['controls'])
    del stripped['batch_plane_event_fronts_dev'];del stripped['plane_event_front_revision']
    assert stripped==off['controls']
    from anyfesolver.nonlinear_restart import analysis_configuration_fingerprint
    assert analysis_configuration_fingerprint(on)!=analysis_configuration_fingerprint(off)


def test_batched_front_unequal_slopes_argmin_and_both_stopping_tolerances():
    # Earliest row begins above the later shallow row; the argmin changes en route.
    def callback(h):return [10.*(.3-h),.01*(.7-h)]
    result=localize_endpoint_event(callback,1.,(False,False),(0,1),minimum_dt=.01,
        time_tolerance=1.e-10,gap_tolerance=1.e-10,maximum_evaluations=64,batch_front=True)
    assert abs(result.duration-.3)<1.e-11
    assert abs(callback(result.duration)[0])<=1.e-10
    assert result.support_indices==(0,)
    # A gap-only or time-only stopping rule would falsely report success here.
    for scale,time_tolerance,gap_tolerance in ((1.e-20,1.e-12,1.),(1.e12,.1,1.e-12)):
        with pytest.raises(EventNeedsCutback,match='budget|cannot resolve'):
            localize_endpoint_event(lambda h:[scale*(.301-h)],1.,(False,),(0,),
                minimum_dt=.01,time_tolerance=time_tolerance,gap_tolerance=gap_tolerance,
                maximum_evaluations=3,batch_front=True)
