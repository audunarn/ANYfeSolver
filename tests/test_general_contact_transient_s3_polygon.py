"""Explicit polygon-pressure profile retains V3 physical guard and transactions."""
from dataclasses import replace
from pathlib import Path
import runpy
import numpy as np
import pytest
from anyfesolver import ContactTransientSolverControls
from anyfesolver.general_contact.normal_s3_skin_path import prepare_normal_s3_skin_path
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from anyfesolver.general_contact.runtime import active_contact_manager

demo=runpy.run_path(str(Path(__file__).parents[1]/'examples/general_contact_deforming_s3_impact_dev.py'))
CONTROLS=replace(demo['CONTROLS'],normal_s3_polygon_pressure_bounds_dev=True)


@pytest.mark.parametrize('settings',({'normal_s3_polygon_pressure_bounds_dev':True},
    {'allow_normal_s3_skin_hermite_dev':True,'normal_s3_polygon_pressure_bounds_dev':1},
    {'allow_normal_s3_skin_hermite_dev':True,'normal_s3_polygon_pressure_bounds_dev':True,
     'q4_plane_event_subdivision_dev':True}))
def test_polygon_selection_is_explicit_and_bounded(settings):
    with pytest.raises(ValueError):ContactTransientSolverControls(**settings)


def test_polygon_pressure_owner_retains_hidden_material_approach_guard():
    model,definition,modes=demo['build_model']()
    pair=replace(definition.interactions[0],gap_tolerance=.0003)
    manager=GeneralS3PatchManager(model,replace(definition,interactions=(pair,)),demo['_fixture']['SETTINGS'])
    owner=prepare_normal_s3_skin_path(model,manager,modes)
    q=np.zeros(7);rate=np.r_[np.full(4,-.0016004),np.zeros(3)]
    lower,failure=owner.overlap_sweep(manager,modes@q,modes@q,modes@rate,-modes@rate,1.,
        event_gap_tolerance_m=CONTROLS.plane_event_gap_tolerance_m)
    assert lower is None and 'hidden_approach' in failure
    phase=owner.pressure_sweep(manager,modes@q,modes@q,modes@rate,-modes@rate,1.,
        active=np.zeros(4,dtype=bool),gap_tolerance_m=CONTROLS.plane_event_gap_tolerance_m,
        polygon_bounds=True)
    assert not phase.success


def test_v16_active_restart_and_pressure_algorithm_identity(monkeypatch):
    from anyfesolver.general_contact import normal_s3_skin_path as module
    original=module.NormalS3SkinPath.overlap_sweep;guards=[]
    def observed(self,*args,**kwargs):
        guards.append(kwargs['event_gap_tolerance_m'])
        return original(self,*args,**kwargs)
    monkeypatch.setattr(module.NormalS3SkinPath,'overlap_sweep',observed)
    _,whole,_=demo['solve'](controls=CONTROLS)
    assert whole.profile=='NORMAL_NODAL_S3_POLYGON_PRESSURE_NEWMARK_EVENT_DEV_V4'
    index=int(np.argmax([row.active_pressure_dofs for row in whole.contact]))
    assert whole.contact[index].active_pressure_dofs>0 and index>0
    _,prefix,_=demo['solve'](end=float(whole.times[index]),controls=CONTROLS,emit=True)
    packet=prefix.restart_checkpoint
    assert packet['version']==16 and packet['schema'].endswith('_V16')
    _,resumed,_=demo['solve'](controls=CONTROLS,checkpoint=packet)
    for name,tolerance in (('displacements',1.e-10),('velocities',1.e-6),('accelerations',1.e-2)):
        combined=np.concatenate((getattr(prefix,name)[:-1],getattr(resumed,name)))
        np.testing.assert_allclose(combined,getattr(whole,name),rtol=0.,atol=tolerance)
    assert [row.active_pressure_dofs for row in (*prefix.contact[:-1],*resumed.contact)]==[
        row.active_pressure_dofs for row in whole.contact]
    with pytest.raises(ValueError,match='version'):
        demo['solve'](checkpoint=packet)
    with pytest.raises(ValueError,match='analysis'):
        demo['solve'](controls=replace(CONTROLS,plane_event_gap_tolerance_m=2.e-10),checkpoint=packet)
    assert guards and set(guards)=={CONTROLS.plane_event_gap_tolerance_m}
    assert active_contact_manager() is None
