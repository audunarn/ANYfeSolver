"""Signed external loading with actual simultaneous shell/fibre histories."""
import numpy as np
import pytest
from dataclasses import replace
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from examples.stiffened_shell_coupled_plastic_history import definition, activity, OPTIONS, CONTROL


@pytest.mark.parametrize('mesh,couple',[
    pytest.param('q4',0.,id='q4'),pytest.param('mixed',0.,id='mixed'),
    pytest.param('s3',0.,id='s3'),pytest.param('q4',100.,id='q4-couple')])
def test_signed_distributed_material_history_restart(request,record_property,mesh,couple):
    model=definition()
    if mesh!='q4':
        shell=replace(model.shell,q4_elements=() if mesh=='s3' else ((1,2,5,4),),
            s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)) if mesh=='s3' else ((2,3,6),(2,6,5)))
        model=(ss.LinearDefinition(model.profile,shell,model.beams,model.points,model.fixed)
               if mesh=='s3' else replace(model,shell=shell))
    a=ss.create_analysis(definition=model);request.addfinalizer(a.close)
    endpoints=tuple(tuple(ss.GEBeamLineAction(e.element_id,(0.,0.,density),
                                           (couple*density/3000.,0.,0.))
                         for e,_,_ in a._beam_parts) for density in (3000.,-3000.))
    programme=ss.GEBeamDistributedProgramme(a.preflight()['identity'],endpoints)
    force_only=(ss.GEBeamDistributedLoad(a._identity,tuple(
        ss.GEBeamLineAction(e.element_id,(0.,0.,3000.)) for e,_,_ in a._beam_parts)).compile(a)
        if couple else None)
    targets=(.5,1.,1.25,1.75,2.);token=CancellationToken();accepted=[];history=[]
    def observe(result):
        row=activity(result,accepted[-1] if accepted else None)
        row['position']=result.load_position
        ground=result.support_reactions[:a._shell_count].reshape(-1,6)
        xyz=a._shell_xyz+result.shell_displacement.reshape(-1,6)[:,:3]
        f=np.asarray(result.distributed_load.applied_force)
        m=np.asarray(result.distributed_load.applied_moment_about_origin)
        assert np.linalg.norm(ground[:,:3].sum(0)+f)/12000. <= 1e-8
        assert np.linalg.norm((np.cross(xyz,ground[:,:3])+ground[:,3:]).sum(0)+m)/24000. <= 1e-8
        p=result.load_position
        amplitude=p if p<=1 else 3-2*p
        expected=12000.*amplitude
        np.testing.assert_allclose(f,(0.,0.,expected),atol=1e-10)
        assert (result.distributed_load.potential is None)==bool(couple*amplitude)
        if force_only is not None:
            force_moment=np.asarray(force_only.evaluate(a._finite_driver._state)[2].applied_moment_about_origin)
            np.testing.assert_allclose(m-amplitude*force_moment,(4*couple*amplitude,0.,0.),atol=1e-9)
            row['applied_couple_Nm']=(4*couple*amplitude,0.,0.)
        accepted.append(result);history.append(row)
    def cancel(result):
        observe(result)
        if result.load_position==1.:token.cancel('yielded staged prefix')
    with pytest.raises(SolveCancelled):
        a.solve(programme,targets,increment_control=CONTROL,on_accepted=cancel,
                cancellation_token=token,**OPTIONS)
    assert any(row['simultaneous'] for row in history)
    b=ss.create_analysis(definition=model);request.addfinalizer(b.close)
    restored=b.restore(a.checkpoint());assert a.checkpoint()==b.checkpoint()
    assert restored.distributed_load==a.recover().distributed_load
    last=a.solve(programme,targets,increment_control=CONTROL,on_accepted=observe,**OPTIONS)
    resumed=b.solve(programme,targets,increment_control=CONTROL,**OPTIONS)
    assert last.load_programme_complete and resumed.load_programme_complete
    assert last.distributed_load.applied_force[2]<0.
    assert any(row['simultaneous'] for row in history if row['position']>1.5)
    assert a.checkpoint()==b.checkpoint()
    assert last.distributed_load==resumed.distributed_load
    record_property('signed_material_history',history)
