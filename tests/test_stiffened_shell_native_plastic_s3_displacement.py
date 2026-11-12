"""Yielded S3 fixed-target lifecycle; not complete coupled qualification."""
from dataclasses import replace
import numpy as np
import pytest
from anymaterial import DNVC208MaterialCurve
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot, encode_snapshot
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import HISTORICAL_POLICY
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_native_point_displacement import owner, DIRECTION, CONTROL, OPTIONS
from test_stiffened_shell_native_coupled_material import definition_for
from test_stiffened_shell_native_s3_displacement_admission import compare_generalized
from anyfesolver.e4_pl_s3_v2d_state import seal_v2d_state


def definition(fibre=False, mixed_shell=False):
    base=definition_for(True,True) if fibre else _panel()
    pressure=20_000.
    curve=DNVC208MaterialCurve(pressure,1.05*pressure,1.1*pressure,
                             .005,.01,4.*pressure,.2)
    return replace(base,shell=replace(base.shell,j2_curve=curve,
        q4_elements=((1,2,5,4),) if mixed_shell else (),
        s3_elements=((2,3,6),(2,6,5)) if mixed_shell else
                    ((1,2,5),(1,5,4),(2,3,6),(2,6,5))),
        q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1 if mixed_shell else
                           HISTORICAL_POLICY)


def compare(left,right):
    assert left.history==right.history and left.amplitude==right.amplitude
    assert left.shell_layer_alpha==right.shell_layer_alpha
    assert left.beam_fibre_energy==right.beam_fibre_energy
    assert left.beam_station_accumulated==right.beam_station_accumulated
    if left.beam_generalized_material:
        compare_generalized(left.beam_generalized_material,right.beam_generalized_material)
    for name in ('shell_displacement','support_reactions','cell_resultants'):
        np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    for a,b in zip(left.shell_q4_material+left.shell_s3_material,
                   right.shell_q4_material+right.shell_s3_material,strict=True):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        if hasattr(a,'station_generalized_resultant'):
            np.testing.assert_array_equal(a.station_generalized_resultant,b.station_generalized_resultant)
    for a,b in zip(left.beam_fibre_material,right.beam_fibre_material,strict=True):
        for name in a.__dataclass_fields__:
            x,y=getattr(a,name),getattr(b,name)
            if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
            else:assert x==y


@pytest.mark.parametrize('fibre,mixed_shell',((False,False),(True,False)))
def test_yielded_s3_displacement_restart_continuation_and_rollback(request,tmp_path,fibre,mixed_shell):
    model=definition(fibre,mixed_shell)
    analysis=owner(request,model); resumed=owner(request,model); bad=owner(request,model)
    if mixed_shell:
        # Separate reference-seeded displacement route; failed .05 force seed stays open.
        reference=owner(request,model)
        # Use the established nominal1000N engineering action for target definition.
        # The earlier50N-equivalent target converged but left S3 material inactive.
        linear=reference.solve_linear((ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),))
        initial=float(linear.beam_coordinates[6*reference._beam_index[105]+2])
        seed=analysis.solve((ss.NodalAction('beam',105,(0.,0.,0.,0.,0.,0.)),),(1.,),**OPTIONS)
    else:
        seed=analysis.solve((ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),),(.05,),**OPTIONS)
        assert max(max(row) for row in seed.shell_layer_alpha)>0.
        initial=analysis._finite_driver._state.q[CONTROL.dof(analysis)]
    assert not analysis._finite_driver._stable_s3
    seed_alpha=np.concatenate([r.alpha for r in seed.shell_s3_material])
    seed_q=analysis._finite_driver._state.q[:analysis._shell_count].copy()
    kw=dict(OPTIONS,displacement_control=CONTROL)
    prefix=analysis.solve(DIRECTION,(initial*1.01,),**kw)
    assert prefix.free_equilibrium_ratio<=1e-8 and prefix.constraint_max<=1e-9
    assert prefix.shell_s3_material and bool(prefix.shell_q4_material)==mixed_shell
    assert np.any(np.concatenate([r.alpha for r in prefix.shell_s3_material])>seed_alpha)
    assert not prefix.production_qualified
    saved=analysis.checkpoint(); (tmp_path/'plastic-s3-prefix.checkpoint').write_bytes(saved)
    metadata,arrays=decode_snapshot(analysis,saved)
    assert metadata['policy']=='B3_GE_NATIVE_POINT_SHELL_J2_S3_DISPLACEMENT_V1_DEV'
    np.testing.assert_array_equal(arrays['shell_origin_q'],seed_q)
    assert len(arrays['s3_states'])==2*len(model.shell.s3_elements)
    restored=resumed.restore(saved); compare(prefix,restored)
    assert resumed.checkpoint()==saved
    np.testing.assert_array_equal(resumed._finite_driver._shell_predecessor_q,seed_q)
    changed=dict(arrays); changed['s3_states']=arrays['s3_states'][len(model.shell.s3_elements):]+arrays['s3_states'][:len(model.shell.s3_elements)]
    corrupt=encode_snapshot(analysis,arrays=changed,**{k:v for k,v in metadata.items() if k not in ('policy','model','steps')})
    with pytest.raises(ValueError):bad.restore(corrupt)
    assert bad._generation==0 and bad._finite_driver is None
    def reject(values,match=None):
        raw=encode_snapshot(analysis,arrays=values,**{k:v for k,v in metadata.items() if k not in ('policy','model','steps')})
        with pytest.raises(ValueError,match=match):bad.restore(raw)
        assert bad._generation==0 and bad._finite_driver is None
    altered=dict(arrays); altered['shell_origin_q']=arrays['shell_origin_q'].copy()
    altered['shell_origin_q'][0]=1e-6
    reject(altered,'support coordinates')
    from scipy.spatial.transform import Rotation
    altered=dict(arrays); altered['shell_origin_anchors']=arrays['shell_origin_anchors'].copy()
    altered['shell_origin_anchors'][-1]=Rotation.from_rotvec((.01,0.,0.)).as_matrix()
    reject(altered,'rebased predecessor charts')
    q4_count=len(model.shell.q4_elements)
    element=analysis._shell_parts[q4_count][1]
    forged=dict(analysis._finite_driver._shell_origins[q4_count])
    forged['layer_stress']=np.array(forged['layer_stress'],copy=True)
    forged['layer_stress'][0,0]+=1.
    blob=element.serialize_nonlinear_state(analysis._shell_model.mesh,
        analysis._shell_model.get_material('shell'),seal_v2d_state(forged),3)
    altered=dict(arrays); altered['s3_states']=(blob,)+arrays['s3_states'][1:]
    reject(altered,'response disagrees with accepted update')
    token=CancellationToken();token.cancel('plastic S3 prefix')
    with pytest.raises(SolveCancelled):
        analysis.solve(DIRECTION,(initial*1.02,),cancellation_token=token,**kw)
    assert analysis.checkpoint()==saved
    with pytest.raises(RuntimeError):
        analysis.solve(DIRECTION,(initial*1.02,),displacement_control=CONTROL,
                       max_iterations=1,max_backtracks=0)
    assert analysis.checkpoint()==saved
    final=analysis.solve(DIRECTION,(initial*1.02,),**kw)
    continued=resumed.solve(DIRECTION,(initial*1.02,),**kw)
    assert analysis.checkpoint()==resumed.checkpoint();compare(final,continued)
    assert final.free_equilibrium_ratio<=1e-8 and final.constraint_max<=1e-9
    assert np.any(np.concatenate([r.alpha for r in final.shell_s3_material])>
                  np.concatenate([r.alpha for r in prefix.shell_s3_material]))
    (tmp_path/'plastic-s3-final.checkpoint').write_bytes(analysis.checkpoint())


def test_mixed_j2_s3_binding_and_rejected_nominal_target(request,tmp_path):
    """Dormant S3 history parity and failure rollback; NOT active mixed qualification."""
    model=definition(True,True)
    reference=owner(request,model)
    linear=reference.solve_linear((ss.NodalAction('beam',105,(0.,0.,1000.,0.,0.,0.)),))
    nominal=float(linear.beam_coordinates[6*reference._beam_index[105]+2])
    dormant=owner(request,model);resumed=owner(request,model);failed=owner(request,model)
    zero=(ss.NodalAction('beam',105,(0.,0.,0.,0.,0.,0.)),)
    dormant.solve(zero,(1.,),**OPTIONS);failed.solve(zero,(1.,),**OPTIONS)
    seed=failed.checkpoint()
    kw=dict(OPTIONS,displacement_control=CONTROL)
    # Frozen large-target failure is used solely as a rollback oracle.
    from anyfesolver._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure
    with pytest.raises(DisplacementCorrectorFailure,match='backtrack budget exhausted'):
        failed.solve(DIRECTION,(nominal*1.01,),**kw)
    assert failed.checkpoint()==seed and getattr(failed,'_displacement_snapshot',None) is None
    prefix=dormant.solve(DIRECTION,(nominal*.05*1.01,),**kw)
    assert prefix.shell_q4_material and prefix.shell_s3_material
    assert max(max(row.alpha) for row in prefix.shell_s3_material)==0.
    raw=dormant.checkpoint();repeated=resumed.restore(raw)
    compare(prefix,repeated);assert resumed.checkpoint()==raw
    (tmp_path/'mixed-dormant-prefix.checkpoint').write_bytes(raw)
