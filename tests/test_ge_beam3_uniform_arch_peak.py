"""New load-peak development samples; not a restart or retry of an old programme."""
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from docs.reference_cases.ge_beam3_native_control_slope import solve_control_slope,accepted_slope
from docs.reference_cases.ge_beam3_uniform_arch_reference import solve as continuum
from docs.reference_cases.ge_beam3_uniform_arch_fields import fields
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse
from anyfesolver._ge_beam3_native_translation import solve_translation
from anyfesolver._ge_beam3_native_translation_restart import decode_checkpoint,encode_checkpoint
from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
from anyfesolver._ge_beam3_native_history_profile import HISTORY8M
from test_ge_beam3_uniform_arch_sixteen import make as base_model,SECTION
from test_ge_beam3_native_translation import bar,packet
from test_ge_beam3_schur_line_program import save

BASE='docs/reference_cases/ge_beam3_uniform_arch_sixteen_status.json'
DROPS=(.006,.008,.010)


def make():
    model,program=base_model()
    return model,replace(program,targets=tuple(-v for v in DROPS))


@pytest.mark.parametrize('k',(-2.,0.,2.))
def test_bordered_slope_at_and_around_singular_stiffness(k,tmp_path):
    value,error=solve_control_slope(sparse.csr_matrix([[k]]),np.array([1.]),np.array([0]),0)
    assert np.array_equal(value,np.array([1.,-k])) and error==0.
    save(tmp_path/'slope.json',dict(stiffness=k,slope=value[-1],stiffness_inverted=False,error=error))


@pytest.mark.parametrize('kind',('column','matrix','free','control'))
def test_slope_input_mutation(kind,tmp_path):
    matrix=sparse.eye(2,format='csr');column=np.array([1.,1.]);free=np.array([0,1]);control=0
    if kind=='column':column[0]=np.nan
    elif kind=='matrix':matrix.data[0]=np.inf
    elif kind=='free':free[:]=0
    else:control=True
    with pytest.raises(ValueError):solve_control_slope(matrix,column,free,control)
    save(tmp_path/'rejected.json',dict(mutation=kind,rejected=True))


def test_analytic_bar_slope(tmp_path):
    model,program=bar();result=solve_translation(model,program)
    assert result.status=='completed',result.failure
    fresh,_=bar();chain,records=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    rows=[accepted_slope(fresh,program,snapshot,row['parameter']) for snapshot,row in zip(chain[1:],records,strict=True)]
    assert all(abs(row['parameter_per_control']-8.)<=1e-11 for row in rows)
    save(tmp_path/'bar.json',dict(rows=rows,checkpoint_sha256=sha256(result.checkpoint).hexdigest()))


def test_reference_peak_samples(tmp_path):
    previous=None;rows=[]
    for drop in DROPS:
        previous=continuum(drop,previous=previous,profile='BVP9')
        rows.append(asdict(previous))
    assert rows[0]['slope']>0 and rows[-1]['slope']<0 and rows[-1]['density']<rows[1]['density']
    save(tmp_path/'reference.json',dict(rows=rows,production_qualified=False))


def test_native_peak_comparison(tmp_path):
    raw=Path(BASE).read_bytes();base=parse(raw)
    assert base['all_load_errors_below_two_percent'] and base['real_large_checkpoint_roundtrip']
    model,program=make();result=solve_translation(model,program,progress=lambda row:print(row,flush=True))
    save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
    assert result.status=='completed',result.failure
    fresh,_=make();chain,records=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    assert encode_checkpoint(fresh,program,chain,records)==result.checkpoint
    rows=[];previous=None
    for snapshot,record in zip(chain[1:],records,strict=True):
        recovery=[];parameters=[]
        for eid,e in sorted(fresh.mesh.elements.items()):
            local=snapshot['states'][eid]
            assert all(sum(h.accumulated)==0. and not any(sum(v) for v in h.plastic) for h in local['response'].history.stations)
            data=recover_native_fields(e,fresh.mesh,local,expected_committed_total_u=snapshot['displacements'][list(e.get_dof_mapping(fresh.mesh))])
            for station in data['stations']:
                parameters.append(float(e.operator.reference.position(station['xi'])[0]));recovery.append(station)
        assert len(recovery)==128
        x=np.array(parameters);samples=np.unique(np.r_[np.linspace(-1.,0.,129),-np.abs(x)])
        previous=continuum(float(-record['target']),previous=previous,profile='BVP9',sample_parameters=samples.tolist())
        expected=fields(previous,x)
        force=np.array([r['resultants']+r['resultants_low'] for r in recovery])
        strain=np.array([r['strain']+r['strain_low'] for r in recovery])
        measures=np.array([r['measure'] for r in recovery])
        positions=np.array([r['current_position']+r['current_position_low'] for r in recovery])
        frames=np.array([r['current_frame'] for r in recovery])
        norm=float(np.sum(measures[:,None]*expected['resultants']**2/SECTION));assert norm>0
        field_error=float(np.sqrt(np.sum(measures[:,None]*(force-expected['resultants'])**2/SECTION)/norm))
        constitutive=float(np.max(np.abs(force-strain*SECTION)/np.maximum(1.,np.abs(force))))
        assert constitutive<=1e-11
        slope=accepted_slope(fresh,program,snapshot,record['parameter'])
        row=dict(step=record['index'],drop=-record['target'],density=record['parameter'],reference_density=previous.density,
            load_error=abs(record['parameter']/previous.density-1.),native_drop_slope=-slope['parameter_per_control'],
            reference_drop_slope=previous.slope,slope_normalized_error=abs(-slope['parameter_per_control']-previous.slope)/max(1.,abs(previous.slope)),
            resultant_compliance_norm_error=field_error,position_max_error=float(np.max(np.abs(positions-expected['position']))),
            frame_max_error=float(np.max(np.abs(frames-expected['frame']))),constitutive_error=constitutive,stations=len(recovery))
        save(tmp_path/('state-'+str(record['index'])+'.json'),dict(comparison=row,recovery=recovery,reference=asdict(previous),expected=expected,slope=slope))
        rows.append(row);print(dict(stage='reference_and_slope_complete',step=record['index']),flush=True)
    comparison=dict(rows=rows,base_status_sha256=sha256(raw).hexdigest(),checkpoint_roundtrip=True,checkpoint_bytes=len(result.checkpoint),
        macros=16,history_profile=HISTORY8M,native_arc_crossing=False,full_spatial_stability=False,production_qualified=False,
        all_load_errors_below_two_percent=all(r['load_error']<.02 for r in rows),
        native_slope_sign_change=rows[0]['native_drop_slope']>0 and rows[-1]['native_drop_slope']<0,
        native_descending_load=rows[-1]['density']<rows[1]['density'])
    save(tmp_path/'comparison.json',comparison)
    assert Path(BASE).read_bytes()==raw
    assert comparison['native_slope_sign_change'] and comparison['native_descending_load']
    assert comparison['all_load_errors_below_two_percent']
