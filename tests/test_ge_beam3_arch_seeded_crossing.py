"""Frozen three-step native continuation from authenticated pre-peak history."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse,canonical
from test_ge_beam3_schur_line_program import save

BASE='docs/reference_cases/ge_beam3_uniform_arch_sixteen_status.json'
COST='docs/reference_cases/ge_beam3_arch_source_cost_status.json'
COST_SHA='bfa0897280679a20a2fee2b8dc5866310db82b5c95dfbe38fc90bdd80f53a2ab'
SOURCE_SHA='c65730e97ec1097832c95da21312ec67162a8b0b2422274c701be4131b1efca0'


def source_bytes():
    cost_raw=Path(COST).read_bytes()
    assert sha256(cost_raw).hexdigest()==COST_SHA
    cost=parse(cost_raw);assert cost['status']=='PASS_EMPTY_HANDOFF_COST_ONLY'
    raw=Path(BASE).read_bytes();base=parse(raw);archive=Path(base['archive']['path'])
    manifest_raw=(archive/'archive-manifest.json').read_bytes()
    assert sha256(manifest_raw).hexdigest().upper()==base['archive']['manifest_sha256']
    rows=[r for r in parse(manifest_raw)['files'] if r['path'].startswith('runs/cycle-a-native/') and r['path'].endswith('/checkpoint.json')]
    assert len(rows)==1;row=rows[0];source=(archive/row['path']).read_bytes()
    assert len(source)==row['bytes']==base['checkpoint_bytes']==2372889
    assert sha256(source).hexdigest()==row['sha256']==base['checkpoint_sha256']==SOURCE_SHA
    assert cost['input']['source_sha256']==SOURCE_SHA
    return source,dict(base_status_sha256=sha256(raw).hexdigest(),cost_status_sha256=COST_SHA,
        manifest_sha256=sha256(manifest_raw).hexdigest(),source_bytes=len(source),source_sha256=SOURCE_SHA)


def test_seeded_arch_crossing(tmp_path):
    source_raw,inputs=source_bytes();save(tmp_path/'input.json',inputs)
    # Mechanics and reference evaluation follow authenticated input checks.
    import numpy as np
    from test_ge_beam3_uniform_arch_sixteen import make,SECTION
    from test_ge_beam3_native_arc import packet
    from anyfesolver._ge_beam3_native_arc import ArcProgram,solve_arc
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
    from anyfesolver._ge_beam3_native_arc_restart import decode_checkpoint,encode_checkpoint
    from anyfesolver._ge_beam3_native_history_profile import HISTORY8M,load
    from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
    from docs.reference_cases.ge_beam3_native_control_slope import accepted_slope
    from docs.reference_cases.ge_beam3_uniform_arch_reference import solve as continuum
    from docs.reference_cases.ge_beam3_uniform_arch_fields import fields
    phase='source_and_continuation'
    try:
        print(dict(stage='authenticated_source'),flush=True)
        model,translation=make()
        seed=TranslationArcSource(translation,source_raw,SOURCE_SHA,1.)
        program=ArcProgram((.12,.12,.12),.01,translation.distributed,
            parameter_scale=.1,history_profile=HISTORY8M,source=seed)
        result=solve_arc(model,program,progress=lambda row:print(row,flush=True))
        save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
        assert result.status=='completed' and result.completed_steps==3,result.failure
        phase='complete_history_roundtrip';print(dict(stage=phase),flush=True)
        fresh,_=make()
        chain,records=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
        assert len(chain)==7 and len(records)==3
        assert encode_checkpoint(fresh,program,chain,records)==result.checkpoint
        packet_value=load(result.checkpoint,HISTORY8M);source_value=load(source_raw,HISTORY8M)
        assert canonical(packet_value['source_checkpoint'])==source_raw
        assert packet_value['accepted_chain']['snapshots'][:4]==source_value['accepted_chain']['snapshots']
        rows=[];previous=None
        for snapshot,record in zip(chain[4:],records,strict=True):
            phase='fields_and_slope_'+str(record['index']);print(dict(stage=phase),flush=True)
            # Frozen model: 33 sequential nodes, six global DOFs per node, crown 17.
            assert snapshot['displacements'].shape==(198,)
            drop=float(-snapshot['displacements'][97])
            assert 0.<drop<.02
            recovery=[];parameters=[]
            for eid,e in sorted(fresh.mesh.elements.items()):
                local=snapshot['states'][eid]
                assert all(sum(h.accumulated)==0. and not any(sum(v) for v in h.plastic) for h in local['response'].history.stations)
                data=recover_native_fields(e,fresh.mesh,local,expected_committed_total_u=snapshot['displacements'][list(e.get_dof_mapping(fresh.mesh))])
                for station in data['stations']:
                    parameters.append(float(e.operator.reference.position(station['xi'])[0]));recovery.append(station)
            assert len(recovery)==128
            x=np.array(parameters);samples=np.unique(np.r_[np.linspace(-1.,0.,129),-np.abs(x)])
            previous=continuum(drop,previous=previous,profile='BVP9',sample_parameters=samples.tolist())
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
            slope=accepted_slope(fresh,translation,snapshot,record['parameter'])
            row=dict(step=record['index'],drop=drop,density=record['parameter'],reference_density=previous.density,
                load_error=abs(record['parameter']/previous.density-1.),native_drop_slope=-slope['parameter_per_control'],
                reference_drop_slope=previous.slope,slope_normalized_error=abs(-slope['parameter_per_control']-previous.slope)/max(1.,abs(previous.slope)),
                resultant_compliance_norm_error=field_error,position_max_error=float(np.max(np.abs(positions-expected['position']))),
                frame_max_error=float(np.max(np.abs(frames-expected['frame']))),constitutive_error=constitutive,stations=128)
            save(tmp_path/('state-'+str(record['index'])+'.json'),dict(comparison=row,recovery=recovery,reference=asdict(previous),expected=expected,slope=slope))
            rows.append(row);print(dict(stage='reference_and_slope_complete',step=record['index'],drop=drop,parameter=record['parameter']),flush=True)
        comparison=dict(rows=rows,inputs=inputs,checkpoint_roundtrip=True,checkpoint_bytes=len(result.checkpoint),
            checkpoint_sha256=sha256(result.checkpoint).hexdigest(),source_prefix_exact=True,
            accepted_snapshots=7,macros=16,history_profile=HISTORY8M,arc_steps_executed=3,
            full_spatial_stability=False,production_qualified=False,independent_review='PENDING',
            all_load_errors_below_two_percent=all(r['load_error']<.02 for r in rows),
            native_slope_sign_change=rows[0]['native_drop_slope']>0 and rows[-1]['native_drop_slope']<0,
            native_descending_load=any(b['density']<a['density'] for a,b in zip(rows,rows[1:])),
            reference_descending_branch=rows[-1]['reference_drop_slope']<0,
            drops_increasing=all(b['drop']>a['drop'] for a,b in zip(rows,rows[1:])))
        save(tmp_path/'comparison.json',comparison)
        assert source_bytes()[0]==source_raw
        assert comparison['all_load_errors_below_two_percent']
        assert comparison['native_slope_sign_change'] and comparison['native_descending_load']
        assert comparison['reference_descending_branch'] and comparison['drops_increasing']
    except Exception as error:
        save(tmp_path/'incident.json',dict(phase=phase,error=type(error).__name__,message=str(error),retry_authorized=False))
        raise
