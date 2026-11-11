"""Bounded native adaptive arch and nonlinear restart development evidence."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import pytest
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse,canonical
from test_ge_beam3_schur_line_program import save

STATUS='docs/reference_cases/ge_beam3_native_arc_adaptive_status.json'
STATUS_SHA='2751196fbb083e9d333a27d6fa680a0a4e1698fb853a82dcac7b6d108b01c778'


def authority():
    raw=Path(STATUS).read_bytes();assert sha256(raw).hexdigest()==STATUS_SHA
    status=parse(raw);assert status['status']=='PASS_DEVELOPMENT_NATIVE_ADAPTIVE_ARC_ONLY'
    root=Path(status['archive']['path']);manifest=(root/'archive-manifest.json').read_bytes()
    assert sha256(manifest).hexdigest().upper()==status['archive']['manifest_sha256']
    return root,parse(manifest),dict(status_sha256=STATUS_SHA,manifest_sha256=sha256(manifest).hexdigest())


def archived(root,manifest,prefix,name):
    rows=[r for r in manifest['files'] if r['path'].startswith(prefix) and r['path'].endswith('/'+name)]
    assert len(rows)==1;row=rows[0];raw=(root/row['path']).read_bytes()
    assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
    return raw


@pytest.mark.parametrize('kind',('accepted','cutback'))
def test_curved_restart(kind,tmp_path):
    root,manifest,inputs=authority();lane='curved' if kind=='accepted' else 'rejection'
    prefix_path='runs/cycle-a-'+lane+'/pytest/'
    source_raw=archived(root,manifest,prefix_path,'source.json')
    expected=archived(root,manifest,prefix_path,'checkpoint.json')
    inputs.update(kind=kind,source_sha256=sha256(source_raw).hexdigest(),expected_sha256=sha256(expected).hexdigest())
    save(tmp_path/'input.json',inputs)
    import numpy as np
    from anyfesolver import _ge_beam3_native_arc_adaptive as adaptive
    from anyfesolver import _ge_beam3_native_arc as arc
    from anyfesolver._ge_beam3_native_arc_restart import encode_checkpoint as encode_arc
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
    from anyfesolver._ge_beam3_native_translation import TranslationProgram
    from anyfesolver._ge_beam3_spatial_nodal_moments import SpatialNodalMoments
    from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
    from test_ge_beam3_native_generalized_restart import make,pattern
    from test_ge_beam3_native_arc_adaptive import packet
    phase='decode_preserved';print(dict(kind=kind,stage=phase),flush=True)
    try:
        model=make('curved-plastic')
        translation=TranslationProgram((.03,),3,'ux',pattern(model),SpatialNodalMoments(((3,.015,-.01,.008),)))
        prototype=arc.ArcProgram((.04,),1.,translation.distributed,translation.nodal_moments,
            source=TranslationArcSource(translation,source_raw,sha256(source_raw).hexdigest(),1.))
        program=adaptive.AdaptiveArcProgram(prototype,2 if kind=='accepted' else 1,.01,.08)
        chain,records,attempts,inner=adaptive.decode_checkpoint(model,program,expected,expected_sha256=sha256(expected).hexdigest())
        keep=1 if kind=='accepted' else 0;phase='prefix_construction'
        fixed=replace(prototype,steps=tuple(r['step_size'] for r in records[:keep]) or prototype.steps)
        prefix_inner=encode_arc(model,fixed,chain[:2+keep],records[:keep])
        capsule=adaptive.wrap_checkpoint(model,program,prefix_inner,attempts[:1])
        save(tmp_path/'prefix.json',capsule)
        fresh=make('curved-plastic')
        prefix_chain,prefix_rows,prefix_attempts,_=adaptive.decode_checkpoint(fresh,program,capsule,expected_sha256=sha256(capsule).hexdigest())
        assert len(prefix_rows)==keep and len(prefix_chain)==2+keep
        assert prefix_attempts[0]['disposition']==('ACCEPTED' if kind=='accepted' else 'CUTBACK')
        phase='native_restart';print(dict(kind=kind,stage=phase),flush=True)
        result=adaptive.solve_adaptive_arc(fresh,program,checkpoint=capsule,expected_sha256=sha256(capsule).hexdigest(),progress=lambda r:print(r,flush=True))
        save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
        assert result.status=='completed',result.failure
        assert result.checkpoint==expected
        assert [r['step'] for r in result.events if r['stage']=='committed']==[keep+1]
        check=make('curved-plastic')
        restored,rows,trace,last_inner=adaptive.decode_checkpoint(check,program,result.checkpoint,expected_sha256=sha256(expected).hexdigest())
        assert adaptive.wrap_checkpoint(check,program,last_inner,trace)==expected
        recovery={i:recover_native_fields(e,check.mesh,restored[-1]['states'][i],expected_committed_total_u=restored[-1]['displacements'][list(e.get_dof_mapping(check.mesh))]) for i,e in check.mesh.elements.items()}
        positive=sum(sum(h.accumulated)>0 for s in restored[-1]['states'].values() for h in s['response'].history.stations)
        assert positive==8 and np.array_equal(result.displacements,chain[-1]['displacements'])
        save(tmp_path/'recovery.json',recovery)
        save(tmp_path/'assessment.json',dict(kind=kind,final_checkpoint_byte_identical=True,source_rerun=False,
            rejected_attempt_rerun=False,new_accepted_steps=1,positive_plastic_stations=positive,full_roundtrip=True,
            expected_sha256=sha256(expected).hexdigest(),prefix_sha256=sha256(capsule).hexdigest(),production_qualified=False))
        assert authority()[2]=={k:inputs[k] for k in ('status_sha256','manifest_sha256')}
    except Exception as error:
        save(tmp_path/'incident.json',dict(kind=kind,phase=phase,error=type(error).__name__,message=str(error),retry_authorized=False))
        raise


def test_adaptive_arch_growth(tmp_path):
    from test_ge_beam3_arch_seeded_crossing import source_bytes,SOURCE_SHA
    from dataclasses import asdict
    _,_,binding=authority()
    source_raw,inputs=source_bytes();inputs.update(adaptive_authority=binding);save(tmp_path/'input.json',inputs)
    # Mechanics and reference evaluation follow authenticated input checks.
    import numpy as np
    from test_ge_beam3_uniform_arch_sixteen import make,SECTION
    from test_ge_beam3_native_arc_adaptive import packet
    from anyfesolver._ge_beam3_native_arc import ArcProgram
    from anyfesolver._ge_beam3_native_arc_adaptive import AdaptiveArcProgram,solve_adaptive_arc,decode_checkpoint,wrap_checkpoint
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
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
        prototype=ArcProgram((.12,),.01,translation.distributed,
            parameter_scale=.1,history_profile=HISTORY8M,source=seed)
        program=AdaptiveArcProgram(prototype,3,.03,.24)
        result=solve_adaptive_arc(model,program,progress=lambda row:print(row,flush=True))
        save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
        assert result.status=='completed' and result.completed_steps==3,result.failure
        phase='complete_history_roundtrip';print(dict(stage=phase),flush=True)
        fresh,_=make()
        chain,records,attempts,inner=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
        assert len(chain)==7 and len(records)==3
        assert wrap_checkpoint(fresh,program,inner,attempts)==result.checkpoint
        assert [r['step_size'] for r in records]==[.12,.12,.24]
        packet_value=load(result.checkpoint,HISTORY8M)['inner_checkpoint'];source_value=load(source_raw,HISTORY8M)
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
            adaptive_steps=[r['step_size'] for r in records],adaptive_attempts=attempts,
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
