"""Read-only successor assessment of preserved adaptive arch checkpoints.

The old production/validation attempt remains blocked. This does not rerun it,
reclassify it, or qualify the beam. Full native replay is deliberately retained.
"""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import pytest
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse, canonical

STATUS = 'docs/reference_cases/ge_beam3_adaptive_continuation_status.json'
STATUS_SHA = '8baaa4de3fd5de70c6f98be658b297db7254c00ac5f28bf25d8a2ea7e26f42a9'


def verified_file(root, rows, prefix, name):
    chosen = [r for r in rows if r['path'].startswith(prefix) and r['path'].endswith('/'+name)]
    if len(chosen) != 1:
        raise ValueError('exactly one registered archive file required')
    row = chosen[0]
    relative = Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative archive path required')
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular archived file required')
    raw = path.read_bytes()
    if len(raw) != row['bytes'] or sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('archived byte count/hash mismatch')
    return raw


def authority():
    raw = Path(STATUS).read_bytes()
    if sha256(raw).hexdigest() != STATUS_SHA:
        raise ValueError('frozen blocked closeout required')
    status = parse(raw)
    assert status['status'] == 'BLOCKED_GE_BEAM3_PROCESS_OR_EVIDENCE'
    root = Path(status['archive']['path'])
    manifest = (root/'archive-manifest.json').read_bytes()
    assert len(manifest) == status['archive']['manifest_bytes']
    assert sha256(manifest).hexdigest().upper() == status['archive']['manifest_sha256']
    return root, parse(manifest)['files'], dict(status_sha256=STATUS_SHA,
        manifest_sha256=sha256(manifest).hexdigest())


@pytest.mark.parametrize('mutation', ('hash', 'bytes', 'duplicate', 'missing', 'traversal'))
def test_archive_mutations(tmp_path, mutation):
    (tmp_path/'data').mkdir()
    raw = b'{"a":1}\n'; (tmp_path/'data'/'packet.json').write_bytes(raw)
    row = dict(path='data/packet.json', bytes=len(raw), sha256=sha256(raw).hexdigest())
    rows = [row]
    assert verified_file(tmp_path, rows, '', 'packet.json') == raw
    if mutation == 'hash': row['sha256'] = '0'*64
    elif mutation == 'bytes': row['bytes'] += 1
    elif mutation == 'duplicate': rows.append(dict(row))
    elif mutation == 'missing': rows.clear()
    else: row['path'] = '../data/packet.json'
    with pytest.raises(ValueError):
        verified_file(tmp_path, rows, '', 'packet.json')


@pytest.mark.parametrize('origin', ('cycle-a-arch', 'cycle-b-arch'))
def test_preserved_arch_assessment(origin, tmp_path):
    root, manifest, binding = authority()
    old = lambda name: verified_file(root, manifest, origin+'/pytest/', name)
    expected = lambda name: verified_file(root, manifest, 'rehearsal-arch/pytest/', name)
    # Authenticate all preserved inputs and reference outputs before mechanics.
    inputs_raw, checkpoint_raw, result_raw = (old(n) for n in ('input.json', 'checkpoint.json', 'result.json'))
    reference_files = {n: expected(n) for n in ('input.json','checkpoint.json','result.json',
        'state-1.json','state-2.json','state-3.json','comparison.json')}
    assert inputs_raw == reference_files['input.json']
    assert checkpoint_raw == reference_files['checkpoint.json']
    assert result_raw == reference_files['result.json']
    from test_ge_beam3_arch_seeded_crossing import source_bytes, SOURCE_SHA
    source_raw, _ = source_bytes()
    from dataclasses import asdict
    import numpy as np
    from test_ge_beam3_schur_line_program import save
    from test_ge_beam3_uniform_arch_sixteen import make, SECTION
    from anyfesolver._ge_beam3_native_arc import ArcProgram
    from anyfesolver._ge_beam3_native_arc_adaptive import AdaptiveArcProgram, decode_checkpoint, wrap_checkpoint
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
    from anyfesolver._ge_beam3_native_history_profile import HISTORY8M, load
    from anyfesolver._ge_beam3_native_generalized_recovery import recover_native_fields
    from docs.reference_cases.ge_beam3_native_control_slope import accepted_slope
    from docs.reference_cases.ge_beam3_uniform_arch_reference import solve as continuum
    from docs.reference_cases.ge_beam3_uniform_arch_fields import fields
    inputs = parse(inputs_raw)
    assert inputs['source_sha256'] == SOURCE_SHA
    result = SimpleNamespace(checkpoint=checkpoint_raw)
    for name, raw in (('input.json',inputs_raw),('checkpoint.json',checkpoint_raw),('result.json',result_raw)):
        save(tmp_path/name,raw)
    phase = 'reconstruct_programme'
    try:
        _, translation = make()
        seed = TranslationArcSource(translation,source_raw,SOURCE_SHA,1.)
        prototype = ArcProgram((.12,),.01,translation.distributed,
            parameter_scale=.1,history_profile=HISTORY8M,source=seed)
        program = AdaptiveArcProgram(prototype,3,.03,.24)
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
        # Entire assessment, not only the terminal flag, must match the completed rehearsal.
        for name, raw in reference_files.items():
            assert (tmp_path/name).read_bytes() == raw, name
        assert authority()[2] == binding
        save(tmp_path/'receipt.json',dict(origin=origin,binding=binding,
            checkpoint_sha256=sha256(checkpoint_raw).hexdigest(),
            mechanics_programme_rerun=False,full_history_replayed=True,
            historical_gate_reclassified=False,production_qualified=False,
            independent_review='PENDING',reference_scientific_files_byte_identical=True))
    except Exception as error:
        save(tmp_path/'incident.json',dict(origin=origin,phase=phase,
            error=type(error).__name__,message=str(error),retry_authorized=False))
        raise
