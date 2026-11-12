"""One bounded source/empty-handoff diagnostic; no continuation step is run."""
from hashlib import sha256
from pathlib import Path
from time import monotonic,process_time
import json
from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse
from test_ge_beam3_schur_line_program import save

BASE='docs/reference_cases/ge_beam3_uniform_arch_sixteen_status.json'


def test_sixteen_macro_empty_handoff(tmp_path):
    base_raw=Path(BASE).read_bytes();base=parse(base_raw);archive=Path(base['archive']['path'])
    manifest_raw=(archive/'archive-manifest.json').read_bytes()
    assert sha256(manifest_raw).hexdigest().upper()==base['archive']['manifest_sha256']
    rows=[r for r in parse(manifest_raw)['files'] if r['path'].startswith('runs/cycle-a-native/') and r['path'].endswith('/checkpoint.json')]
    assert len(rows)==1;row=rows[0];source_raw=(archive/row['path']).read_bytes()
    assert len(source_raw)==row['bytes']==base['checkpoint_bytes']
    assert sha256(source_raw).hexdigest()==row['sha256']==base['checkpoint_sha256']
    save(tmp_path/'input.json',dict(base_status_sha256=sha256(base_raw).hexdigest(),source_bytes=len(source_raw),
        source_sha256=sha256(source_raw).hexdigest(),manifest_sha256=sha256(manifest_raw).hexdigest()))
    from test_ge_beam3_uniform_arch_sixteen import make
    from test_ge_beam3_native_arc import packet
    from anyfesolver._ge_beam3_native_arc import ArcProgram,solve_arc
    from anyfesolver._ge_beam3_native_arc_source import TranslationArcSource
    from anyfesolver._ge_beam3_native_arc_restart import decode_checkpoint,encode_checkpoint
    from anyfesolver._ge_beam3_native_history_profile import HISTORY8M
    from anyfesolver._ge_beam3_p5_seeded.core import canonical
    model,translation=make();source=TranslationArcSource(translation,source_raw,row['sha256'],1.)
    program=ArcProgram((.12,.12,.12),.01,translation.distributed,parameter_scale=.1,history_profile=HISTORY8M,source=source)
    started=monotonic();cpu=process_time();timings={};phase='initial_handoff'
    try:
        print(dict(stage=phase),flush=True)
        result=solve_arc(model,program,stop_after=0)
        timings[phase]=dict(wall_seconds=monotonic()-started,cpu_seconds=process_time()-cpu)
        save(tmp_path/'result.json',packet(result));save(tmp_path/'checkpoint.json',result.checkpoint)
        assert result.status=='paused' and result.completed_steps==0 and result.failure is None,result.failure
        assert len(result.checkpoint)<8*1024**2
        phase='decode';started=monotonic();cpu=process_time();print(dict(stage=phase),flush=True)
        fresh,_=make();chain,records=decode_checkpoint(fresh,program,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
        timings[phase]=dict(wall_seconds=monotonic()-started,cpu_seconds=process_time()-cpu)
        assert len(chain)==4 and records==()
        phase='encode';started=monotonic();cpu=process_time();print(dict(stage=phase),flush=True)
        assert encode_checkpoint(fresh,program,chain,records)==result.checkpoint
        timings[phase]=dict(wall_seconds=monotonic()-started,cpu_seconds=process_time()-cpu)
        value=json.loads(result.checkpoint)
        assert canonical(value['source_checkpoint'])==source_raw
        assert value['accepted_chain']['snapshots']==value['source_checkpoint']['accepted_chain']['snapshots']
        assert result.parameter==json.loads(source_raw)['records'][-1]['parameter']
        save(tmp_path/'assessment.json',dict(status='PASS_EMPTY_HANDOFF_COST_ONLY',checkpoint_bytes=len(result.checkpoint),
            checkpoint_sha256=sha256(result.checkpoint).hexdigest(),source_prefix_exact=True,roundtrip_exact=True,
            arc_steps_executed=0,accepted_snapshots=len(chain),newton_solve_executed=False,production_qualified=False))
    except Exception as error:
        save(tmp_path/'incident.json',dict(phase=phase,error=type(error).__name__,message=str(error),retry_authorized=False))
        raise
    finally:
        save(tmp_path/'timings-diagnostic.json',timings)
    assert Path(BASE).read_bytes()==base_raw
