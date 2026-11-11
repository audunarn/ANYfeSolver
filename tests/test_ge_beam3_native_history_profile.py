"""Explicit bounded history capacity, default bytes, and real native transactions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import pytest
from anyfesolver import _ge_beam3_native_history_profile as capacity
from anyfesolver._ge_beam3_p5_seeded.core import canonical,sha
from anyfesolver._ge_beam3_p5_seeded import codec as legacy
from test_ge_beam3_schur_line_program import save


def factory(kind):
    if kind=='translation':
        from test_ge_beam3_native_translation import bar,packet
        from anyfesolver._ge_beam3_native_translation import solve_translation as solve
        from anyfesolver import _ge_beam3_native_translation_restart as codec
    else:
        from test_ge_beam3_native_arc import bar,packet
        from anyfesolver._ge_beam3_native_arc import solve_arc as solve
        from anyfesolver import _ge_beam3_native_arc_restart as codec
    return bar,solve,codec,packet


@pytest.mark.parametrize('profile',[False,True,0,8388608,'wrong',{},[]])
def test_unknown_profile(profile,tmp_path):
    with pytest.raises(ValueError):capacity.load(b'{}\n',profile)
    for kind in ('translation','arc'):
        bar,_,_,_=factory(kind);m,p=bar()
        with pytest.raises(ValueError):replace(p,history_profile=profile).require(m)
    save(tmp_path/'rejected.json',dict(unknown_profile_rejected=True))


@pytest.mark.parametrize('raw',[b'{"a":0,"a":1}\n',b'{"a":NaN}\n',b'{"a":1e999}\n',b'{}',b' { }\n',b'\xff',b'['*33+b'0'+b']'*33+b'\n'])
def test_strict_parser(raw,tmp_path):
    for profile in (None,capacity.HISTORY8M):
        with pytest.raises((ValueError,UnicodeError)):capacity.load(raw,profile)
    save(tmp_path/'rejected.json',dict(strict_parser_rejected=True))


def test_size_bound_and_parallel_isolation(tmp_path):
    raw=canonical(dict(value='x'*(3*1024**2)))
    def evaluate(profile):
        if profile is None:
            with pytest.raises(ValueError):capacity.load(raw,profile)
            return 'DEFAULT_REJECTED'
        assert len(capacity.load(raw,profile)['value'])==3*1024**2
        return 'EXPLICIT_PROFILE_PARSED_ONLY'
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(evaluate,(None,capacity.HISTORY8M,None,capacity.HISTORY8M)))
    with pytest.raises(ValueError):capacity.load(b' '*(8*1024**2+1),capacity.HISTORY8M)
    assert legacy.MAX_BYTES==2*1024**2 and capacity.limit(None)==2*1024**2
    save(tmp_path/'size.json',dict(bytes=len(raw),parallel_results=values,legacy_bound_unchanged=True,
        synthetic_parser_payload_only=True,large_mechanical_checkpoint_tested=False))


@pytest.mark.parametrize('kind',('translation','arc'))
def test_invalid_programme_before_mechanics(kind,tmp_path):
    _,_,codec,_=factory(kind);raw=b'{}\n'
    with pytest.raises(ValueError,match='programme required'):
        codec.decode_checkpoint(None,None,raw,expected_sha256=sha256(raw).hexdigest())
    save(tmp_path/'programme.json',dict(programme=kind,invalid_programme_rejected_before_mechanics=True))


@pytest.fixture(scope='module',params=('translation','arc'))
def saved(request,tmp_path_factory):
    kind=request.param;bar,solve,codec,packet=factory(kind);m,old=bar();p=replace(old,history_profile=capacity.HISTORY8M)
    root=tmp_path_factory.mktemp(kind+'-profile');r=solve(m,p)
    save(root/'result.json',packet(r));save(root/'checkpoint.json',r.checkpoint)
    assert r.status=='completed',r.failure
    return kind,root,p,r


def test_profiled_roundtrip_resume_and_cross_profile(saved):
    kind,root,p,r=saved;bar,solve,codec,packet=factory(kind);m,default=bar()
    chain,records=codec.decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    assert codec.encode_checkpoint(m,p,chain,records)==r.checkpoint
    value=json.loads(r.checkpoint)
    assert value['history_profile']==value['program']['history_profile']==value['accepted_chain']['history_profile']==capacity.HISTORY8M
    assert value['schema']==capacity.schema(codec.SCHEMA,capacity.HISTORY8M)
    with pytest.raises(ValueError):codec.decode_checkpoint(m,default,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    prefix=codec.encode_checkpoint(m,p,chain[:2],records[:1]);fresh,_=bar()
    resumed=solve(fresh,p,checkpoint=prefix,expected_sha256=sha256(prefix).hexdigest())
    assert resumed.status=='completed' and resumed.checkpoint==r.checkpoint,resumed.failure
    save(root/'roundtrip.json',dict(profile=capacity.HISTORY8M,roundtrip=True,prefix_resume_byte_identical=True,default_rejects_successor=True))


@pytest.mark.parametrize('kind',['profile','missing','schema','inner-profile','inner-schema','programme','genesis','hash'])
def test_resealed_profile_or_history_mutation(saved,kind,tmp_path):
    name,_,p,r=saved;bar,_,codec,_=factory(name);m,_=bar();v=json.loads(r.checkpoint)
    if kind=='profile':v['history_profile']='wrong'
    elif kind=='missing':v.pop('history_profile')
    elif kind=='schema':v['schema']=codec.SCHEMA
    elif kind=='inner-profile':v['accepted_chain']['history_profile']='wrong'
    elif kind=='inner-schema':v['accepted_chain']['schema']='wrong'
    elif kind=='programme':v['program']['history_profile']='wrong';v['program_sha256']=sha(v['program'])
    elif kind=='genesis':v['accepted_chain']['snapshots'][0]['states'][0]['state']['epoch']=1
    v['checkpoint_sha256']=sha({k:x for k,x in v.items() if k!='checkpoint_sha256'});raw=canonical(v)
    with pytest.raises(ValueError):codec.decode_checkpoint(m,p,raw,expected_sha256='0'*64 if kind=='hash' else sha256(raw).hexdigest())
    save(tmp_path/'mutation.json',dict(programme=name,mutation=kind,rejected=True))


def test_profile_mutation_preserves_accepted_prefix(saved,tmp_path):
    kind,_,p,r=saved;bar,solve,codec,packet=factory(kind);m,_=bar()
    chain,records=codec.decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    expected=codec.encode_checkpoint(m,p,chain[:2],records[:1]);m,default=bar();changed=replace(default,history_profile=capacity.HISTORY8M)
    def observer(row):
        if row['stage']=='before_assembly' and row.get('target',row.get('step'))==2:object.__setattr__(changed,'history_profile',None)
    result=solve(m,changed,progress=observer)
    assert result.status=='failed' and result.checkpoint==expected
    fresh,_=bar();codec.decode_checkpoint(fresh,p,result.checkpoint,expected_sha256=sha256(result.checkpoint).hexdigest())
    save(tmp_path/'preserved.json',dict(programme=kind,result=packet(result),accepted_prefix_preserved=True))


def test_validation_deadline_unchanged(saved,monkeypatch,tmp_path):
    from anyfesolver import _ge_beam3_native_generalized_combined_restart as combined
    kind,_,p,r=saved;bar,_,codec,_=factory(kind);m,_=bar()
    chain,_=codec.decode_checkpoint(m,p,r.checkpoint,expected_sha256=sha256(r.checkpoint).hexdigest())
    clock=iter(range(0,100000,61));monkeypatch.setattr(combined,'monotonic',lambda:float(next(clock)))
    with pytest.raises(RuntimeError,match='deadline'):combined.encode_checkpoint(m,chain,history_profile=capacity.HISTORY8M)
    save(tmp_path/'deadline.json',dict(programme=kind,original_validation_deadline_enforced=True))


@pytest.mark.parametrize('kind',('translation','arc'))
def test_archived_default_bytes(kind,tmp_path):
    from docs.reference_cases.ge_beam3_preserved_arch_load_comparison import parse
    status=parse(Path('docs/reference_cases/ge_beam3_native_'+kind+'_status.json').read_bytes());archive=Path(status['archive']['path'])
    manifest=(archive/'archive-manifest.json').read_bytes()
    assert sha256(manifest).hexdigest().upper()==status['archive']['manifest_sha256']
    matches=[r for r in parse(manifest)['files'] if r['path'].startswith('runs/cycle-a-local/') and r['path'].endswith('/checkpoint.json')]
    assert len(matches)==1;row=matches[0];raw=(archive/row['path']).read_bytes()
    assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
    bar,_,codec,_=factory(kind);m,p=bar();chain,records=codec.decode_checkpoint(m,p,raw,expected_sha256=row['sha256'])
    assert codec.encode_checkpoint(m,p,chain,records)==raw and 'history_profile' not in p.descriptor()
    with pytest.raises(ValueError):codec.decode_checkpoint(m,replace(p,history_profile=capacity.HISTORY8M),raw,expected_sha256=row['sha256'])
    save(tmp_path/'compatibility.json',dict(programme=kind,original_checkpoint_sha256=row['sha256'],default_byte_identical=True,successor_rejects_default=True))
