"""Static preservation and scope checks for the shared-kinematic successor."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).parents[1]
RECORD='docs/reference_cases/ge_beam3_common_kinematic_chain_evidence.json'
EXTRA={RECORD,'docs/agent_plans/GE_BEAM3_COMMON_KINEMATIC_CHAIN_CHECKPOINT.md',
       'tests/test_ge_beam3_common_kinematic_chain_checkpoint.py'}


def parse(raw):
    def unique(rows):
        value={}
        for k,v in rows:
            if k in value:raise ValueError('duplicate')
            value[k]=v
        return value
    def reject(value):raise ValueError('nonfinite')
    value=json.loads(raw,object_pairs_hook=unique,parse_constant=reject)
    if raw!=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii'):
        raise ValueError('noncanonical')
    return value


def read():return parse((ROOT/RECORD).read_bytes().replace(b'\r\n',b'\n'))


@pytest.mark.parametrize('raw',[b'{"x":0,"x":1}\n',b'{"x":NaN}\n',b'{"x": Infinity}\n'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):parse(raw)


def test_development_success_is_not_full_qualification_or_a_wheel_pass():
    p=read()
    assert p['status']=='DEVELOPMENT_CHECKS_PASS_FACTOR_CHAIN_QUALIFICATION_INCOMPLETE'
    assert p['production_qualified'] is p['public_routing_changed'] is p['release_authorized'] is False
    assert p['independent_review']=='PENDING' and p['formal_resource_requests_consumed']==[]
    assert p['installed_successor_tested'] is False
    assert p['preserved_package_failure_commit']=='9e2418020d09b43098da80c94b73b66a8029436d'
    assert p['covariance_rtol']==p['covariance_atol']==1e-11
    assert p['nonlinear_controller_replaced'] is False


def test_ten_sources_match_normalized_bytes():
    rows=read()['sources']; assert len(rows)==len({r['path'] for r in rows})==10
    for row in rows:
        raw=(ROOT/row['path']).read_bytes().replace(b'\r\n',b'\n')
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']


def test_runs_and_deterministic_pair_hashes_are_bound_to_actual_archive_entries():
    p=read(); archive=p['archive']; rows=archive['files']; index={r['path']:r for r in rows}
    assert len(index)==len(rows)==archive['file_count']
    assert sum(r['bytes'] for r in rows)==archive['bytes']
    assert archive['ordinary_context_hash_verified'] is True
    assert len(p['runs'])==2 and len(p['deterministic_json_pairs'])==28
    assert sum(r['passed'] for r in p['inventories'])==47
    for run in p['runs']:
        assert (run['passed'],run['failed'],run['errors'],run['skipped'])==(47,0,0,0)
        assert run['junit'] in index
        for pair in p['deterministic_json_pairs']:
            bound=index[run['archive_prefix']+'/'+pair['path']]
            assert (bound['bytes'],bound['sha256'])==(pair['bytes'],pair['sha256'])
    assert any(r['path'].endswith('unrepresentable-mode-positive-test.py') for r in rows)
    for row in rows:
        assert row['bytes']>=0 and len(row['sha256'])==64
        assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts


def test_additive_extent_leaves_every_old_tracked_source_unchanged():
    p=read()
    def git(*args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=True,timeout=30).stdout
    assert git('show','-s','--format=%T',p['base_commit']).strip()==p['base_tree']
    rows=[r.split('\t') for r in git('diff','--name-status',p['base_commit'],'--').splitlines()]
    assert all(kind=='A' for kind,_ in rows)
    paths={path for _,path in rows}|set(git('ls-files','--others','--exclude-standard').splitlines())
    assert paths=={r['path'] for r in p['sources']}|EXTRA


def test_same_author_decimal_audit_has_no_production_or_numerical_library_imports():
    tree=ast.parse((ROOT/'docs/reference_cases/ge_beam3_decimal_chain_audit.py').read_text())
    assert {n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}=={'decimal','time'}
    assert not any(isinstance(n,ast.Import) for n in ast.walk(tree))


def test_split_reuses_preserved_derivative_and_section_building_blocks():
    tree=ast.parse((ROOT/'src/anyfesolver/_ge_beam3_shared_kinematic_split.py').read_text())
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imports=={'anyfesolver._ge_beam3_mixed_ad','anyfesolver._ge_beam3_p5.compensated',
        'anyfesolver._ge_beam3_centered_mixed','anyfesolver._ge_beam3_p5.algebra'}
    assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(tree))


def test_private_result_types_retain_signs_and_reject_qualification_upgrade():
    from anyfesolver._native_factor_chain_modes import FactorChainModes
    from anyfesolver._ge_beam3_factor_chain_modes import SignedLoadedModes
    for cls in (FactorChainModes,SignedLoadedModes):
        assert cls.__dataclass_fields__['negative_eigenvalues_retained'].default is True
        assert cls.__dataclass_fields__['production_qualified'].default is False
        assert cls.__dataclass_fields__['certified_intervals'].default is False
    assert SignedLoadedModes.__dataclass_fields__['buckling_factor_authorized'].default is False
