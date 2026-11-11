"""Static preservation checks for an explicitly incomplete development gate."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).parents[1]
RECORD='docs/reference_cases/ge_beam3_seeded_development_evidence.json'
EXTRA={RECORD,'docs/agent_plans/GE_BEAM3_SEEDED_DEVELOPMENT_CHECKPOINT.md',
       'tests/test_ge_beam3_seeded_development_checkpoint.py'}


def unique(pairs):
    value={}
    for k,v in pairs:
        if k in value: raise ValueError('duplicate key')
        value[k]=v
    return value


def read():
    raw=(ROOT/RECORD).read_bytes().replace(b'\r\n',b'\n')
    value=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
    assert raw==(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')
    return value


def test_checkpoint_cannot_be_read_as_qualification():
    p=read()
    assert p['status']=='DEVELOPMENT_INCOMPLETE_CURVED_SPECTRAL_COVARIANCE'
    assert p['production_qualified'] is False and p['public_routing_changed'] is False
    assert p['independent_review']=='PENDING' and p['formal_resource_requests_consumed']==[]
    assert p['tests']['checkpoint']==dict(total=66,passed=65,failed=1,errors=0,junit='checkpoint-junit.xml')
    assert p['unresolved']['rtol']==p['unresolved']['atol']==1e-11
    assert sum(x['passed'] for x in p['tests']['inventories'])==65
    assert sum(x['failed'] for x in p['tests']['inventories'])==1


def test_current_source_bytes_match_twenty_bound_sources():
    p=read(); assert len(p['sources'])==20
    paths=[x['path'] for x in p['sources']]; assert len(set(paths))==len(paths)
    for row in p['sources']:
        data=(ROOT/row['path']).read_bytes().replace(b'\r\n',b'\n')
        assert len(data)==row['bytes'],row['path']
        assert hashlib.sha256(data).hexdigest()==row['sha256'],row['path']


def test_archive_inventory_preserves_failures_and_actual_junit():
    p=read(); archive=p['archive']; rows=archive['artifacts']
    assert len(rows)==archive['file_count']==74
    assert sum(r['bytes'] for r in rows)==archive['bytes']==1746933
    assert len({r['path'] for r in rows})==len(rows)
    for row in rows:
        assert row['bytes']>0 and len(row['sha256'])==64
        assert '..' not in Path(row['path']).parts and not Path(row['path']).is_absolute()
    assert any(r['path']=='checkpoint-junit.xml' for r in rows)
    assert any('failed-residual-only-program.py' in r['path'] for r in rows)
    assert any('failed-full-q-kinetic-kernel.py' in r['path'] for r in rows)
    assert any(r['path'].endswith('/audit.json') for r in rows)


def test_extent_adds_private_successors_without_modifying_historical_sources():
    p=read()
    result=subprocess.run(['git','diff','--name-status',p['base_commit'],'--'],cwd=ROOT,
        check=True,capture_output=True,text=True,timeout=30)
    rows=[line.split('\t') for line in result.stdout.splitlines()]
    assert all(kind=='A' for kind,_ in rows)
    assert {path for _,path in rows}=={r['path'] for r in p['sources']}|EXTRA
    assert all(path.startswith(('src/anyfesolver/_ge_beam3_','src/anyfesolver/_native_signed_'))
        for _,path in rows if path.startswith('src/'))


def test_seeded_package_is_only_namespace_identity_and_seed_successor():
    prefix='src/anyfesolver/'
    for name in ('__init__','codec','core','element','state','work'):
        old=(ROOT/(prefix+'_ge_beam3_p5_loads/'+name+'.py')).read_text()
        expected=old.replace('_ge_beam3_p5_loads','_ge_beam3_p5_seeded').replace('_V4','_V5')
        if name=='core':
            expected=expected.replace("SCHEMA='",'from anyfesolver._ge_beam3_authoritative_cell_seed import cell_initial_rotations, POLICY as SEED_POLICY\n\nSCHEMA=\'')
            expected=expected.replace('coordinate_policy=POLICY, element=',
                'coordinate_policy=POLICY, initial_rotation_policy=SEED_POLICY, element=')
            expected=expected.replace('return model.solve(positions, operators@self.reference.nodal_triads,\n',
                'return model.solve(positions, operators@self.reference.nodal_triads,\n            initial_rotations=cell_initial_rotations(operators),\n')
        actual=(ROOT/(prefix+'_ge_beam3_p5_seeded/'+name+'.py')).read_text()
        assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(actual)),name


def test_decimal_audit_has_no_mechanical_or_scientific_library_imports():
    tree=ast.parse((ROOT/'docs/reference_cases/ge_beam3_decimal_factor_audit.py').read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(n.name for n in node.names)
        if isinstance(node,ast.ImportFrom): imports.append(node.module)
    assert set(imports)=={'decimal','time'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id in ('eval','exec','__import__') for n in ast.walk(tree))


def test_failure_gate_is_not_skipped_xfailed_or_tolerance_relaxed():
    tree=ast.parse((ROOT/'tests/test_ge_beam3_curved_contrast_probe.py').read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('skip','skipif','xfail') for n in ast.walk(tree))
    comparisons=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='assert_allclose']
    assert len(comparisons)==2
    for call in comparisons:
        assert {k.arg:k.value.value for k in call.keywords}=={'rtol':1e-11,'atol':1e-11}
