"""Six frozen exact affine recovery nodes; reviewed bounded runner only."""
import ast
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'docs/reference_cases'
sys.path.insert(0,str(REFERENCE))
from ge_beam3_q4_exact_field import Field
import ge_beam3_q4_affine_recovery_producer as producer

SCIENTIFIC_RECORDS=[]
FIXTURES=('AFFINE_Q4_SQUARE','AFFINE_Q4_RECTANGLE','AFFINE_Q4_RHOMBUS')
SOURCE_BINDINGS={
 'src/anyfesolver/e4_pl_element.py':'7fc46a18046e043512a5fb2c3f61ed0b95b834e4dde764f63c500a885e87ea38',
 'src/anyfesolver/elements.py':'f8c59792947a3c9b84416c61a4d9db98e42926d1bf21e74e736afbf6a9a88b37',
 'src/anyfesolver/_ge_beam3_g3c_local_shell.py':'69d9f27a96ed7e9a5959a42a081eeafbbe72021de6e4da1f91355c5fb3850f10',
 'docs/GE_BEAM3_G3C_MATRIX_SHELL_CONTRACT.md':'59a09dc7db6256a145520ae7b4f2d325dee1abca5d5b5c381a7ada57a2ad92fd',
 'docs/GE_BEAM3_G3C_MO16_SOURCE_ASSESSMENT.md':'e6b702fba5c32ef22537a137fc150fa54b63028c6910df96c5c5f0c6a722026e',
 'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json':'d5fecc80c3203ac7d191b4031d64bf35c56d2900066cb01fc1dfa4276d24c006',
 'src/anyfesolver/_ge_beam3_g3c_definition.py':'4b46b870fec010e3378df83c374d763f858d8f25820c14f9704dc6f0a656f0c2'}


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')


def strict(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate key')
            result[k]=v
        return result
    def bad(value):raise ValueError('nonfinite '+value)
    result=json.loads(raw.decode('ascii'),object_pairs_hook=pairs,parse_constant=bad)
    if canonical(result)!=raw:raise ValueError('noncanonical JSON')
    return result


def read(path):
    for item in (path,*path.parents):
        info=item.lstat()
        if getattr(info,'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT:raise ValueError('reparse evidence')
    if not path.is_file():raise ValueError('regular evidence required')
    return path.read_bytes()


def write(path,value):
    raw=canonical(value)
    with path.open('xb') as stream:stream.write(raw)
    if read(path)!=raw:raise ValueError('exclusive bytes differ')
    return raw


def test_affine_exact_arithmetic_and_schema():
    roots=[Field.root(p) for p in (2,3,5)]
    for r,p in zip(roots,(2,3,5)):
        assert r*r==p and r*r.inverse()==r.inverse()*r==1
    a=1+roots[0]+roots[1]*roots[2];b=2-roots[2];c=Fraction(3,7)+roots[1]
    assert a*b==b*a and (a*b)*c==a*(b*c) and (a+b)*c==a*c+b*c
    assert a*a.inverse()==a.inverse()*a==1 and a-a==0
    assert strict(canonical({'coefficients':a.coefficients()}))=={'coefficients':a.coefficients()}
    for raw in (b'{"a":1,"a":2}\n',b'{"a":NaN}\n',b'{"a":Infinity}\n',b'{"a":1e999}\n',b'{ "a":1}\n'):
        with pytest.raises(ValueError):strict(raw)
    SCIENTIFIC_RECORDS.append(dict(test='affine_exact_arithmetic_and_schema',exact_arithmetic=True,negative_json_cases=5))


def test_affine_source_and_representation_boundaries():
    for path,digest in SOURCE_BINDINGS.items():
        raw=read(ROOT/path).replace(b'\r\n',b'\n')
        assert sha256(raw).hexdigest()==digest and sha256(raw+b'\n').hexdigest()!=digest
    permitted={'fractions','itertools','math','json','ge_beam3_q4_exact_field',
               'ge_beam3_q4_recovery_coefficient_producer','ge_beam3_q4_recovery_coefficient_checker'}
    for name in ('ge_beam3_q4_affine_recovery_producer.py','ge_beam3_q4_affine_recovery_checker.py'):
        tree=ast.parse(read(REFERENCE/name))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(a.name in permitted for a in node.names)
            if isinstance(node,ast.ImportFrom):assert node.level==0 and node.module in permitted
            if isinstance(node,ast.Constant):assert not isinstance(node.value,float)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                assert node.func.id not in ('eval','exec','compile','__import__','float')
        imports={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        imports|={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        if 'checker' in name:assert 'ge_beam3_q4_recovery_coefficient_producer' not in imports
        else:assert 'ge_beam3_q4_recovery_coefficient_checker' not in imports
    SCIENTIFIC_RECORDS.append(dict(test='affine_source_and_representation_boundaries',source_bindings=7,
        producer_checker_independent_imports=True,representation_id='GE_BEAM3_Q4_AFFINE_STATION_RECOVERY_64_V1',
        physical_recovery_qualified=False,full_g3c_qualified=False))


def fixture_proof(fixture):
    value=os.environ.get('BEAM_QUALIFICATION_OUTPUT')
    if not value or not Path(value).is_absolute():raise ValueError('bounded output required')
    out=Path(value);lease=strict(read(out/'lease.json'))
    if lease['gate']!='q4-affine-exact' or lease['lane']!='core':raise ValueError('affine gate required')
    target=out/fixture;target.mkdir(exist_ok=False)
    proof=producer.audit(fixture,lambda x:print('AFFINE '+fixture+' '+x,flush=True))
    raw=write(target/'proof.json',proof);digest=sha256(raw).hexdigest()
    processes=[];streams=[]
    try:
        for replica in ('1','2'):
            stream=(target/('checker'+replica+'.log')).open('xb');streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-I','-S','-B','-u',
                str(ROOT/'scripts/run_ge_beam3_qualification.py'),'--q4-affine-checker',str(out),fixture,replica,digest],
                cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+600
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise ValueError('affine checker failed')
            if time.monotonic()>=deadline:raise TimeoutError('affine checker wait ceiling')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise ValueError('checker nonzero exit')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=10)
        for stream in streams:stream.close()
    first=read(target/'checker1.json');second=read(target/'checker2.json')
    if first!=second:raise ValueError('checker replica disagreement')
    verification=dict(fixture_id=fixture,coefficient_count=7125,nonzero_count=proof['nonzero_count'],
        first_nonzero=proof['first_nonzero'],independently_verified=True,physical_recovery_qualified=False,full_g3c_qualified=False)
    expected=dict(schema='GE_BEAM3_Q4_AFFINE_CHECKER_RESULT_V1',candidate=lease['candidate'],
        inputs_sha256=sha256(canonical(lease['inputs'])).hexdigest(),proof_sha256=digest,verification=verification)
    if canonical(strict(first))!=canonical(expected) or read(target/'proof.json')!=raw:raise ValueError('checker/proof authority')
    SCIENTIFIC_RECORDS.append(dict(test=fixture,proof=proof,verification=verification,checker_replicas_byte_identical=True))


def test_affine_square_chart_polynomial():fixture_proof(FIXTURES[0])
def test_affine_rectangle_chart_polynomial():fixture_proof(FIXTURES[1])
def test_affine_rhombus_chart_polynomial():fixture_proof(FIXTURES[2])


def test_affine_stationary_schur_and_mutations(monkeypatch):
    from ge_beam3_q4_affine_recovery_checker import reconstruct,verify_against_reconstruction
    def forbidden_producer_recomputation(*args,**kwargs):
        raise AssertionError('mutation node must not rerun producer audit or solve')
    # These function bindings are producer-owned; the independent checker keeps
    # its own reconstruction/solver. Pytest restores both at node teardown.
    monkeypatch.setattr(producer,'audit',forbidden_producer_recomputation)
    monkeypatch.setattr(producer,'solve',forbidden_producer_recomputation)
    rows=[];out=Path(os.environ['BEAM_QUALIFICATION_OUTPUT'])
    for fixture in FIXTURES:
        expected=canonical(reconstruct(fixture,lambda s:print('AFFINE mutation '+fixture+' '+s,flush=True)))
        expected_sha=sha256(expected).hexdigest()
        original=read(out/fixture/'proof.json')
        verify_against_reconstruction(strict(original),strict(expected))
        results=[]
        def verify_bound(raw,digest):
            if sha256(raw).hexdigest()!=digest:raise ValueError('bound hash mismatch')
            return verify_against_reconstruction(strict(raw),strict(expected))
        def increment(value):value[0]=str(Fraction(value[0])+1)
        # Prepared context comes only from the already verified immutable proof.
        # Never solve the original 35-variable system or regenerate the full
        # coefficient inventory in a corruption subcase.
        baseline=strict(original)
        def decode(value):
            if type(value)is list and len(value)==8 and all(type(x)is str for x in value):
                return Field.from_coefficients(value)
            return [decode(v) for v in value]
        def encode(value):
            if isinstance(value,Field):return value.coefficients()
            return [encode(v) for v in value]
        nodes=decode(baseline['nodes']);mixed=decode(baseline['frames']['mixed'])
        centre=decode(baseline['frames']['inherited']);C=decode(baseline['constitutive'])
        Z=decode(baseline['nullspace']);station=baseline['stations'][0]
        r,s=decode(station['natural']);weight=decode(station['weight'])
        M=decode(station['mixed_strain_map']);n=decode(station['nonlinear_map'])
        inverse=decode(station['internal_inverse'])
        local=producer._local_coordinates(nodes,mixed);inherited=producer._local_coordinates(nodes,centre)
        rotation=producer.matmul(producer.transpose(mixed),centre)
        transform=producer._engineering_transform(rotation)
        _,dx,dy,_,_=producer._geometry(inherited,r,s)
        constraints,rref,pivots,prepared_Z=producer._constraints(nodes,centre)
        block,prepared_inverse,coupling,solution=producer._station_saddle(C,weight)
        sigma,epsilon=producer._source_fields(local,r,s)
        prepared_M=[[-v for v in row] for row in producer.matmul(epsilon,
            decode(baseline['stationary_solution'])[14:])]
        prepared_n=producer.matmul(transform,producer._nonlinear_coefficients(dx,dy))+producer.zeros(5,10)
        actual_mixed,actual_centre=producer._frames(nodes)
        # Positive primitive checks precede every negative; their references
        # come from a complete independent reconstruction, not a cached producer.
        for actual,want in ((actual_mixed,mixed),(actual_centre,centre),(prepared_M,M),(prepared_n,n),
            (constraints,decode(baseline['constraints'])),(rref,decode(baseline['rref'])),(prepared_Z,Z),
            (block,decode(station['internal_block'])),(prepared_inverse,inverse)):
            assert encode(actual)==encode(want)
        assert pivots==baseline['pivots']
        assert producer._geometry(local,r,s)[3]==weight

        def compare_component(name,path,value):
            altered=strict(original);target=altered
            for key in path[:-1]:target=target[key]
            target[path[-1]]=encode(value)
            with pytest.raises(ValueError,match='independent affine proof mismatch'):
                verify_against_reconstruction(altered,strict(expected))
            results.append(dict(id=name,rejection='INDEPENDENT_COMPONENT_CHECKER'))
            assert sha256(expected).hexdigest()==expected_sha
        changed_nodes=decode(baseline['nodes'])
        for node in changed_nodes:
            node[0]*=2;node[1]*=2
        # Exercise changed geometry through the real frame/coordinate primitive;
        # the exact registered node authority is independently compared afterward.
        changed_frame,_=producer._frames(changed_nodes)
        producer._local_coordinates(changed_nodes,changed_frame)
        compare_component('geometry',['nodes'],changed_nodes)
        rotated=[[row[1],-row[0],row[2]] for row in actual_mixed]
        compare_component('frame',['frames','mixed'],rotated)
        compare_component('weight',['stations',0,'weight'],2*producer._geometry(local,r,s)[3])
        changed_M=[row[:] for row in prepared_M];changed_M[0][0]+=1
        compare_component('M',['stations',0,'mixed_strain_map'],changed_M)
        changed_n=producer.matmul(transform,[[2*v for v in row] for row in producer._nonlinear_coefficients(dx,dy)])+producer.zeros(5,10)
        compare_component('n',['stations',0,'nonlinear_map'],changed_n)
        constraints[0][0]+=1
        compare_component('constraint',['constraints'],constraints)
        prepared_Z[0][0]+=1
        compare_component('Z',['nullspace'],prepared_Z)
        changed_coupling=[[-v for v in row] for row in coupling]
        with pytest.raises(ArithmeticError,match='targeted station equilibrium'):
            producer._require_equal(producer.matmul(block,solution),
                [[-v for v in row] for row in changed_coupling],'targeted station equilibrium')
        results.append(dict(id='station_coupling_sign',rejection='STATION_EQUILIBRIUM_INVARIANT'))
        changed_inverse=[row[:] for row in prepared_inverse];changed_inverse[0][0]+=1
        identity=[[Field(int(i==j)) for j in range(16)] for i in range(16)]
        with pytest.raises(ArithmeticError,match='targeted station inverse'):
            producer._require_equal(producer.matmul(block,changed_inverse),identity,'targeted station inverse')
        compare_component('inverse',['stations',0,'internal_inverse'],changed_inverse)
        # Prepare one station's equilibrium/work polynomials without a solve.
        # Select a genuinely nonzero force-weighted Hessian entry and check it
        # normally before removing that term from the same actual calculation.
        context=producer._nonlinear_station_context(M,n,C,weight,Z,inverse)
        selected=None
        for j in range(18):
            for k in range(18):
                if any(producer._poly_derivative(context['J'][a][j],k) for a in range(8)):
                    _,geometric=producer._nonlinear_hessian_terms(context,j,k)
                    if geometric:selected=(j,k);break
            if selected is not None:break
        assert selected is not None
        producer._verify_nonlinear_hessian_entry(context,*selected)
        called=[]
        def omit_geometric(material,geometric):called.append(bool(geometric));return material.copy()
        with monkeypatch.context() as patch:
            patch.setattr(producer,'_combine_schur_terms',omit_geometric)
            with pytest.raises(ArithmeticError,match='nonlinear external Schur coefficient|energy second derivative coefficient'):
                producer._verify_nonlinear_hessian_entry(context,*selected)
        assert any(called)
        results.append(dict(id='nonlinear_geometric_term',rejection='NONLINEAR_SCHUR_INVARIANT'))
        for name in ('coefficient','identity','bound_hash'):
            altered=strict(original)
            if name=='coefficient':increment(altered['coefficient_records'][0]['coefficient'])
            elif name=='identity':altered['representation_id']='OLD_RETAINED_35_VARIABLE_SYSTEM'
            raw=canonical(altered);digest=sha256(raw).hexdigest()
            if name=='bound_hash':digest=('0' if digest[0]!='0' else '1')+digest[1:]
            with pytest.raises(ValueError):verify_bound(raw,digest)
            results.append(dict(id=name,rejection='BOUND_HASH' if name=='bound_hash' else 'INDEPENDENT_CHECKER'))
        # Every registered dimension above plus the additional actual geometric-term omission.
        assert len(results)==13 and len({r['id'] for r in results})==13
        assert read(out/fixture/'proof.json')==original and sha256(expected).hexdigest()==expected_sha
        rows.append(dict(fixture_id=fixture,independent_baseline_sha256=expected_sha,baseline_immutable=True,
                         registered_mutation_dimensions=12,additional_geometric_term_omission=True,
                         full_producer_audits_in_mutations=0,prepared_contexts=1,mutations=results))
    SCIENTIFIC_RECORDS.append(dict(test='affine_stationary_schur_and_mutations',fixtures=rows,
        independent_reconstructions=3,physical_recovery_qualified=False,full_g3c_qualified=False))
