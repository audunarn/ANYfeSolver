"""Inert runner guard tests: fake jobs only, never import mechanics."""
import ast
import copy
from hashlib import sha256
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_ge_beam3_qualification as r

class Clock:
    def __init__(self):self.value=0
    def __call__(self):return self.value
    def sleep(self,value):self.value+=value

class Job:
    def __init__(self,memory=0):self.active=1;self.mem=0;self.killed=False;self.closed=False;self.code=None;self.cpu=0
    def accounting(self):return self.cpu,self.active,self.mem
    def terminate(self):self.killed=True;self.active=0;self.code=124;return True
    def close(self):self.closed=True
    def poll(self):return self.code
    def launch(self,*a,**kw):return self


def inert_zeros(shape):
    return [inert_zeros(shape[1:]) for _ in range(shape[0])] if shape else 0.


def inert_row(table,identity):
    """Schema fixtures only: deliberately NOT mechanics or accepted evidence."""
    row={'id':identity};digest=lambda shape:r.numeric_array(inert_zeros(shape),shape)
    if table in r.PHYSICAL_TABLES:
        state=dict(construction_id=r.row_construction(table,identity),coordinates=inert_zeros([4,3]),q=inert_zeros([24]),
            accepted=inert_zeros([4,3,3]),normal=[0.,0.,1.],material_direction=[1.,0.,0.],director_polarity=-1 if identity.endswith('::DIRECTOR:-1') else 1)
        row.update(state=state,state_sha256=sha256(r.canonical(state)).hexdigest(),physical_checks={})
        checks=dict(symmetry=0.,spatial_force=0.,spatial_tangent=0.)
        mapping={'energy':'physical_energy','force':'physical_force','hessian':'physical_hessian'}
        if table=='work' and not identity.endswith('::ZERO'):
            mapping.update(source_energy='source_physical_energy',source_force='source_physical_force',source_hessian='source_physical_hessian')
        for key,predicate in mapping.items():
            check=dict(relative_error=0.,passed=True,actual=digest(r.PHYSICAL_SHAPES[predicate]))
            checks[key]=check;row['physical_checks'][predicate]=check
        row['checks']=checks
    if table=='definitions':row.update(recipe_sha256='0'*64,descriptor_sha256='0'*64)
    elif table=='source_graph':row['hashes']=dict(r.NUMERICAL_SOURCE_HASHES)
    elif table=='extension_lemma':row.update(lemma_sha256='156d33ae5a621b953b5d04050218618f6416bac1042f94d3d3fc5c305c9f8762',review_sha256='e28f184023ce1bba825a89079bcd2f9af99cf7861d701d7d918e2d30492b073e')
    elif table=='station_join':row.update(station_association_id=r.STATION_ASSOCIATION_ID,verified=True,rejections=4)
    elif table=='chart_authority':row.update(chart_numerics_id=r.CHART_ID,addendum_sha256=r.CHART_ADDENDUM_SHA,
        review_sha256=r.CHART_REVIEW_SHA,increment_addendum_sha256=r.INCREMENT_ADDENDUM_SHA,
        increment_review_sha256=r.INCREMENT_REVIEW_SHA,sources=dict(r.CHART_SOURCES),verified=True)
    elif table=='fingerprint':row.update(distinct_fingerprints=20,nonfinite_rejections=5,evidence_sha256='0'*64,verified=True)
    elif table=='station':row.update(checks={key:0. for key in ('d','D','D2','R','Q','x')},energy=0.,stations=4)
    elif table=='independent':row['stations']=[{key:0. for key in ('M','strain','resultant','frame','constitutive')} for _ in range(4)]
    elif table=='schur':row.update(schur=digest([24,24]),full_internal_dimension=64)
    elif table=='work':row['chart_image_sentinels']=True
    elif table=='directional':row.update(work=0.,tangent=0.)
    elif table=='rigid':row.update(rigid_columns=6,total_positive_modes=18,eigenvalues=digest([24]))
    elif table in ('common_motion','passive','rebase','tiny'):row['energy']=1.
    elif table=='d4':
        k=int(identity.rsplit(':',1)[1]);row['station_map']=[(i+k%4)%4 for i in ((0,1,2,3) if k<4 else (0,3,2,1))]
    elif table=='director':row['physical_polarity']=int(identity.rsplit(':',1)[1])
    elif table=='graph':
        ids=[101,102,103,104] if identity.startswith('J_Q4_PAIR::') else [301,302,303,304]
        element=11 if identity.startswith('J_Q4_PAIR::') else 13
        if '::RENUMBERED::' in identity:ids=[10000+7*i for i in ids];element=20000+5*element
        if '::CONNECTIVITY_REVERSED::' in identity:ids=[ids[i] for i in (0,3,2,1)]
        row.update(node_ids=ids,element_id=element,recipe_sha256='0'*64)
    elif table=='channels':row.update(physical=0.,numerical=[dict(name=n,energy=0.) for n in ('NUMERICAL_PL','NUMERICAL_HOURGLASS')])
    elif table=='races':row['rejected_before_family']=True
    elif table=='immutability':
        row.update(verified=True,prior_arrays_sha256='0'*64)
        if identity=='all_detached_arrays':row['array_count']=31
        if identity=='reentry':row['rejections']=2
        if identity=='nested_candidate_bytes':row['fingerprint_sha256']='0'*64
    elif table=='rejections':
        if identity in ('before_work','before_publication','invalid_callback'):
            row.update(callbacks=2 if identity=='before_publication' else 1,family_entries=1 if identity=='before_publication' else 0,published=False)
        else:row['rejected_before_family']=True
    elif table=='eigen_derivatives':row.update(derivative_coordinates=24,derivative_pairs=576,
        derivative_residuals={key:0. for key in ('first_normalization','second_normalization','first_stationarity','second_stationarity','second_symmetry')})
    elif table=='chart_mutations':
        compared=identity in ('missing_delta_covariance','naive_reference_subtraction','omitted_delta_derivatives')
        row['rejection']=('INDEPENDENT_CHART_COMPARISON' if compared else 'BRANCH_REJECTION' if identity in ('wrong_davenport_branch','wrong_polar_branch')
            else 'DERIVATIVE_IDENTITY_REJECTION' if identity in ('eigen_first_derivative','eigen_second_derivative') else 'NONCONVERGENCE_REJECTION')
        if compared:row['relative_error']=1e-8
    elif table=='mutations':
        mutation=identity.split('::',1)[1]
        row['rejection']=('INDEPENDENT_HESSIAN' if mutation in ('force_weighted_Hessian','chart_second') else
            'STATION_EQUILIBRIUM' if mutation=='coupling_sign' else 'STATION_INVERSE' if mutation=='inverse' else
            'MATERIAL_ENERGY' if mutation=='numerical_energy_leak' else 'INDEPENDENT_STATION_COMPARISON')
    if table=='tiny':row['chart']={key:0. for key in ('d','D','D2','R','Q','x')}
    return row


def inert_records(index):
    return [dict(test=r.NUMERICAL_TESTS[index].removeprefix('test_affine_recovery_'),
        tables={table:[inert_row(table,identity) for identity in ids] for table,ids in r.NUMERICAL_TABLE_IDS[index].items()},
        full_g3c_qualified=False,production_qualified=False)]


def inert_observations(node,records):
    return {(node,table,row['id']):row['state_sha256'] for table,rows in records[0]['tables'].items()
            if table in r.PHYSICAL_TABLES for row in rows}


def inert_contradiction(node,table,row,lease,predicate='physical_energy'):
    """Protocol-only synthetic witness, never passed to the physics checker."""
    state=row['state'];actual=inert_zeros(r.PHYSICAL_SHAPES[predicate])
    error=.5;fingerprint=r.numeric_array(actual,r.PHYSICAL_SHAPES[predicate])
    check=dict(relative_error=error,passed=False,actual=fingerprint)
    row['physical_checks'][predicate]=check
    key={'physical_energy':'energy','physical_force':'force','physical_hessian':'hessian',
         'source_physical_energy':'source_energy','source_physical_force':'source_force','source_physical_hessian':'source_hessian'}[predicate]
    row['checks'][key]=check
    if predicate.startswith('source_'):row['chart_image_sentinels']=False
    payload=dict(schema='Q4_AFFINE_NUMERICAL_CONTRADICTION_V1',predicate=predicate,actual=actual,tolerance=1e-11,
        scale_mode='REFERENCE_EDGE_NONDIMENSIONAL_V1',source_identity=sha256(r.canonical(lease['inputs'])).hexdigest(),
        candidate_identity=lease['candidate']['commit'],fixture_identity=state['construction_id'],
        node=node,table=table,row_id=row['id'],state_sha256=row['state_sha256'],
        **{key:state[key] for key in ('coordinates','q','accepted','normal','material_direction','director_polarity')})
    verification=dict(accepted=True,predicate=predicate,relative_error=error,node=node,table=table,row_id=row['id'],
        fixture_identity=state['construction_id'],payload_sha256=sha256(r.canonical(payload)).hexdigest(),
        state_sha256=row['state_sha256'],actual=fingerprint,expected=dict(fingerprint,sha256='1'*64))
    return dict(payload=payload,verification=verification)

class GuardTests(unittest.TestCase):
    def test_registered_inventory(self):
        self.assertEqual(len(r.inventory('core')),8);self.assertEqual(len(r.inventory('smoke')),2)
        self.assertEqual(len(r.inventory('core','b2-adapter')),3)
        self.assertEqual(len(r.inventory('smoke','b2-adapter')),1)
        self.assertEqual(len(r.inventory('core','q4-audit')),5)
        self.assertEqual(len(r.inventory('smoke','q4-audit')),2)
        self.assertEqual(len(r.inventory('core','q4-affine-exact')),6)
        self.assertEqual(len(r.inventory('smoke','q4-affine-exact')),2)
        self.assertEqual(len(r.inventory('core','q4-affine-numerical')),15)
        self.assertEqual([p.split('::')[-1] for p in r.inventory('smoke','q4-affine-numerical')],
                         [r.NUMERICAL_TESTS[0],r.NUMERICAL_SMOKE])
        self.assertEqual(len(r.inventory('smoke','g4a-station')),1)
        self.assertEqual(len(r.inventory('core','g4a-station')),13)
        self.assertEqual(r.inventory('core','g4a-station'),r.inventory('formal','g4a-station'))
        self.assertEqual(len(r.inventory('smoke','g4-completion')),1)
        self.assertEqual(len(r.inventory('core','g4-completion')),26)
        self.assertEqual(r.inventory('core','g4-completion'),r.inventory('formal','g4-completion'))
        self.assertEqual(len(r.inventory('smoke','g5-completion')),1)
        self.assertEqual(len(r.inventory('core','g5-completion')),7)
        self.assertEqual(r.inventory('core','g5-completion'),r.inventory('formal','g5-completion'))
        with self.assertRaises(ValueError):r.inventory('all')

    def test_g4a_adjudication_is_scoped_and_mutation_detecting(self):
        row=dict(test='G4A_HETEROGENEOUS_SIX_STAGE_TRANSACTION',
            station_families=['EXACT_ELASTIC','GENERALIZED_ELLIPSOID','PHYSICAL_FIBRE'],
            accepted_epochs=6,independent_replay=True,atomic_publication=True,
            physical_fibre_recovery=True,snapshot_sha256='a'*64,production_qualified=False)
        self.assertEqual(r.g4a_adjudication([row],'formal'),{
            'terminal':'PROVISIONAL_GO_GE_BEAM3_G4A_STATION_TRANSACTION_ONLY',
            'g4a_qualified':True,'g4_qualified':False,'production_qualified':False})
        self.assertEqual(r.g4a_adjudication([row],'smoke')['terminal'],'NOT_ADJUDICATED_SMOKE_ONLY')
        for key,value in (('accepted_epochs',5),('independent_replay',False),
                          ('snapshot_sha256','bad'),('production_qualified',True)):
            changed=copy.deepcopy(row);changed[key]=value
            with self.assertRaises(ValueError):r.g4a_adjudication([changed],'formal')

    def test_g4_completion_adjudication_closes_only_s19_through_s24(self):
        row=dict(test='G4_COMPLETE_MATERIAL_STATE_CONTROL_AND_RESTART',accepted_epochs=6,
            station_families=['EXACT_ELASTIC','GENERALIZED_ELLIPSOID','PHYSICAL_FIBRE'],
            mixed_atomic_publication=True,generalized_connected_history=True,
            fibre_connected_history=True,force_control=True,displacement_control=True,
            arc_control=True,load_unload_reversal=True,shell_joint_history=True,
            accepted_origin_replay=True,authenticated_restart=True,physical_recovery=True,
            resultant_only_no_invented_fibres=True,checkpoint_sha256='a'*64,
            production_qualified=False)
        assert r.g4_adjudication([row],'formal')==dict(
            terminal='PROVISIONAL_GO_GE_BEAM3_G4_GENERAL_STATIC_MATERIAL_STATE_ONLY',
            closed_rows=['S19','S20','S21','S22','S23','S24'],g4_qualified=True,
            g5_qualified=False,production_qualified=False)
        assert r.g4_adjudication([row],'smoke')['terminal']=='NOT_ADJUDICATED_SMOKE_ONLY'
        for key,value in (('arc_control',False),('accepted_epochs',5),
                          ('checkpoint_sha256','bad'),('production_qualified',True)):
            changed=copy.deepcopy(row);changed[key]=value
            with self.assertRaises(ValueError):r.g4_adjudication([changed],'formal')

    def test_g5_adjudication_requires_all_ordered_actual_routes(self):
        routes=('LEGACY_B2_REFERENCE_ANALYSIS','LEGACY_B3_REFERENCE_ANALYSIS',
            'GE_REFERENCE_LOAD_INERTIA','GE_REFERENCE_MODAL_BUCKLING_TRANSIENT',
            'GE_NATIVE_STATIC_RESTART_RECOVERY','UNSUPPORTED_ROUTES_FAIL_CLOSED',
            'SELECTOR_AND_PROVENANCE_BOUNDARY')
        rows=[]
        for route in routes:
            body=dict(policy='GE_BEAM3_G5_GENERAL_STATIC_INTEGRATION_V1',route=route,
                checks={'passed':True},production_qualified=False)
            rows.append({**body,'record_sha256':sha256(r.canonical(body)).hexdigest()})
        result=r.g5_adjudication(rows,'formal')
        self.assertEqual(result['terminal'],'FUNCTIONAL_GE_BEAM3_G5_READY_FOR_INSTALLED_ARTIFACT')
        self.assertEqual(result['closed_rows'],['S25'])
        self.assertFalse(result['g5_qualified'])
        self.assertEqual(r.g5_adjudication(rows[:1],'smoke')['terminal'],'NOT_ADJUDICATED_SMOKE_ONLY')
        for changed in (rows[::-1],rows[:-1]):
            with self.assertRaises(ValueError):r.g5_adjudication(changed,'formal')

    def test_physical_registered_inventories_are_exact_and_inert(self):
        inventories={lane:r.inventory(lane,'g3c-physical') for lane in ('local','smoke','rehearsal','formal')}
        self.assertEqual({lane:len(rows) for lane,rows in inventories.items()},
                         {'local':26,'smoke':5,'rehearsal':234,'formal':3825})
        self.assertEqual(sum(row['kind']=='history' for row in inventories['formal']),375)
        self.assertEqual(sum(row['kind']=='prefix' for row in inventories['formal']),3450)
        self.assertEqual(sum(row['kind']=='authority-mutation' for row in inventories['rehearsal']),6)
        self.assertEqual(sum(row['kind']=='mutation' for row in inventories['rehearsal']),136)
        self.assertEqual([row['case']['case_id'] for row in inventories['smoke']],
                         [graph+'::BASE::S0::NONE' for graph in r.physical_support().GRAPHS])
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('anyfesolver',sys.modules)

    def test_physical_formal_case_association_is_exact_and_complete(self):
        rows=r.physical_formal_case_associations();inventory=r.physical_inventory('formal')
        self.assertEqual(len(rows),375)
        self.assertEqual(sha256(r.canonical([row['case'] for row in rows])).hexdigest(),
                         r.PHYSICAL_FORMAL_HISTORY_INVENTORY_SHA)
        self.assertEqual(sha256(r.canonical(inventory)).hexdigest(),r.PHYSICAL_FORMAL_ASSIGNMENT_INVENTORY_SHA)
        history=[row['history_assignment_index'] for row in rows]
        prefixes=[index for row in rows for index in row['prefix_assignment_indexes']]
        self.assertEqual(history,list(range(375)))
        self.assertEqual(prefixes,list(range(375,3825)))
        self.assertEqual(sum(row['case']['accepted_stages'] for row in rows),3075)
        self.assertEqual(len(prefixes),3450)
        for row in rows:
            self.assertEqual(inventory[row['history_assignment_index']]['case'],row['case'])
            self.assertEqual([inventory[index]['prefix'] for index in row['prefix_assignment_indexes']],
                             list(range(row['case']['accepted_stages']+1)))
        for value in (-1,False,375,0.0):
            with self.assertRaises(ValueError):r.physical_formal_shard(value,'history-producer')

    def test_physical_formal_partition_requires_unique_exact_coverage(self):
        associations=r.physical_formal_case_associations()
        shards=[]
        for row in associations:
            ordinal=row['case_ordinal'];count=len(row['prefix_assignment_indexes'])
            shards.append(r.physical_formal_shard(ordinal,'history-producer'))
            shards.append(r.physical_formal_shard(ordinal,'prefix-range',0,count))
        partition=r.physical_formal_partition(shards,'formal')
        self.assertEqual(len(partition['body']['shards']),750)
        self.assertEqual(partition['self_sha256'],sha256(r.canonical(partition['body'])).hexdigest())
        for changed in (shards[:-1],shards+[shards[-1]],shards[1:]):
            with self.assertRaises(ValueError):r.physical_formal_partition(changed,'formal')
        measurement=r.physical_formal_partition([
            r.physical_formal_shard(0,'history-producer'),r.physical_formal_shard(0,'prefix-range',0,1)],'measurement')
        self.assertEqual(measurement['body']['mode'],'measurement')

    def test_physical_formal_measurement_partition_is_exact_and_nonclassifying(self):
        partition=r.physical_formal_measurement_partition();rows=partition['body']['shards']
        self.assertEqual(len(rows),12)
        self.assertEqual([row['assignment']['case_ordinal'] for row in rows],[0]*4+[224]*4+[374]*4)
        for offset in (0,4,8):
            self.assertEqual(rows[offset]['assignment']['kind'],'history-producer')
            self.assertEqual([(row['assignment']['prefix_start'],row['assignment']['prefix_stop'])
                for row in rows[offset+1:offset+4]],[(0,1),(1,3),(3,6)])
        self.assertEqual(partition['body']['mode'],'measurement')
        self.assertEqual(partition['self_sha256'],sha256(r.canonical(partition['body'])).hexdigest())

    def test_physical_formal_common_manifest_is_closed_and_mutation_detecting(self):
        shards=[r.physical_formal_shard(0,'history-producer'),r.physical_formal_shard(0,'prefix-range',0,1)]
        partition=r.physical_formal_partition(shards,'measurement')
        candidate={'commit':'a'*40,'tree':'b'*40};inputs={'x':{'bytes':1,'sha256':'c'*64}}
        review={'decision':'synthetic-inert-test'}
        value=r.physical_formal_common_manifest(candidate,inputs,review,'d'*64,partition)
        self.assertEqual(r.physical_validate_formal_common(value)['mode'],'measurement')
        mutators=(
            lambda x:x['body'].__setitem__('cycle',1),
            lambda x:x['body'].__setitem__('history_inventory_sha256','0'*64),
            lambda x:x['body']['limits'].__setitem__('workers',4),
            lambda x:x['body']['associations'][0].__setitem__('history_assignment_index',1),
            lambda x:x['body']['partition']['body']['shards'][0].__setitem__('shard_id','foreign'),
            lambda x:x.__setitem__('self_sha256','0'*64))
        for mutate in mutators:
            bad=copy.deepcopy(value);mutate(bad)
            if bad['self_sha256']==value['self_sha256']:
                bad['self_sha256']=sha256(r.canonical(bad['body'])).hexdigest()
            with self.assertRaises(ValueError):r.physical_validate_formal_common(bad)

    def test_physical_formal_lease_rejects_cross_case_packets(self):
        producer=r.physical_formal_shard(224,'history-producer')
        replay=r.physical_formal_shard(224,'prefix-range',0,1)
        foreign=r.physical_formal_shard(374,'prefix-range',0,1)
        shards=[producer,replay,foreign]
        partition=r.physical_formal_partition(shards,'measurement')
        value=r.physical_formal_common_manifest({'commit':'a'*40,'tree':'b'*40},{},{},'c'*64,partition)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r.write(root/'common.json',value)
            descriptor=r.physical_formal_manifest_descriptor(root/'common.json')
            out=root/'producer';out.mkdir()
            producer_lease=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',
                schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_LEASE_V1',run_id=str(uuid.uuid4()),
                mode='measurement',cycle=0,common_manifest=descriptor,shard_id='measurement-0000',assignment=producer,input_packets={})
            for name in r.physical_formal_expected_files(producer,process=False):
                if name=='lease.json':r.write(out/name,producer_lease)
                elif name=='producer.receipt.json':continue
                elif name.startswith('prefix-'):r.write(out/name,{'prefix':int(name[7:9])})
                else:r.write(out/name,{'name':name})
            r.write(out/'producer.receipt.json',r.physical_formal_producer_receipt(out,producer_lease,value['body']))
            files={path.name:r.fingerprint(path.read_bytes()) for path in out.iterdir() if path.is_file()}
            process=dict(status='PASSED',active_processes=0,peak_tree_bytes=0,drained=True,elapsed_seconds=0.0,
                kind='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_PROCESS_V1',shard_id='measurement-0000',
                assignment_sha256=sha256(r.canonical(producer)).hexdigest(),returncode=0,files=files)
            r.write(out/'process.json',process)
            receipt_descriptor=r.packet_descriptor(out/'producer.receipt.json')
            receipt=r.physical_validate_formal_producer_receipt(out/'producer.receipt.json',value['body'])
            packets={'prefix-00':receipt['packets']['prefix-00'],
                'final':receipt['packets'][f"prefix-{producer['case']['accepted_stages']:02d}"]}
            lease=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',
                schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_LEASE_V1',run_id=str(uuid.uuid4()),
                mode='measurement',cycle=0,common_manifest=descriptor,shard_id='measurement-0001',assignment=replay,
                input_packets=packets,producer_receipt=receipt_descriptor)
            r.physical_validate_formal_lease(lease,value['body'])
            cross_case=copy.deepcopy(lease);cross_case['run_id']=str(uuid.uuid4())
            cross_case['shard_id']='measurement-0002';cross_case['assignment']=foreign
            with self.assertRaisesRegex(ValueError,'case lineage'):
                r.physical_validate_formal_lease(cross_case,value['body'])
            for mutate in (
                lambda x:x.__setitem__('cycle',1),lambda x:x.__setitem__('cycle',False),
                lambda x:x.__setitem__('shard_id','measurement-9999'),
                lambda x:x['assignment'].__setitem__('case_ordinal',1),
                lambda x:x['input_packets'].__setitem__('prefix-01',x['input_packets'].pop('prefix-00')),
                lambda x:x.__setitem__('producer_receipt',dict(x['producer_receipt'],sha256='0'*64))):
                bad=copy.deepcopy(lease);mutate(bad)
                with self.assertRaises(ValueError):r.physical_validate_formal_lease(bad,value['body'])

    def test_physical_formal_execution_remains_locked_and_worker_import_is_guarded(self):
        args=SimpleNamespace(review=Path('review.json'),review_sha256='c'*64,lane='formal',prior=[],
            partition_id=None,finalize_partitions=False,inherit_correction=False,failed_guards_process=None,
            interrupted_guards_root=None,guard_segment_id=None,finalize_guard_segments=False)
        expected=({'commit':'a','tree':'b'},{},b'{}\n')
        with patch.object(r,'authority',return_value=expected),patch.object(r,'physical_run_phase',
                side_effect=AssertionError('formal mechanics dispatched')):
            with self.assertRaisesRegex(ValueError,'formal physical partition/authorization addendum not frozen'):
                r.execute_physical(args,SimpleNamespace())
        source=Path(r.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
        worker=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='physical_formal_worker')
        calls=[node.func.id for node in ast.walk(worker) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
        self.assertLess(calls.index('authority'),calls.index('claim_attempt'))
        authority_line=next(node.lineno for node in ast.walk(worker) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id=='authority')
        mode_guard=next(node.lineno for node in ast.walk(worker) if isinstance(node,ast.If)
            and 'formal execution authorization not frozen' in ast.get_source_segment(source,node))
        self.assertLess(mode_guard,authority_line)
        runtime_guard=next(node.lineno for node in ast.walk(worker) if isinstance(node,ast.If)
            and 'formal runtime authority' in ast.get_source_segment(source,node))
        self.assertLess(runtime_guard,authority_line)
        imports=[node for node in ast.walk(worker) if isinstance(node,(ast.Import,ast.ImportFrom))
                 and (getattr(node,'module','')or'').startswith('anyfesolver')]
        self.assertTrue(imports);self.assertTrue(all(node.lineno>authority_line for node in imports))
        formal_test=(Path(r.__file__).resolve().parents[1]/r.PHYSICAL_FORMAL_TEST).read_text(encoding='utf-8')
        formal_tree=ast.parse(formal_test);case=next(node for node in formal_tree.body
            if isinstance(node,ast.FunctionDef) and node.name=='test_physical_formal_shard_assignment')
        prefix_loop=next(node for node in ast.walk(case) if isinstance(node,ast.For)
            and isinstance(node.target,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='prefix' for x in node.target.elts))
        calls=[node for node in ast.walk(prefix_loop) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id=='common_identity']
        deletes=[node for node in ast.walk(prefix_loop) if isinstance(node,ast.Delete)
            and any(isinstance(target,ast.Name) and target.id=='owner' for target in node.targets)]
        frozen_hash=[node for node in ast.walk(prefix_loop) if isinstance(node,ast.Name)
            and node.id=='frozen_assignment_sha256']
        self.assertEqual(len(calls),2);self.assertEqual(len(deletes),1);self.assertGreaterEqual(len(frozen_hash),2)

    def test_physical_formal_measurement_batch_is_bounded(self):
        with self.assertRaises(ValueError):
            r.physical_formal_measurement_batch(Path('.'),Path('common.json'),[],b'',{},SimpleNamespace(),1e9)
        specs=[dict(shard_id=str(i)) for i in range(4)]
        with self.assertRaises(ValueError):
            r.physical_formal_measurement_batch(Path('.'),Path('common.json'),specs,b'',{},SimpleNamespace(),1e9)
        source=inspect.getsource(r.execute_physical_formal_measurement)
        self.assertNotIn("write(root/'scientific.json'",source)
        self.assertIn("classification='NONCLASSIFYING'",source)

    def test_physical_formal_process_schema_is_exact(self):
        assignment=r.physical_formal_shard(0,'prefix-range',0,1)
        lease={'shard_id':'measurement-0000','assignment':assignment}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for name in r.physical_formal_expected_files(assignment,process=False):r.write(out/name,{'name':name})
            files={path.name:r.fingerprint(path.read_bytes()) for path in out.iterdir() if path.is_file()}
            value=dict(status='PASSED',active_processes=0,peak_tree_bytes=0,drained=True,
                elapsed_seconds=0.0,kind='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_PROCESS_V1',
                shard_id=lease['shard_id'],assignment_sha256=sha256(r.canonical(assignment)).hexdigest(),
                returncode=0,files=files)
            r.write(out/'process.json',value);r.physical_validate_formal_process(out,lease)
            for field,replacement in (('active_processes',False),('elapsed_seconds',600.0),
                                      ('shard_id','foreign'),('returncode',False)):
                bad=copy.deepcopy(value);bad[field]=replacement;(out/'process.json').unlink();r.write(out/'process.json',bad)
                with self.assertRaises(ValueError):r.physical_validate_formal_process(out,lease)
            (out/'process.json').unlink();r.write(out/'process.json',value);r.write(out/'extra.json',{})
            with self.assertRaises(ValueError):r.physical_validate_formal_process(out,lease)

    def test_physical_formal_union_is_assignment_ordered_and_deterministic(self):
        producer=r.physical_formal_shard(0,'history-producer');prefix=r.physical_formal_shard(0,'prefix-range',0,1)
        partition=r.physical_formal_partition([producer,prefix],'measurement')
        common=r.physical_formal_common_manifest({'commit':'a'*40,'tree':'b'*40},{},{},'c'*64,partition)['body']
        common_value=dict(body=common,self_sha256=sha256(r.canonical(common)).hexdigest())
        common_sha=sha256(r.canonical(common_value)).hexdigest()
        case=producer['case'];history_record=dict(kind='history',case_id=case['case_id'],events=case['accepted_stages'],
            directional_errors=[0.,0.,0.],packets=[dict(name=f'prefix-{i:02d}.json',bytes=1,
                sha256='0'*64,prefix=i) for i in range(case['accepted_stages']+1)],passed=True)
        transport=dict(accepted_diagnostic_sha256='0'*64,definition_sha256='1'*64,expanded_sha256='2'*64,
            programs_sha256='3'*64,final_state_sha256='4'*64,case_id=case['case_id'],graph=case['graph'],
            variant=case['variant'],force_scale=case['force_scale'],common_motion=case['common_motion'],immutable=True,passed=True)
        prefix_record=dict(kind='prefix',case_id=case['case_id'],prefix=0,input_sha256='5'*64,final_sha256='6'*64,passed=True)
        def science(shard_id,assignment,record):
            return dict(schema='GE_BEAM3_G3C_PHYSICAL_FORMAL_SHARD_SCIENCE_V1',mode='measurement',cycle=0,
                candidate=common['candidate'],common_manifest_sha256=common_sha,shard_id=shard_id,assignment=assignment,
                record=record,full_g3c_qualified=False,production_qualified=False)
        rows=[science('measurement-0000',producer,dict(kind='history-producer',case_ordinal=0,
            history_assignment_index=0,history=history_record,transport=transport,passed=True)),
            science('measurement-0001',prefix,dict(kind='prefix-range',case_ordinal=0,prefix_start=0,prefix_stop=1,
                assignment_indexes=prefix['assignment_indexes'],records=[dict(assignment_index=375,record=prefix_record)],
                fresh_owners=1,passed=True))]
        first=r.physical_formal_scientific_union(rows,common);second=r.physical_formal_scientific_union(list(reversed(rows)),common)
        self.assertEqual(r.canonical(first),r.canonical(second));self.assertEqual([x['assignment_index'] for x in first['records']],[0,375])
        self.assertNotIn('cycle',first)
        wrong_cycle=copy.deepcopy(rows);wrong_cycle[0]['cycle']=False
        with self.assertRaises(ValueError):r.physical_formal_scientific_union(wrong_cycle,common)
        with self.assertRaises(ValueError):r.physical_formal_scientific_union(rows+[rows[0]],common)

    def test_physical_rehearsal_partition_manifest_is_exact(self):
        rows=r.physical_inventory('rehearsal');manifest=r.physical_partition_manifest(rows)
        self.assertEqual(sha256(r.canonical(rows)).hexdigest(),r.PHYSICAL_PARTITION_INVENTORY_SHA)
        self.assertEqual(sha256(r.canonical(manifest)).hexdigest(),r.PHYSICAL_PARTITION_MANIFEST_SHA)
        covered=[]
        for expected_id,(partition_id,indices,digest),actual in zip(
                r.PHYSICAL_PARTITION_IDS,r.PHYSICAL_PARTITIONS,manifest['partitions']):
            self.assertEqual(expected_id,partition_id);self.assertEqual(actual['partition_id'],partition_id)
            self.assertEqual(actual['indices'],list(indices));covered.extend(indices)
            self.assertEqual(sha256(r.canonical([rows[i] for i in indices])).hexdigest(),digest)
        self.assertEqual(covered,list(range(234)));self.assertEqual(len(covered),len(set(covered)))
        self.assertEqual(r.physical_partition_prerequisite_ids('R-GUARDS'),list(r.PHYSICAL_PARTITION_IDS[:-1]))
        with self.assertRaises(ValueError):r.physical_partition_spec('FOREIGN')

        segments=r.physical_correction_guard_segment_manifest(rows);segment_coverage=[]
        self.assertEqual(sha256(r.canonical(segments)).hexdigest(),r.PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA)
        for expected_id,(segment_id,indices,digest),actual in zip(
                r.PHYSICAL_CORRECTION_GUARD_SEGMENT_IDS,r.PHYSICAL_CORRECTION_GUARD_SEGMENTS,segments['segments']):
            self.assertEqual(expected_id,segment_id);self.assertEqual(actual['segment_id'],segment_id)
            self.assertEqual(actual['indices'],list(indices));segment_coverage.extend(indices)
            self.assertEqual(actual['assignment_sha256'],digest)
            self.assertEqual(sha256(r.canonical([rows[i] for i in indices])).hexdigest(),digest)
        self.assertEqual(segment_coverage,list(range(90,234)))
        with self.assertRaises(ValueError):r.physical_correction_guard_segment_spec('FOREIGN')

    def test_physical_rehearsal_partition_mode_is_explicit(self):
        expected=({'commit':'a','tree':'b'},{},b'{}\n');watchdog=SimpleNamespace()
        def arguments(lane,partition_id=None,finalize=False):
            return SimpleNamespace(review=Path('review.json'),review_sha256='c'*64,lane=lane,prior=[],
                                   partition_id=partition_id,finalize_partitions=finalize)
        with patch.object(r,'authority',return_value=expected):
            for args in (arguments('rehearsal'),arguments('rehearsal','R-HISTORY',True),
                         arguments('smoke','R-HISTORY'),arguments('local',finalize=True)):
                with self.assertRaises(ValueError):r.execute_physical(args,watchdog)

    def test_physical_correction_mode_is_exact_and_correction_only(self):
        expected=({'commit':'a','tree':'b'},{},b'{}\n');watchdog=SimpleNamespace()
        def arguments(partition_id=None,finalize=False,lane='rehearsal',failed=Path('failed.json'),
                      interrupted=Path('interrupted'),segment=None,finalize_segments=False):
            return SimpleNamespace(review=Path('review.json'),review_sha256='c'*64,lane=lane,prior=[],
                partition_id=partition_id,finalize_partitions=finalize,inherit_correction=True,
                failed_guards_process=failed,interrupted_guards_root=interrupted,
                guard_segment_id=segment,finalize_guard_segments=finalize_segments)
        with patch.object(r,'authority',return_value=expected):
            for args in (arguments(),arguments('R-PREFIX-D'),arguments('R-GUARDS',True),
                         arguments('R-GUARDS',lane='formal'),arguments('R-GUARDS',failed=None),
                         arguments('R-GUARDS',interrupted=None),arguments('R-GUARDS')):
                with self.assertRaises(ValueError):r.execute_physical(args,watchdog)
            with patch.object(r,'execute_physical_correction_guard_segment',return_value=17) as segment:
                self.assertEqual(r.execute_physical(arguments('R-GUARDS',segment='R-GUARDS-A'),watchdog),17);segment.assert_called_once()
            with patch.object(r,'execute_physical_correction_guard_union',return_value=18) as guards:
                self.assertEqual(r.execute_physical(arguments('R-GUARDS',finalize_segments=True),watchdog),18);guards.assert_called_once()
            with patch.object(r,'execute_physical_correction_union',return_value=19) as union:
                self.assertEqual(r.execute_physical(arguments(finalize=True),watchdog),19);union.assert_called_once()

    def test_physical_runtime_compatibility_is_exact_and_nonimplicit(self):
        support=r.physical_support();live=support.runtime_identity()
        body=dict(schema=support.CORRECTION_COMPATIBILITY_SCHEMA,mode='R-GUARDS',
            predecessor=dict(commit=r.PHYSICAL_PREDECESSOR['commit'],tree=r.PHYSICAL_PREDECESSOR['tree'],
                runtime_sha256=r.PHYSICAL_PREDECESSOR_RUNTIME_SHA,review_sha256=r.PHYSICAL_PREDECESSOR_REVIEW_SHA),
            successor=dict(commit='a'*40,tree='b'*40,runtime_sha256=live,review_sha256='c'*64),
            addendum_sha256=r.PHYSICAL_CORRECTION_ADDENDUM_SHA,
            design_review_sha256=r.PHYSICAL_CORRECTION_REVIEW_SHA,
            revision_sha256=r.PHYSICAL_CORRECTION_REVISION_SHA,
            revision_review_sha256=r.PHYSICAL_CORRECTION_REVISION_REVIEW_SHA,
            partition_recovery_sha256=r.PHYSICAL_CORRECTION_RECOVERY_SHA,
            partition_recovery_review_sha256=r.PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA,
            guard_segment_manifest_sha256=r.PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
            unchanged_inputs_sha256=r.PHYSICAL_CORRECTION_UNCHANGED_INPUTS_SHA,
            allowed_changed_paths=sorted(r.PHYSICAL_CORRECTION_CHANGED_PATHS))
        value=dict(body,self_sha256=support.packet.digest(body))
        import ge_beam3_g3c_correction_lease_binding as binding
        captured=support.packet.digest(value)
        with patch.object(binding,'compatibility_identity',return_value=captured):
            self.assertEqual(support.validate_runtime_compatibility(value,r.PHYSICAL_PREDECESSOR_RUNTIME_SHA,live),live)
            for mutate in (
                lambda x:x['predecessor'].__setitem__('runtime_sha256','0'*64),
                lambda x:x['successor'].__setitem__('commit','0'*40),
                lambda x:x['successor'].__setitem__('tree','0'*40),
                lambda x:x['successor'].__setitem__('runtime_sha256','0'*64),
                lambda x:x['successor'].__setitem__('review_sha256','0'*64),
                lambda x:x.__setitem__('mode','FOREIGN'),
                lambda x:x.__setitem__('unchanged_inputs_sha256','0'*64),
                lambda x:x.__setitem__('guard_segment_manifest_sha256','0'*64),
                lambda x:x.__setitem__('allowed_changed_paths',x['allowed_changed_paths'][:-1]),
                lambda x:x.__setitem__('self_sha256','0'*64)):
                bad=copy.deepcopy(value);mutate(bad)
                if bad['self_sha256']==value['self_sha256']:
                    bad['self_sha256']=support.packet.digest({k:v for k,v in bad.items()if k!='self_sha256'})
                with self.assertRaises(ValueError):
                    support.validate_runtime_compatibility(bad,r.PHYSICAL_PREDECESSOR_RUNTIME_SHA,live)
            tree=ast.parse((Path(__file__).resolve().parent/'test_ge_beam3_g3c_physical_correction_guards.py').read_text())
            declared=[]
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(
                        isinstance(target,ast.Name)
                        and target.id=='RUNTIME_COMPATIBILITY_NEGATIVE_COUNT'
                        for target in node.targets):
                    declared.append(ast.literal_eval(node.value))
            self.assertEqual(declared,[r.PHYSICAL_CORRECTION_COMPATIBILITY_NEGATIVE_COUNT])
        with self.assertRaisesRegex(ValueError,'runtime compatibility unnecessary'):
            support.resume(b'{}\n','0'*64,expected_runtime_sha256=live,runtime_compatibility=value)
        with self.assertRaises(ValueError):
            r.physical_runtime_compatibility(({'commit':'a','tree':'b'},{'x':{'sha256':'0'}},b'{}\n'),
                                             (r.PHYSICAL_PREDECESSOR,{'x':{'sha256':'1'}},b'{}\n'))

    def test_physical_correction_lease_binds_exact_segment_and_slice(self):
        inventory=r.physical_inventory('rehearsal');candidate={'commit':'a'*40,'tree':'b'*40};rows={};review=b'{}\n'
        expected=(candidate,rows,review);runtime='c'*64;index=90;_,segment=r.physical_correction_guard_segment_spec('R-GUARDS-A',inventory)
        lease=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',schema=r.SCOPE,run_id=str(uuid.uuid4()),
            gate='g3c-physical',lane='rehearsal',candidate=candidate,inputs=rows,
            review_sha256='d'*64,implementation_review={},contract_sha256=r.PHYSICAL_PLAN_SHA,
            assignment_index=index,assignment=inventory[index],
            whole_inventory_sha256=sha256(r.canonical(inventory)).hexdigest(),runtime_sha256=runtime,
            input_packets={'origin':{'path':str(Path(__file__).resolve()),'bytes':Path(__file__).stat().st_size,
                                     'sha256':sha256(Path(__file__).read_bytes()).hexdigest()}},
            runtime_compatibility={},guard_segment_id='R-GUARDS-A',
            guard_segment_manifest_sha256=r.PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
            guard_segment_assignments_sha256=segment['assignment_sha256'])
        with patch.object(r,'physical_validate_runtime_compatibility_record'),patch.object(
                r.physical_support(),'runtime_identity',return_value=runtime):
            r.physical_lease_expected(lease,expected,'d'*64)
            for key,value in (('guard_segment_id','R-GUARDS-B'),
                              ('guard_segment_manifest_sha256','0'*64),
                              ('guard_segment_assignments_sha256','0'*64)):
                bad=copy.deepcopy(lease);bad[key]=value
                with self.assertRaises(ValueError):r.physical_lease_expected(bad,expected,'d'*64)
            bad=copy.deepcopy(lease);bad['assignment_index']=114;bad['assignment']=inventory[114]
            with self.assertRaises(ValueError):r.physical_lease_expected(bad,expected,'d'*64)

    def test_physical_segment_receipt_indices_are_exact_integers(self):
        self.assertEqual(r.exact_assignment_index(90,90),90)
        for value in (90.0,True,'90',None):
            with self.assertRaises(ValueError):r.exact_assignment_index(value,90)

    def test_physical_correction_tests_and_dual_runtime_view_are_registered(self):
        preflight=next(row for row in r.physical_inventory('rehearsal') if row['kind']=='preflight')
        selected=r.physical_assignment_nodes(preflight,True)
        self.assertEqual([node.rsplit('::',1)[-1] for node in selected],[
            'test_physical_correction_preflight_guards','test_runtime_compatibility_negatives',
            'test_runtime_compatibility_positive'])
        support=r.physical_support();old='1'*64;new='2'*64
        predecessor=support.packet.canonical(dict(schema='X',runtime_sha256=old,value=3))
        successor=support.packet.canonical(dict(schema='X',runtime_sha256=new,value=3))
        self.assertTrue(support.compatible_checkpoint_equal(predecessor,successor,old,new))
        changed=support.packet.canonical(dict(schema='X',runtime_sha256=new,value=4))
        self.assertFalse(support.compatible_checkpoint_equal(predecessor,changed,old,new))

    def test_physical_correction_bundle_requires_final_authority_before_promotion(self):
        watchdog=SimpleNamespace(check=lambda:None)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stage=root/'staged';stage.mkdir()
            for name in ('process.json','scientific.json','receipt.json'):r.write(stage/name,{'name':name})
            def rejected():raise ValueError('late authority mutation')
            with self.assertRaisesRegex(ValueError,'late authority mutation'):
                r.promote_physical_correction_bundle(stage,root,watchdog,rejected)
            self.assertTrue(stage.is_dir())
            self.assertFalse(any((root/name).exists() for name in ('process.json','scientific.json','receipt.json')))
            r.promote_physical_correction_bundle(stage,root,watchdog,lambda:None)
            self.assertFalse(stage.exists())
            self.assertTrue(all((root/name).is_file() for name in ('process.json','scientific.json','receipt.json')))

    def test_physical_failed_guard_incident_distinguishes_unlaunched_nodes(self):
        results={str(index):({'status':'FAILED','files':{}} if index==157 else
            {'status':'NOT_LAUNCHED'} if index>=159 else {'status':'PASSED','files':{}})
            for index in range(90,234)}
        self.assertEqual(r.physical_failed_guard_layout(results),list(range(90,159)))
        for index,replacement in ((157,{'status':'NOT_LAUNCHED'}),(159,{'status':'PASSED','files':{}})):
            changed=copy.deepcopy(results);changed[str(index)]=replacement
            with self.assertRaises(ValueError):r.physical_failed_guard_layout(changed)
        changed=copy.deepcopy(results);changed['159']['extra']=True
        with self.assertRaises(ValueError):r.physical_failed_guard_layout(changed)

    def test_physical_union_requires_ordered_original_indices(self):
        values=[]
        for partition_id,indices,_ in r.PHYSICAL_PARTITIONS:
            values.append(dict(partition_id=partition_id,
                records=[dict(assignment_index=index) for index in indices]))
        records=r.physical_union_records(values)
        self.assertEqual([row['assignment_index'] for row in records],list(range(234)))
        for mutate in (
            lambda rows:rows.pop(),
            lambda rows:rows.__setitem__(0,dict(rows[0],partition_id='R-PREFIX-A')),
            lambda rows:rows[1]['records'].__setitem__(0,dict(rows[1]['records'][0],assignment_index=10.0)),
            lambda rows:rows[1]['records'].__setitem__(0,dict(rows[1]['records'][0],assignment_index=11)),
        ):
            bad=copy.deepcopy(values);mutate(bad)
            with self.assertRaises(ValueError):r.physical_union_records(bad)

    def test_physical_union_finalizer_has_no_mechanics_import(self):
        source=(Path(r.__file__).read_text(encoding='utf-8'))
        tree=ast.parse(source);names={'execute_physical_union','physical_validate_union','physical_union_records',
            'execute_physical_correction_guard_union','physical_validate_correction_guards',
            'execute_physical_correction_union','physical_validate_correction_union'}
        selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
        self.assertEqual({node.name for node in selected},names)
        self.assertFalse(any(isinstance(node,(ast.Import,ast.ImportFrom)) for fn in selected for node in ast.walk(fn)))
        calls={node.func.id for fn in selected for node in ast.walk(fn)
               if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        self.assertFalse(calls&{'physical_run_phase','physical_child','job_type','subprocess','worker'})

    def test_physical_review_schema_and_mutations(self):
        candidate=dict(commit='a'*40,tree='b'*40)
        rows={'docs/reference_cases/ge_beam3_g3c_fixtures_v1.json':dict(bytes=1,sha256='c'*64),
              'src/anyfesolver/_ge_beam3_g3c_definition.py':dict(bytes=1,sha256='d'*64)}
        scope=dict(scope_id=r.SCOPE,gate='g3c-physical',subject_tree=candidate['tree'],
            inputs_sha256=sha256(r.canonical(rows)).hexdigest(),contract_sha256=r.PHYSICAL_PLAN_SHA,
            partition_addendum_sha256=r.PHYSICAL_PARTITION_ADDENDUM_SHA,
            formal_addendum_sha256=r.PHYSICAL_FORMAL_ADDENDUM_SHA,
            formal_design_review_sha256=r.PHYSICAL_FORMAL_DESIGN_REVIEW_SHA,
            formal_measurement_authorized=True,formal_execution_authorized=False,
            proof_compressed_plan_sha256=r.PHYSICAL_PROOF_COMPRESSED_PLAN_SHA,
            proof_compressed_tool_sha256=r.PHYSICAL_PROOF_COMPRESSED_TOOL_SHA,
            proof_compressed_checker_sha256=r.PHYSICAL_PROOF_COMPRESSED_CHECKER_SHA,
            proof_compressed_initial_review_sha256=r.PHYSICAL_PROOF_COMPRESSED_INITIAL_REVIEW_SHA,
            proof_compressed_measurement_sha256=r.PHYSICAL_PROOF_COMPRESSED_MEASUREMENT_SHA,
            proof_compressed_partition_sha256=r.PHYSICAL_PROOF_COMPRESSED_PARTITION_SHA,
            proof_compressed_checker_initial_review_sha256=r.PHYSICAL_PROOF_COMPRESSED_CHECKER_INITIAL_REVIEW_SHA,
            proof_compressed_cycle_authorized=True,
            execution_authorized=True,full_g3c_qualified=False,production_qualified=False,
            correction_addendum_sha256=r.PHYSICAL_CORRECTION_ADDENDUM_SHA,
            correction_design_review_sha256=r.PHYSICAL_CORRECTION_REVIEW_SHA,
            correction_revision_sha256=r.PHYSICAL_CORRECTION_REVISION_SHA,
            correction_revision_review_sha256=r.PHYSICAL_CORRECTION_REVISION_REVIEW_SHA,
            correction_partition_recovery_sha256=r.PHYSICAL_CORRECTION_RECOVERY_SHA,
            correction_partition_recovery_review_sha256=r.PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA,
            correction_guard_segment_manifest_sha256=r.PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
            predecessor_commit=r.PHYSICAL_PREDECESSOR['commit'],
            predecessor_runtime_sha256=r.PHYSICAL_PREDECESSOR_RUNTIME_SHA,
            correction_inheritance_authorized=True,correction_partition_recovery_authorized=True)
        review=dict(decision='ACCEPTED_G3C_PHYSICAL_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',findings=[],
            reviewer=dict(independent=True),scope=scope,subject_commit=candidate['commit'])
        raw=r.canonical(review);digest=sha256(raw).hexdigest()
        r.verify_review(raw,digest,candidate,rows,'g3c-physical')
        base=dict(candidate=candidate,inputs=rows,implementation_review=review,review_sha256=digest,
                  input_packets={'origin':dict(path='X',bytes=1,sha256='f'*64)})
        probes=[row for row in r.physical_support().mutation_inventory() if row['executor']=='authority']
        for row in probes:
            records=r.physical_authority_probe(dict(base,assignment=dict(kind='authority-mutation',
                probe={k:v for k,v in row.items() if k!='executor'})))
            self.assertEqual(len(records),1);self.assertTrue(records[0]['passed'])
        for field,value in (('findings',['x']),('subject_commit','e'*40),('scope',{})):
            bad=dict(review,**{field:value});encoded=r.canonical(bad)
            with self.assertRaises(ValueError):r.verify_review(encoded,sha256(encoded).hexdigest(),candidate,rows,'g3c-physical')

    def test_physical_lease_and_prerequisite_bindings(self):
        candidate=dict(commit='a'*40,tree='b'*40);rows={'x':dict(bytes=1,sha256='c'*64)}
        scope=dict(scope_id=r.SCOPE,gate='g3c-physical',subject_tree=candidate['tree'],
            inputs_sha256=sha256(r.canonical(rows)).hexdigest(),contract_sha256=r.PHYSICAL_PLAN_SHA,
            partition_addendum_sha256=r.PHYSICAL_PARTITION_ADDENDUM_SHA,
            formal_addendum_sha256=r.PHYSICAL_FORMAL_ADDENDUM_SHA,
            formal_design_review_sha256=r.PHYSICAL_FORMAL_DESIGN_REVIEW_SHA,
            formal_measurement_authorized=True,formal_execution_authorized=False,
            execution_authorized=True,full_g3c_qualified=False,production_qualified=False,
            correction_addendum_sha256=r.PHYSICAL_CORRECTION_ADDENDUM_SHA,
            correction_design_review_sha256=r.PHYSICAL_CORRECTION_REVIEW_SHA,
            correction_revision_sha256=r.PHYSICAL_CORRECTION_REVISION_SHA,
            correction_revision_review_sha256=r.PHYSICAL_CORRECTION_REVISION_REVIEW_SHA,
            correction_partition_recovery_sha256=r.PHYSICAL_CORRECTION_RECOVERY_SHA,
            correction_partition_recovery_review_sha256=r.PHYSICAL_CORRECTION_RECOVERY_REVIEW_SHA,
            correction_guard_segment_manifest_sha256=r.PHYSICAL_CORRECTION_GUARD_SEGMENT_MANIFEST_SHA,
            predecessor_commit=r.PHYSICAL_PREDECESSOR['commit'],
            predecessor_runtime_sha256=r.PHYSICAL_PREDECESSOR_RUNTIME_SHA,
            correction_inheritance_authorized=True,correction_partition_recovery_authorized=True)
        review=dict(decision='ACCEPTED_G3C_PHYSICAL_IMPLEMENTATION_FOR_BOUNDED_DEVELOPMENT',findings=[],
            reviewer=dict(independent=True),scope=scope,subject_commit=candidate['commit'])
        review_raw=r.canonical(review);inventory=r.physical_inventory('local')
        lease=dict(kind='G3C_PHYSICAL_PRIVATE_DEVELOPMENT',schema=r.SCOPE,run_id=str(uuid.uuid4()),
            gate='g3c-physical',lane='local',candidate=candidate,inputs=rows,
            review_sha256=sha256(review_raw).hexdigest(),implementation_review=review,
            contract_sha256=r.PHYSICAL_PLAN_SHA,assignment_index=0,assignment=inventory[0],
            whole_inventory_sha256=sha256(r.canonical(inventory)).hexdigest(),
            runtime_sha256=r.physical_support().runtime_identity(),input_packets={})
        expected=(candidate,rows,review_raw);r.physical_lease_expected(lease,expected,lease['review_sha256'])
        for key,value in (('assignment_index',1),('assignment',inventory[1]),('runtime_sha256','0'*64),
                          ('whole_inventory_sha256','1'*64),('input_packets',{'x':{}})):
            bad=copy.deepcopy(lease);bad[key]=value
            with self.assertRaises((ValueError,OSError)):r.physical_lease_expected(bad,expected,lease['review_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scientific.json'
            prior_inventory=r.physical_inventory('smoke');prior_records=[]
            for index,assignment in enumerate(prior_inventory):
                science=dict(schema='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1',candidate=candidate,lane='smoke',
                    assignment_index=index,assignment=assignment,records=[{}],
                    full_g3c_qualified=False,production_qualified=False)
                prior_records.append(dict(assignment_index=index,
                    science_sha256=sha256(r.canonical(science)).hexdigest(),science=science))
            body=dict(schema='GE_BEAM3_G3C_PHYSICAL_AGGREGATE_V1',candidate=candidate,lane='smoke',
                      inventory_sha256=sha256(r.canonical(prior_inventory)).hexdigest(),records=prior_records,
                      passed=True,terminal='COMPLETE_GE_BEAM3_G3C_PHYSICAL_SMOKE_ONLY',
                      full_g3c_qualified=False,production_qualified=False)
            value=dict(body,self_sha256=sha256(r.canonical(body)).hexdigest());r.write(path,value)
            r.verify_physical_priors([],expected,'smoke')
            # A self-rehashed aggregate without its accepted node/process DAG
            # is never an admissible prerequisite.
            with self.assertRaises(ValueError):r.verify_physical_priors([path],expected,'local')
            bad=dict(value,self_sha256='0'*64);path.unlink();r.write(path,bad)
            with self.assertRaises(ValueError):r.verify_physical_priors([path],expected,'local')
            bad=dict(value,inventory_sha256='0'*64);bad['self_sha256']=sha256(r.canonical(
                {k:v for k,v in bad.items() if k!='self_sha256'})).hexdigest()
            path.unlink();r.write(path,bad)
            with self.assertRaises(ValueError):r.verify_physical_priors([path],expected,'local')

    def test_physical_packet_dependency_map(self):
        rehearsal=r.physical_inventory('rehearsal');histories={}
        for row in rehearsal:
            if row['kind']=='history':
                histories[row['case']['case_id']]={i:dict(path='X',bytes=1,sha256=str(i%10)*64)
                                                   for i in range(row['stages']+1)}
        prefix=next(row for row in rehearsal if row['kind']=='prefix')
        packets=r.physical_inputs_for(prefix,histories)
        self.assertEqual(set(packets),{'prefix','final'})
        preflight=next(row for row in rehearsal if row['kind']=='preflight')
        self.assertEqual(set(r.physical_inputs_for(preflight,histories)),{'origin'})
        missing=copy.deepcopy(histories);missing.pop(prefix['case']['case_id'])
        with self.assertRaises(ValueError):r.physical_inputs_for(prefix,missing)
        lease=dict(assignment=prefix,input_packets=packets);r.physical_join_input_packets(lease,histories)
        with tempfile.TemporaryDirectory() as directory:
            foreign=Path(directory)/'foreign.json';foreign.write_bytes(b'{"valid":"foreign"}\n')
            bad=copy.deepcopy(lease);bad['input_packets']['prefix']=r.packet_descriptor(foreign)
            with self.assertRaises(ValueError):r.physical_join_input_packets(bad,histories)

    def test_physical_prerequisite_lineage_is_exact(self):
        current=[dict(kind='base',identity=identity,path='X'+str(index),science={'bytes':1,'sha256':'a'*64},
                      receipt={'bytes':1,'sha256':'b'*64},process={'bytes':1,'sha256':'c'*64})
                 for index,identity in enumerate(('smoke','local','R-HISTORY','R-PREFIX-A'))]
        r.physical_join_prerequisite_chain(current,current[:3])
        mutated=copy.deepcopy(current[:3]);mutated[0]['path']='FOREIGN'
        for predecessor in (current[1:3],mutated,current+[dict(current[0])]):
            with self.assertRaises(ValueError):r.physical_join_prerequisite_chain(current,predecessor)

    def test_physical_node_record_schema_rejects_extra_fields(self):
        assignment=next(row for row in r.physical_inventory('rehearsal')
                        if row['kind']=='authority-mutation')
        lease=dict(candidate=dict(commit='a'*40,tree='b'*40),lane='rehearsal',assignment_index=0,
                   assignment=assignment,input_packets={'origin':dict(path='X',bytes=1,sha256='c'*64)})
        record=dict(kind='authority-mutation',assignment=assignment['probe'],origin_sha256='c'*64,
                    before_sha256='d'*64,after_sha256='e'*64,expected_error='review hash',passed=True)
        science=dict(schema='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1',candidate=lease['candidate'],
                     lane='rehearsal',assignment_index=0,assignment=assignment,records=[record],
                     full_g3c_qualified=False,production_qualified=False)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);r.write(out/'scientific.node.json',science)
            raw=(out/'scientific.node.json').read_bytes()
            r.write(out/'completion.json',dict(assignment_index=0,assignment=assignment,
                selected=[],passed=[],scientific=r.fingerprint(raw)))
            r.physical_verify_node(out,lease)
            science['records'][0]['unregistered']=False;(out/'scientific.node.json').unlink()
            r.write(out/'scientific.node.json',science)
            raw=(out/'scientific.node.json').read_bytes()
            (out/'completion.json').unlink()
            r.write(out/'completion.json',dict(assignment_index=0,assignment=assignment,
                selected=[],passed=[],scientific=r.fingerprint(raw)))
            with self.assertRaises(ValueError):r.physical_verify_node(out,lease)

    def test_physical_atomic_canonical_promotion(self):
        watchdog=SimpleNamespace(check=lambda:None)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'scientific.json'
            r.atomic_canonical(target,{'accepted':True},watchdog)
            self.assertEqual(target.read_bytes(),r.canonical({'accepted':True}))
            self.assertFalse((root/'scientific.pending.json').exists())
            raced=root/'raced.json'
            def race(source,destination):
                Path(destination).write_bytes(b'foreign\n');raise FileExistsError('raced destination')
            with patch.object(r.os,'link',side_effect=race):
                with self.assertRaises(FileExistsError):r.atomic_canonical(raced,{'accepted':True},watchdog)
            self.assertEqual(raced.read_bytes(),b'foreign\n')
            with self.assertRaises(ValueError):r.atomic_canonical(target,{'accepted':True},watchdog)
            failed=root/'receipt.json'
            with patch.object(r.os,'link',side_effect=OSError('injected promotion failure')):
                with self.assertRaises(OSError):r.atomic_canonical(failed,{'accepted':True},watchdog)
            self.assertFalse(failed.exists());self.assertTrue((root/'receipt.pending.json').is_file())

    def test_physical_owner_record_types_and_obligation_map_are_exact(self):
        assignment=r.physical_inventory('local')[0];candidate=dict(commit='a'*40,tree='b'*40)
        records=[dict(test='inventory',assignment=None,histories=375,events=3075,prefixes=3450,
                      obligations=r.PHYSICAL_OBLIGATIONS,production_qualified=False,full_g3c_qualified=False),
                 dict(test='inert_schema',assignment=None,rejections=6,
                      production_qualified=False,full_g3c_qualified=False)]
        template=dict(schema='GE_BEAM3_G3C_PHYSICAL_NODE_SCIENCE_V1',candidate=candidate,lane='local',
                      assignment_index=0,assignment=assignment,records=records,
                      full_g3c_qualified=False,production_qualified=False)
        lease=dict(candidate=candidate,lane='local',assignment_index=0,assignment=assignment,input_packets={})
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            def store(value,completion_mutator=None):
                for name in ('scientific.node.json','completion.json'):
                    path=out/name
                    if path.exists():path.unlink()
                r.write(out/'scientific.node.json',value);raw=(out/'scientific.node.json').read_bytes()
                nodes=r.physical_assignment_nodes(assignment)
                completion=dict(assignment_index=0,assignment=assignment,
                    selected=nodes,passed=nodes,scientific=r.fingerprint(raw))
                if completion_mutator is not None:completion_mutator(completion)
                r.write(out/'completion.json',completion)
            store(template);r.physical_verify_node(out,lease)
            mutations=[]
            bad=copy.deepcopy(template);bad['records'][0]['histories']=375.0;mutations.append(bad)
            bad=copy.deepcopy(template);bad['records'][0]['obligations']['MO01']='UNBOUND';mutations.append(bad)
            bad=copy.deepcopy(template);bad['full_g3c_qualified']=None;mutations.append(bad)
            bad=copy.deepcopy(template);bad['production_qualified']=[];mutations.append(bad)
            bad=copy.deepcopy(template);bad['assignment_index']=False;mutations.append(bad)
            for bad in mutations:
                store(bad)
                with self.assertRaises(ValueError):r.physical_verify_node(out,lease)
            for mutate in (
                lambda value:value.__setitem__('assignment_index',False),
                lambda value:value.__setitem__('assignment_index',0.0),
                lambda value:value['scientific'].__setitem__('bytes',float(value['scientific']['bytes'])),
                lambda value:value['scientific'].__setitem__('bytes',False),
            ):
                store(template,mutate)
                with self.assertRaises(ValueError):r.physical_verify_node(out,lease)

    def test_physical_atomicity_stages_follow_registered_variants(self):
        assignments=[row['assignment'] for row in r.physical_inventory('local') if row['kind']=='owner']
        _,source=r.physical_support().packet.authorities()
        for assignment in assignments:
            expanded=source.expand_inert(assignment['graph'],assignment['variant'])[1]
            ids=[e['id'] for e in expanded['graph']['elements'] if e['family']!='NATIVE']
            self.assertEqual(r.physical_adapter_ids(assignment),ids)
            self.assertEqual(r.physical_atomicity_stages(assignment),
                ['family:'+str(ids[-1]),'prepare','native_committed','before_publish'])
            self.assertEqual(r.physical_observation_stages(assignment),list(dict.fromkeys(
                ('pose','family:'+str(ids[0]),'family:'+str(ids[-1]),'prepare','before_publish'))))
        self.assertEqual(len(assignments),25)
        renumbered=next(row for row in assignments if row=={'graph':'J_B2_PAIR','variant':'RENUMBERED'})
        self.assertEqual(r.physical_observation_stages(renumbered)[1],'family:20055')

    def test_physical_process_metadata_types_are_exact(self):
        assignment=r.physical_inventory('smoke')[0]
        lease=dict(assignment_index=0,assignment=assignment)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for name in r.physical_expected_node_files(assignment)-{'process.json'}:
                r.write(out/name,{'name':name})
            files={name:r.fingerprint((out/name).read_bytes())
                   for name in sorted(r.physical_expected_node_files(assignment)-{'process.json'})}
            template=dict(status='PASSED',active_processes=0,peak_tree_bytes=0,drained=True,
                elapsed_seconds=0.0,kind='GE_BEAM3_G3C_PHYSICAL_CHILD_PROCESS_V1',
                assignment_index=0,assignment_sha256=sha256(r.canonical(assignment)).hexdigest(),
                returncode=0,files=files)
            def store(value):
                path=out/'process.json'
                if path.exists():path.unlink()
                r.write(path,value)
            store(template);r.physical_verify_process(out,lease)
            for field,value in (('active_processes',False),('assignment_index',False),
                                ('assignment_index',0.0),('returncode',False),('returncode',0.0)):
                bad=copy.deepcopy(template);bad[field]=value;store(bad)
                with self.assertRaises(ValueError):r.physical_verify_process(out,lease)
            bad=copy.deepcopy(template);bad['files']['scientific.node.json']['bytes']=False;store(bad)
            with self.assertRaises(ValueError):r.physical_verify_process(out,lease)
            store(template);r.write(out/'unregistered.json',{'accepted':False})
            with self.assertRaises(ValueError):r.physical_verify_process(out,lease)

    def test_physical_root_closed_world_rejects_extra_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r.write(root/'scientific.json',{'accepted':True})
            r.physical_closed_world(root,{'scientific.json'})
            r.write(root/'extra.json',{'accepted':False})
            with self.assertRaises(ValueError):r.physical_closed_world(root,{'scientific.json'})
            (root/'extra.json').unlink();(root/'extra').mkdir()
            with self.assertRaises(ValueError):r.physical_closed_world(root,{'scientific.json'})

    def test_strict_json(self):
        for raw in (b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n'):
            with self.assertRaises(ValueError):r.environment.strict(raw)

    def test_review_hash_identity_and_inputs(self):
        candidate=dict(commit='a'*40,tree='b'*40);rows={'x':dict(bytes=1,sha256='c'*64)}
        value=dict(decision='ACCEPTED_GE_BEAM3_REGISTERED_GATE_FOR_BOUNDED_EXECUTION',findings=[],
            reviewer=dict(independent=True),subject_commit=candidate['commit'],scope=dict(scope_id=r.SCOPE,gate='b2-core',
            subject_tree=candidate['tree'],inputs_sha256=sha256(r.canonical(rows)).hexdigest(),contract_sha256=r.CONTRACT_SHA))
        raw=r.canonical(value);h=sha256(raw).hexdigest();r.verify_review(raw,h,candidate,rows)
        with self.assertRaises(ValueError):r.verify_review(raw,'0'*64,candidate,rows)
        with self.assertRaises(ValueError):r.verify_review(raw,h,candidate,{})
        with self.assertRaises(ValueError):r.verify_review(raw,h,candidate,rows,'b2-adapter')
        for key,v in (('subject_commit','d'*40),('findings',['defect']),('reviewer',dict(independent=False))):
            bad=dict(value,**{key:v});b=r.canonical(bad)
            with self.assertRaises(ValueError):r.verify_review(b,sha256(b).hexdigest(),candidate,rows)

    def test_lease_mismatch_before_construction(self):
        expected=({'commit':'a','tree':'b'},{},b'{}\n')
        lease=dict(schema=r.SCOPE,run_id=str(uuid.uuid4()),gate='b2-core',lane='smoke',
            candidate=expected[0],inputs={},review_sha256='c',selected=r.inventory('smoke'))
        r.validate_lease(lease,expected,'c','smoke',Path('.'))
        for key,v in (('gate','other'),('candidate',{}),('selected',[]),('inputs',{'altered':1}),('run_id','not-a-uuid')):
            bad=copy.deepcopy(lease);bad[key]=v
            with self.assertRaises(ValueError):r.validate_lease(bad,expected,'c','smoke',Path('.'))

    def test_exclusive_outputs_and_attempt_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r.claim_attempt(root,'test')
            with self.assertRaises(FileExistsError):r.claim_attempt(root,'test')
            self.assertFalse((root/'scientific.json').exists())

    def test_inactivity_memory_and_timeout_drain(self):
        for mode in ('inactivity','memory','wall'):
            j=Job();c=Clock()
            if mode=='memory':j.mem=r.MEMORY+1
            def progress():
                if mode=='wall':j.cpu+=1
                return (0,0)
            result=r.monitor(j,j,progress,c,c.sleep)
            self.assertEqual(result['status'],'RESOURCE_BLOCKED');self.assertTrue(j.killed)
            self.assertEqual(result['active_processes'],0);self.assertLess(c.value,600)

    def test_exited_parent_does_not_release_live_tree(self):
        j=Job();j.code=0;c=Clock()
        result=r.monitor(j,j,lambda:(0,0),c,c.sleep)
        self.assertEqual(result['status'],'RESOURCE_BLOCKED');self.assertTrue(j.killed)
        for code,status in ((0,'PASSED'),(1,'FAILED')):
            j=Job();j.active=0;j.code=code
            self.assertEqual(r.monitor(j,j,lambda:(0,0),c,c.sleep)['status'],status)

    def test_exception_closes_and_drains_no_canonical_output(self):
        with tempfile.TemporaryDirectory() as d:
            j=Job();root=Path(d)
            expected=({'commit':'a','tree':'b'},{},b'{}\n')
            args=SimpleNamespace(review=root/'review-source',review_sha256='c',gate='b2-core',lane='smoke')
            with patch.object(r,'authority',return_value=expected),patch.object(r,'job_type',return_value=lambda n:j), \
                 patch.object(r.tempfile,'mkdtemp',return_value=d),patch.object(r,'monitor',side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError):r.execute(args)
            self.assertTrue(j.killed and j.closed);self.assertEqual(j.active,0)
            self.assertFalse((root/'scientific.json').exists());self.assertTrue((root/'process.json').exists())

    def test_no_mechanics_import_before_authority(self):
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('anyfesolver',sys.modules)

    def test_checker_gate_and_replica_before_import(self):
        with self.assertRaises(ValueError):r.q4_checker(Path('.'),'wrong','1','bad')
        with self.assertRaises(ValueError):r.q4_checker(Path('.'),r.Q4_FIXTURES[0],'3','bad')
        with patch.object(r,'read',return_value=r.canonical({'gate':'b2-adapter','lane':'core'})), \
             patch.object(r,'authority',side_effect=AssertionError('wrong gate entered authority')):
            with self.assertRaises(ValueError):r.q4_checker(Path('.'),r.Q4_FIXTURES[0],'1','bad')
        self.assertNotIn('ge_beam3_q4_recovery_coefficient_checker',sys.modules)
        with self.assertRaises(ValueError):r.q4_checker(Path('.'),r.Q4_FIXTURES[0],'1','bad','q4-affine-exact')
        with patch.object(r,'read',return_value=r.canonical({'gate':'q4-audit','lane':'core'})), \
             patch.object(r,'authority',side_effect=AssertionError('cross gate entered authority')):
            with self.assertRaises(ValueError):r.q4_checker(Path('.'),r.AFFINE_FIXTURES[0],'1','bad','q4-affine-exact')
        self.assertNotIn('ge_beam3_q4_affine_recovery_checker',sys.modules)

    def test_affine_review_gate_binding(self):
        candidate=dict(commit='a'*40,tree='b'*40);rows={}
        value=dict(decision='ACCEPTED_GE_BEAM3_REGISTERED_GATE_FOR_BOUNDED_EXECUTION',findings=[],
            reviewer=dict(independent=True),subject_commit=candidate['commit'],scope=dict(scope_id=r.SCOPE,
            gate='q4-affine-exact',subject_tree=candidate['tree'],inputs_sha256=sha256(r.canonical(rows)).hexdigest(),
            contract_sha256=r.AFFINE_PLAN_SHA))
        raw=r.canonical(value);digest=sha256(raw).hexdigest()
        r.verify_review(raw,digest,candidate,rows,'q4-affine-exact')
        with self.assertRaises(ValueError):r.verify_review(raw,digest,candidate,rows,'q4-audit')

    def test_affine_terminal_inventory_and_first_witness(self):
        rows=[];zero={'coefficient':['0']*8}
        for fixture in r.AFFINE_FIXTURES:
            proof=dict(schema='GE_BEAM3_Q4_AFFINE_RECOVERY_EXACT_FIXTURE_V1',fixture_id=fixture,
                coefficient_records=[zero]*7125,coefficient_count=7125,degree_counts={'3':1140,'4':5985},
                nonzero_count=0,zero_count=7125,first_nonzero=None,physical_recovery_qualified=False,full_g3c_qualified=False)
            verification=dict(fixture_id=fixture,coefficient_count=7125,nonzero_count=0,first_nonzero=None,
                independently_verified=True,physical_recovery_qualified=False,full_g3c_qualified=False)
            rows.append(dict(test=fixture,proof=proof,verification=verification,checker_replicas_byte_identical=True))
        self.assertEqual(r.affine_adjudication(rows,'core')['coefficient_count'],21375)
        self.assertEqual(r.affine_adjudication(rows,'core')['terminal'],'UNCLASSIFIED_G3C_Q4_AFFINE_RECOVERY_EXACT_IDENTITIES_ONLY')
        witness={'coefficient':['1']+['0']*7}
        for row in reversed(rows):
            row['proof']['coefficient_records'][0]=witness
            row['proof'].update(first_nonzero=witness,nonzero_count=1,zero_count=7124)
            row['verification'].update(first_nonzero=witness,nonzero_count=1)
            self.assertEqual(r.affine_adjudication(rows,'core')['first_nonzero']['fixture_id'],row['test'])
        for wrong in (rows[::-1],rows[:-1],rows+rows[:1]):
            with self.assertRaises(ValueError):r.affine_adjudication(wrong,'core')
        self.assertEqual(r.affine_adjudication([],'smoke')['terminal'],'NOT_ADJUDICATED_SMOKE_ONLY')

    def test_q4_terminal_order_and_incomplete_rejection(self):
        # Inert decision-layer fixtures, not algebraic or scientific evidence.
        zero={'coefficient':['0']*8};rows=[]
        for fixture in r.Q4_FIXTURES:
            proof=dict(coefficient_records=[zero]*20150,coefficient_count=20150,
                       nonzero_count=0,zero_count=20150,first_nonzero=None)
            rows.append(dict(test=fixture,proof=proof,checker_replicas_byte_identical=True,
                verification=dict(first_nonzero=None,nonzero_count=0,independently_verified=True)))
        self.assertEqual(r.q4_adjudication(rows,'core')['terminal'],'UNCLASSIFIED_G3C_Q4_TWO_FIXTURE_COEFFICIENT_IDENTITIES')
        nonzero={'coefficient':['1']+['0']*7}
        for row in reversed(rows):
            row['proof']['coefficient_records'][0]=nonzero
            row['proof'].update(first_nonzero=nonzero,nonzero_count=1,zero_count=20149)
            row['verification'].update(first_nonzero=nonzero,nonzero_count=1)
            result=r.q4_adjudication(rows,'core')
            self.assertEqual(result['terminal'],'NO_GO_G3C_Q4_NATURAL_RETAINED_SPACE_FINITE_IDENTITY')
            self.assertEqual(result['first_nonzero']['fixture_id'],row['test'])
        with self.assertRaises(ValueError):r.q4_adjudication(rows[::-1],'core')
        with self.assertRaises(ValueError):r.q4_adjudication(rows[:1],'core')
        self.assertEqual(r.q4_adjudication([],'smoke')['terminal'],'NOT_ADJUDICATED_SMOKE_ONLY')

    def test_whole_invocation_watchdog(self):
        timers=[];exits=[]
        class Timer:
            def __init__(self,delay,callback):
                self.delay=delay;self.callback=callback;self.started=False;self.cancelled=False;timers.append(self)
            def start(self):self.started=True
            def cancel(self):self.cancelled=True
        w=r.WaveWatchdog(timer=Timer,exit_process=exits.append)
        self.assertEqual([t.delay for t in timers],[1780,1800])
        self.assertTrue(all(t.started and t.daemon for t in timers))
        def blocked_authority(*args):
            timers[0].callback()  # deadline before a job exists
            return ({},{},b'{}\n')
        with patch.object(r,'WaveWatchdog',return_value=w),patch.object(r,'authority',side_effect=blocked_authority):
            with self.assertRaises(TimeoutError):r.execute(SimpleNamespace(review=None,review_sha256=None,gate='b2-core'))
        self.assertEqual(exits,[124]);self.assertTrue(all(t.cancelled for t in timers))
        w=r.WaveWatchdog(timer=Timer,exit_process=exits.append);j=Job();w.attach(j)
        w.expire();self.assertTrue(j.killed)
        with self.assertRaises(TimeoutError):w.check()  # no publication after expiration
        w.hard_exit();self.assertEqual(exits,[124,124,124]);w.close()

    def test_numerical_assignment_binds_single_node_and_whole_inventory(self):
        selected=['test.py::n'+str(i) for i in range(15)]
        lease=dict(schema=r.SCOPE,run_id=str(uuid.uuid4()),gate='q4-affine-numerical',lane='core',
            candidate=dict(commit='a'*40,tree='b'*40),inputs={},review_sha256='c'*64,selected=selected,
            observation_manifest_sha256='f'*64)
        assignment=r.numerical_assignment(lease,4);r.validate_assignment(assignment,lease,4)
        for key,value in (('index',5),('node',selected[5]),('lane','smoke'),('parent_lease_sha256','0'*64),
                          ('whole_inventory_sha256','0'*64),('inputs_sha256','0'*64),('run_id',str(uuid.uuid4()))):
            bad=dict(assignment,**{key:value})
            with self.assertRaises(ValueError):r.validate_assignment(bad,lease,4)
        for wrong in (-1,15,True,'1'):
            with self.assertRaises(ValueError):r.numerical_assignment(lease,wrong)
        with self.assertRaises(ValueError):r.numerical_assignment(dict(lease,gate='q4-affine-exact'),0)

    def test_numerical_batch_independent_limits_and_worker_cap(self):
        class Watch:
            def check(self):pass
        for failure in ('inactivity','memory','wall'):
            clock=Clock();entries=[]
            for i in range(3):
                job=Job()
                if failure=='memory' and i==0:job.mem=r.MEMORY+1
                def progress(j=job,index=i):
                    if index or failure=='wall':j.cpu+=1
                    return (0,0)
                entries.append(dict(job=job,process=job,start=0,progress=progress))
            r.monitor_batch(entries,Watch(),clock,clock.sleep)
            self.assertEqual(entries[0]['record']['status'],'RESOURCE_BLOCKED')
            self.assertTrue(all(e['job'].killed and e['record']['active_processes']==0 for e in entries))
            self.assertLess(clock.value,600)
        for count in (0,4):
            with self.assertRaises(ValueError):r.monitor_batch([{}]*count,Watch())

    def test_numerical_batch_does_not_accept_live_descendant(self):
        clock=Clock();job=Job();job.code=0
        entry=dict(job=job,process=job,start=0,progress=lambda:(0,0))
        r.monitor_batch([entry],SimpleNamespace(check=lambda:None),clock,clock.sleep)
        self.assertEqual(entry['record']['status'],'RESOURCE_BLOCKED')
        self.assertTrue(job.killed)

    def test_numerical_result_schema_and_union_completeness(self):
        lease=dict(run_id=str(uuid.uuid4()),gate='q4-affine-numerical',lane='core',candidate=dict(commit='a'*40,tree='b'*40),inputs={},
                   selected=['fixture.py::'+name for name in r.NUMERICAL_TESTS],observation_manifest_sha256='f'*64)
        values=[];observations={}
        for i in range(15):
            a=r.numerical_assignment(lease,i);digest=sha256(r.canonical(a)).hexdigest()
            value=dict(schema='GE_BEAM3_REGISTERED_NUMERICAL_NODE_V1',node=a['node'],index=i,candidate=lease['candidate'],
                inputs_sha256=a['inputs_sha256'],whole_inventory_sha256=a['whole_inventory_sha256'],lane='core',
                records=inert_records(i),contradictions=[],status='PASSED',
                physical_recovery_scope='REGISTERED_AFFINE_LOCAL_ONLY',full_g3c_qualified=False,production_qualified=False,
                observation_manifest_sha256=lease['observation_manifest_sha256'])
            observations.update(inert_observations(a['node'],value['records']))
            raw=r.canonical(value);completion=dict(assignment_sha256=digest,node=a['node'],scientific=r.fingerprint(raw))
            self.assertEqual(r.validate_numerical_result(raw,completion,lease,i,digest,observations),value);values.append(value)
            for key,bad_value in (('index',True),('records',[]),('status','CONTRADICTION'),('production_qualified',True)):
                bad=dict(value,**{key:bad_value});bad_raw=r.canonical(bad)
                with self.assertRaises(ValueError):r.validate_numerical_result(bad_raw,
                    dict(completion,scientific=r.fingerprint(bad_raw)),lease,i,digest,observations)
        self.assertEqual(r.numerical_union(lease,values,observations)['terminal'],'PROVISIONAL_GO_G3C_Q4_AFFINE_LOCAL_PHYSICAL_RECOVERY_ONLY')
        for bad in (values[:-1],values[::-1],values+values[:1]):
            with self.assertRaises(ValueError):r.numerical_union(lease,bad,observations)
        values[4]['contradictions']=[inert_contradiction(values[4]['node'],'work',values[4]['records'][0]['tables']['work'][1],lease)]
        values[4]['status']='CONTRADICTION'
        self.assertEqual(r.numerical_union(lease,values,observations)['terminal'],'NO_GO_G3C_Q4_AFFINE_RECOVERY_VARIATIONAL_OR_STATE')

    def test_numerical_ordered_table_identity_not_only_count(self):
        for i,name in enumerate(r.NUMERICAL_TESTS):
            value=inert_records(i)[0]
            r.validate_numerical_tables('fixture.py::'+name,[value])
            for key in value['tables']:
                bad=copy.deepcopy(value);bad['tables'][key][0]['id']='substituted-but-unique'
                with self.assertRaises(ValueError):r.validate_numerical_tables('fixture.py::'+name,[bad])
                if len(value['tables'][key])>1:
                    bad=copy.deepcopy(value);bad['tables'][key]=bad['tables'][key][::-1]
                    with self.assertRaises(ValueError):r.validate_numerical_tables('fixture.py::'+name,[bad])

    def test_every_numerical_row_rejects_missing_extra_and_wrong_types(self):
        def paths(value,path=()):
            if type(value)is dict:
                for key,item in value.items():
                    yield path+(key,),item
                    yield from paths(item,path+(key,))
            elif type(value)is list:
                for index,item in enumerate(value):yield from paths(item,path+(index,))
        def owner(value,path):
            for key in path[:-1]:value=value[key]
            return value,path[-1]
        checked=set()
        for tables in r.NUMERICAL_TABLE_IDS:
            for table,ids in tables.items():
                # Distinct schema branches include ZERO/nonzero, all negative
                # probe IDs, graph renumberings and both director polarities.
                selected=ids if table in ('immutability','rejections','graph','director','chart_mutations') else [ids[0],ids[-1]]
                for identity in selected:
                    if (table,identity) in checked:continue
                    checked.add((table,identity));row=inert_row(table,identity)
                    r.validate_numerical_row(table,row)
                    for path,item in paths(row):
                        if type(path[-1])is str:
                            bad=copy.deepcopy(row);target,key=owner(bad,path);del target[key]
                            with self.assertRaises(ValueError):r.validate_numerical_row(table,bad)
                        if type(item) in (float,int):
                            for wrong in (True,float('nan'),float('inf'),'0'):
                                bad=copy.deepcopy(row);target,key=owner(bad,path);target[key]=wrong
                                with self.assertRaises(ValueError):r.validate_numerical_row(table,bad)
                    bad=copy.deepcopy(row);bad['unregistered_summary']=True
                    with self.assertRaises(ValueError):r.validate_numerical_row(table,bad)
                    if table in r.PHYSICAL_TABLES:
                        bad=copy.deepcopy(row);bad['physical_checks']['physical_force']['actual']['shape']=[23]
                        with self.assertRaises(ValueError):r.validate_numerical_row(table,bad)

    def test_contradiction_receipt_bijection_and_frozen_observed_state(self):
        node='fixture.py::'+r.NUMERICAL_TESTS[4];records=inert_records(4)
        lease=dict(candidate={'commit':'a'*40},inputs={})
        observations=inert_observations(node,records)
        row=records[0]['tables']['work'][1];witness=inert_contradiction(node,'work',row,lease)
        r.validate_contradictions(node,records,[witness],lease,observations)
        for bad in ([],[witness,witness],[dict(payload=witness['payload'],verification={'accepted':True})]):
            with self.assertRaises(ValueError):r.validate_contradictions(node,records,bad,lease,observations)
        for part,key,value in (('payload','row_id',records[0]['tables']['work'][2]['id']),
            ('payload','node','other-node'),('payload','candidate_identity','c'*40),
            ('payload','actual',1.),('payload','state_sha256','d'*64),
            ('verification','payload_sha256','e'*64),('verification','relative_error',0.),
            ('verification','state_sha256','b'*64),('verification','expected',{'shape':[24],'sha256':'0'*64})):
            bad=copy.deepcopy(witness);bad[part][key]=value
            with self.assertRaises(ValueError):r.validate_contradictions(node,records,[bad],lease,observations)
        # Even a coherently rehashed row/payload/receipt cannot substitute an
        # unregistered reference or trial state after the parent pins authority.
        for field in ('coordinates','q','accepted'):
            bad_records=copy.deepcopy(records);bad_row=bad_records[0]['tables']['work'][1]
            if field=='coordinates':bad_row['state'][field][0][0]=.25
            elif field=='q':bad_row['state'][field][0]=.25
            else:bad_row['state'][field][0][0][0]=.25
            bad_row['state_sha256']=sha256(r.canonical(bad_row['state'])).hexdigest()
            bad_witness=inert_contradiction(node,'work',bad_row,lease)
            with self.assertRaises(ValueError):r.validate_contradictions(node,bad_records,[bad_witness],lease,observations)
        orphan_records=inert_records(4)
        with self.assertRaises(ValueError):r.validate_contradictions(node,orphan_records,[witness],lease,observations)

    def test_observation_manifest_is_canonical_pinned_and_ordered(self):
        expected=dict(source_bindings={},generator_bindings={},capsule_sha256='a'*64,
                      inventory_sha256=sha256(r.canonical(r.observation_specs())).hexdigest())
        value=dict(schema='GE_BEAM3_REGISTERED_OBSERVATION_HASHES_V1',**expected,
            observations=[dict(spec,state_sha256='b'*64) for spec in r.observation_specs()],
            fixture_data_only=True,numerical_qualification=False)
        raw=r.canonical(value);digest=sha256(raw).hexdigest()
        with patch.object(r,'read',return_value=raw),patch.object(r,'OBSERVATION_MANIFEST_SHA',digest), \
             patch.object(r,'observation_authority',return_value=expected):
            self.assertEqual(len(r.frozen_observations()),130)
        for mode in ('omitted','reordered','source','state_hash','wrong_id','duplicate'):
            bad=copy.deepcopy(value)
            if mode=='omitted':bad['observations'].pop()
            elif mode=='reordered':bad['observations'].reverse()
            elif mode=='source':bad['capsule_sha256']='c'*64
            elif mode=='state_hash':bad['observations'][0]['state_sha256']='not-a-digest'
            elif mode=='wrong_id':bad['observations'][0]['row_id']='foreign'
            else:bad['observations'][1]=copy.deepcopy(bad['observations'][0])
            bad_raw=r.canonical(bad)
            with patch.object(r,'read',return_value=bad_raw),patch.object(r,'OBSERVATION_MANIFEST_SHA',sha256(bad_raw).hexdigest()), \
                 patch.object(r,'observation_authority',return_value=expected):
                with self.assertRaises(ValueError):r.frozen_observations()
            with patch.object(r,'read',return_value=bad_raw),patch.object(r,'OBSERVATION_MANIFEST_SHA',digest):
                with self.assertRaises(ValueError):r.frozen_observations()

    def test_numerical_watchdog_drains_all_jobs(self):
        class Timer:
            def __init__(self,*args):pass
            def start(self):pass
            def cancel(self):pass
        exits=[];watch=r.WaveWatchdog(timer=Timer,exit_process=exits.append)
        jobs=[Job() for _ in range(3)]
        for job in jobs:watch.attach(job)
        watch.expire();self.assertEqual(exits,[124]);self.assertTrue(all(j.killed for j in jobs));watch.close()

if __name__=='__main__':unittest.main()
