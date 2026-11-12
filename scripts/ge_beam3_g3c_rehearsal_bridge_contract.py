"""Inert, exact R06 correction and role-separated consumer bridge contract."""
import ast
from hashlib import sha256
from pathlib import Path
import ge_beam3_g3c_history_contract as common

ROOT=Path(__file__).resolve().parents[1]
PATH='docs/reference_cases/ge_beam3_g3c_rehearsal_consumer_bridge_v1.json'
BASE='e818ec9541d01c4de1d83e59ca096769aba79f43'
TREE='74f981544ed201d056c5dcb8e415570e1a5eb354'
EXTENT=['docs/GE_BEAM3_G3C_REHEARSAL_CONSUMER_BRIDGE_CONTRACT.md',PATH,
        'scripts/ge_beam3_g3c_rehearsal_bridge_contract.py',
        'tests/test_ge_beam3_g3c_rehearsal_bridge_contract.py']
SOURCES={
    'G1':('_ge_beam3_g1_analysis.py','SCHEMA','GE_BEAM3_G1_ELASTIC_RESTART_V1',17505,'ead8d2209f8c695bfa5cc3cec504a27a2c57ebff5e3e8957321a6420141198b8'),
    'G2':('_ge_beam3_g2_analysis.py','SCHEMA','GE_BEAM3_G2_ELASTIC_RESTART_V1',12035,'b0af46d619f3307fbf315bf3f8f6089edf916fed728be7864ec7f94f492f44a0'),
    'G3a':('_ge_beam3_g3_analysis.py','SCHEMA','GE_BEAM3_G3_GRAPH_ELASTIC_RESTART_V1',22760,'04dca633963a83c5ad14dfdd381eca789f21499894c80329f7aacd7c139ba5a9'),
    'G3b':('_ge_beam3_g3b_owner.py','SCHEMA','GE_BEAM3_G3B_REFERENCE_OWNER_RESTART_V1',14959,'1bb6f795edba24b1f50dbaba09f21f027b9417a892a6971d97b78ca25ca8d30b'),
    'legacy':('nonlinear_restart.py','NONLINEAR_CHECKPOINT_SCHEMA','ANYSOLVER_NONLINEAR_CHECKPOINT_V1',56181,'44951edf08cbf3efdcb56341ac7020025f5bf1b009912b7079c98af7abfdf137'),
}


def expected():
    return dict(schema='G3C_R06_CONSUMER_BRIDGE_CONTRACT_V1',stage='DESIGN_ONLY',
        base=dict(commit=BASE,tree=TREE),extent=EXTENT,execution_authorized=False,
        producer=dict(commit='e22be49d221685441c2e16539198e8c66a9b3ca3',tree='ab1e30f26e3fa38d4c5d296b33b3c1813534131b',
            implementation_review_sha256='915241023d685fe7edeeff007a7cfa6890ef181eed35355635ec79a16aa88d22',
            accepted_producer_review_sha256='b8ff895b4239196012bc902ca46614c429137e6332aabe3ab8497be3f160cbcf',
            inputs_sha256='30ab44195e49d724f6a67a94f0960a958963d65844378e4210b2e36f2627c1a1',inputs_count=3389,
            runtime_sha256='21d7ad87310957da81018eaa0d438effc4ff90e4b1568a42d13cf427f6e066e8',
            environment_sha256='2ce154b866565e1d03019651b1ae7fdd070e5554f38d72adf21d8080558b6756',
            archive_manifest_sha256='c67f08b3abdff5c4ed5a97b98dd2650da7e6b159f31f483f6de7cfd8cafab648',
            archive_root='C:/Users/AudunArnesenNyhus/AppData/Local/ANYrelease/ge-beam3-g3c-rehearsal-producers-20260912-e22be49',
            waves=dict(none=dict(bytes=4019,sha256='03b2fa1e9959a1eccf45beb3843b166ec48d39b882ef25f729916523744d3772'),
                       cm3=dict(bytes=4190,sha256='ff5b71c1c7021e990e57f230c5ecac37a3324ca1ae8d57f997701febb78dc109')),
            allowed_roles=['none','cm3'],rerun_authorized=False),
        consumer_scope='G3C_R06_CORRECTED_REHEARSAL_CONSUMER_V1',
        consumer_waves=['positive','preflight','virgin_review','negative-0','negative-1','negative-2','negative-3','negative-4'],
        correction_modify=['scripts/ge_beam3_g3c_rehearsal_mutations.py','tests/test_ge_beam3_g3c_rehearsal_mutations_static.py'],
        implementation_add=['scripts/ge_beam3_g3c_rehearsal_bridge.py','scripts/run_ge_beam3_g3c_rehearsal_consumer.py',
                            'tests/test_ge_beam3_g3c_rehearsal_consumer_static.py','docs/GE_BEAM3_G3C_REHEARSAL_CONSUMER_IMPLEMENTATION.md'],
        source_schemas={k:dict(path='src/anyfesolver/'+v[0],constant=v[1],literal=v[2],fingerprint=dict(bytes=v[3],sha256=v[4])) for k,v in SOURCES.items()},
        hypothetical=dict(member='predecessor_G3c',literal='GE_BEAM3_G3C_MIXED_ELASTIC_RESTART_V1',
                          disposition='HYPOTHETICAL_NOT_A_PREVIOUSLY_ACCEPTED_IMPORT_FORMAT'),
        ri03_assessment_sha256='273b8280600150db176916399f3f0f738d890263455df1cd3382e52b037c62ed',
        original_reference_policy='EXACT_SERIALIZED_REFERENCES_CHECKED_AGAINST_BOUND_ARCHIVE_MAPPING_NO_REWRITE',
        consumer_review_binding=['subject_commit','subject_tree','inputs_sha256','bridge_contract_sha256','source_map_sha256','scope_id'],
        positive_replays_required=2,negative_probes_required=142,production_activation_authorized=False,
        full_g3c_qualified=False)


def validate(value):
    if common.canonical(value)!=common.canonical(expected()): raise ValueError('exact bridge contract mismatch')
    return value


def schema_literal(raw,constant):
    tree=ast.parse(raw)
    found=[n.value for n in tree.body if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id==constant for t in n.targets)]
    if (len(found)!=1 or not isinstance(found[0],ast.Constant)
            or type(found[0].value) is not str):
        raise ValueError('unique source schema literal required')
    return found[0].value


def source_literals(root=ROOT):
    result={}
    for member,row in expected()['source_schemas'].items():
        raw=common.normalized(root/row['path'])
        if common.fingerprint(raw)!=row['fingerprint'] or schema_literal(raw,row['constant'])!=row['literal']:
            raise ValueError('historical schema source binding: '+member)
        result[member]=row['literal']
    result['predecessor_G3c']=expected()['hypothetical']['literal']
    return result


def audit():
    value=validate(common.strict(common.normalized(ROOT/PATH)))
    source_literals()
    for name,expected_hash in [
        ('ge_beam3_g3c_rehearsal_producer_review_v1.json',value['producer']['accepted_producer_review_sha256']),
        ('ge_beam3_g3c_rehearsal_producer_archive_v1.json',value['producer']['archive_manifest_sha256']),
        ('ge_beam3_g3c_rehearsal_r06_assessment_v1.json',value['ri03_assessment_sha256'])]:
        raw=common.normalized(ROOT/'docs/reference_cases'/name)
        if sha256(raw).hexdigest()!=expected_hash: raise ValueError('producer/incident authority changed')
        common.strict(raw)
    return dict(stage='DESIGN_ONLY_NO_MECHANICS',source_bound_schemas=5,hypothetical_schemas=1,
                producer_roles=2,positive_replays=2,negative_probes=142)


if __name__=='__main__': print(common.canonical(audit()).decode(),end='')
