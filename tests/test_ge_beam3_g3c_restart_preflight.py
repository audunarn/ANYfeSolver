"""Inert schema fixtures are fabricated syntax, NEVER scientific evidence."""
import copy
from hashlib import sha256
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import ge_beam3_g3c_restart_preflight as p


def zeros(*shape):
    return [zeros(*shape[1:]) for _ in range(shape[0])] if shape else 0.


def eye():
    return [[float(i == j) for j in range(3)] for i in range(3)]


def syntax_packet(graph='J_B2_PAIR', variant='BASE'):
    _, source = p.authorities()
    definition, expanded, _ = source.expand(graph, variant, 'NONE')
    definition['policy_sha256'] = p.MAP_SHA
    g = expanded['graph']; n = len(g['nodes']); native = []; adapters = []
    for e in g['elements']:
        if e['family'] == 'NATIVE':
            response = dict(correction=zeros(24), full=dict(conservative=True, hessian=zeros(42,42),
                jacobian=zeros(42,42), potential=0., residual=zeros(42)), history=[], internal_error=0.,
                iterations=0, lift=zeros(24,18), residual=zeros(18), resultants=zeros(18),
                rotations=[eye(),eye()], tangent=zeros(18,18))
            payload = dict(schema='GE_BEAM3_G1_ELASTIC_STATE_V1', element_identity='1'*64,
                epoch=0, committed_total_u=zeros(18), committed_nodal_rotation_matrices=[eye() for _ in range(3)],
                line=zeros(3), couple=zeros(3), previous_state_sha256=None, response=response, history=[])
            payload['state_sha256'] = p.digest(payload)
            native.append(dict(element_id=e['id'], payload=payload))
        else:
            adapters.append(dict(element_id=e['id'], policy=expanded['local_policies'][e['family']],
                definition_sha256='2'*64, epoch=0, previous_sha256=None, origin_sha256=None,
                pose_sha256=p.digest(dict(node_ids=e['nodes'], total_translations=zeros(len(e['nodes']),3),
                                          rotations=[eye() for _ in e['nodes']])),
                deformation=zeros(6*len(e['nodes'])), diagnostic_sha256='3'*64,
                source_material_committed=False, physical_recovery_complete=False))
    state = dict(schema='GE_BEAM3_G3C_MIXED_ELASTIC_STATE_V1', epoch=0,
        definition_sha256=p.digest(definition), previous_sha256=None, total_u=zeros(6*n),
        rotations=[eye() for _ in range(n)], native_rows=native, adapter_rows=adapters,
        multipliers=zeros(6*(len(g['fixed_nodes'])+len(g['joints']))))
    return dict(schema='GE_BEAM3_G3C_STABLE_GRAPH_RESTART_V1',
        policy='GE_BEAM3_G3C_STABLE_MIXED_ELASTIC_OWNER_V1', definition=definition,
        definition_sha256=p.digest(definition), final_state=state, history=[],
        initial_sha256=p.digest(state), final_sha256=p.digest(state), runtime_sha256='4'*64)


def rehash(packet):
    for row in packet['final_state']['native_rows']:
        v = row['payload']; v['state_sha256'] = p.digest({k:x for k,x in v.items() if k != 'state_sha256'})
    packet['final_sha256'] = p.digest(packet['final_state'])
    if not packet['history']: packet['initial_sha256'] = packet['final_sha256']
    return packet


def check(packet):
    raw = p.canonical(packet)
    return p.preflight(raw, sha256(raw).hexdigest(), expected_runtime_sha256='4'*64)


class PreflightTests(unittest.TestCase):
    def test_all_graph_variants_have_inert_layout_coverage(self):
        for graph in ('J_B2_PAIR','J_B3_PAIR','J_Q4_PAIR','J_S3_PAIR','J_MULTIFAMILY_LOOP'):
            for variant in ('BASE','SHUFFLED_INSERTION','RENUMBERED','CONNECTIVITY_REVERSED','PROPER_GLOBAL_TRANSFORM'):
                with self.subTest(graph=graph,variant=variant):
                    result = check(syntax_packet(graph,variant))
                    self.assertEqual(result.epoch,0)
                    self.assertEqual(result.fixture_id,graph)
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)

    def test_external_digest_canonical_and_byte_bounds(self):
        raw = p.canonical(syntax_packet())
        for changed in (raw+b' ', b'{"a":1,"a":2}\n', b'{"a":NaN}\n',
                        b'{"a":1e999}\n', b' '* (8*1024**2+1)):
            with self.assertRaises(ValueError):
                p.preflight(changed,sha256(changed).hexdigest(),expected_runtime_sha256='4'*64)
        with self.assertRaises(ValueError): p.preflight(raw,'0'*64,expected_runtime_sha256='4'*64)
        with self.assertRaises(ValueError): p.preflight(raw,sha256(raw).hexdigest().upper(),expected_runtime_sha256='4'*64)

    def test_nested_rehashed_mutations_fail_before_construction(self):
        original = syntax_packet()
        mutations = [
            lambda s:s.update(epoch=True),
            lambda s:s['total_u'].__setitem__(0,0),
            lambda s:s['rotations'][0][0].__setitem__(0,-1.),
            lambda s:s['native_rows'][0]['payload']['response']['full'].update(extra=0.),
            lambda s:s['native_rows'][0]['payload']['response']['full'].update(residual=zeros(41)),
            lambda s:s['native_rows'][0]['payload']['line'].__setitem__(0,1.),
            lambda s:s['native_rows'][0]['payload'].update(epoch=1),
            lambda s:s['native_rows'][0].update(element_id=True),
            lambda s:s['adapter_rows'][0].update(physical_recovery_complete=True),
            lambda s:s['adapter_rows'][0].update(source_material_committed=True),
            lambda s:s['adapter_rows'][0].update(policy='unregistered'),
            lambda s:s['adapter_rows'][0].update(deformation=zeros(1)),
            lambda s:s['adapter_rows'][0].update(pose_sha256='0'*64),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                packet=copy.deepcopy(original); mutate(packet['final_state']); rehash(packet)
                with self.assertRaises(ValueError): check(packet)

    def test_constructor_derived_commitments_are_explicitly_not_proven(self):
        # Fake numerical data intentionally passes syntax: ONLY real replay may
        # authenticate these commitments. This result cannot construct an owner.
        packet=syntax_packet()
        packet['final_state']['native_rows'][0]['payload']['element_identity']='a'*64
        packet['final_state']['adapter_rows'][0]['definition_sha256']='b'*64
        result=check(rehash(packet))
        self.assertEqual(result.raw,p.canonical(packet))
        with self.assertRaises(AttributeError): result.epoch=1

    def test_journal_hash_and_command_consistency(self):
        packet=syntax_packet(); state=packet['final_state']; original=packet['initial_sha256']
        state['epoch']=1; state['previous_sha256']=original
        for row in state['native_rows']:
            row['payload']['epoch']=1; row['payload']['previous_state_sha256']='5'*64
        for row in state['adapter_rows']:
            row.update(epoch=1,origin_sha256=original,previous_sha256='6'*64)
        packet['history']=[dict(epoch=1,previous_entry_sha256=None,origin_sha256=original,
            accepted_state_sha256='0'*64,command=dict(kind='LOAD_STAGE',root_stage=0,load_factor=0.,force_scale=.01))]
        rehash(packet); packet['history'][0]['accepted_state_sha256']=packet['final_sha256']
        self.assertEqual(check(packet).epoch,1)
        for key, value in (('epoch',True),('origin_sha256','0'*64),('previous_entry_sha256','0'*64)):
            bad=copy.deepcopy(packet); bad['history'][0][key]=value
            with self.assertRaises(ValueError): check(bad)
        bad=copy.deepcopy(packet); bad['history'][0]['command']['root_stage']=1
        with self.assertRaises(ValueError): check(bad)

    def test_source_runtime_and_layout_authority(self):
        for key,value in (('schema','GE_BEAM3_G3B_RESTART_V1'),('runtime_sha256','0'*64),
                          ('policy','legacy'),('unregistered',False)):
            packet=syntax_packet(); packet[key]=value
            with self.assertRaises(ValueError): check(packet)
        packet=syntax_packet(); packet['definition']['fixture_sha256']='0'*64
        with self.assertRaises(ValueError): check(packet)

    def test_signed_zero_native_copies_are_exact(self):
        for field in ('committed_total_u','committed_nodal_rotation_matrices'):
            packet=syntax_packet(); payload=packet['final_state']['native_rows'][0]['payload']
            if field == 'committed_total_u': payload[field][0]=-0.
            else: payload[field][0][0][1]=-0.
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'native graph pose'):
                check(rehash(packet))

    def test_self_loop_and_multiple_epoch_state_cycles_rejected(self):
        for count in (2,3):
            packet=syntax_packet(); state=packet['final_state']; initial='7'*64
            packet['initial_sha256']=initial
            state['epoch']=count
            for row in state['native_rows']:
                row['payload']['epoch']=count; row['payload']['previous_state_sha256']='5'*64
            history=[]; origin=initial; previous=None
            for i in range(count):
                accepted = initial if i == count-2 else ('8'*64 if i == 0 else '0'*64)
                entry=dict(epoch=i+1, previous_entry_sha256=previous, origin_sha256=origin,
                    accepted_state_sha256=accepted,
                    command=dict(kind='LOAD_STAGE',root_stage=i,load_factor=(0.,.5,1.)[i],force_scale=.01))
                history.append(entry); previous=p.digest(entry); origin=entry['accepted_state_sha256']
            state['previous_sha256']=history[-1]['origin_sha256']
            for row in state['adapter_rows']:
                row.update(epoch=count,origin_sha256=state['previous_sha256'],previous_sha256='6'*64)
            packet['history']=history; rehash(packet)
            history[-1]['accepted_state_sha256']=packet['final_sha256']
            with self.subTest(count=count), self.assertRaisesRegex(ValueError,'cyclic state commitments'):
                check(packet)


if __name__ == '__main__': unittest.main()
