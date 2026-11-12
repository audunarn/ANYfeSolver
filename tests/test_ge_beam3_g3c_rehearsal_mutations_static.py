"""Fabricated syntax only; never a numerical accepted state or evidence."""
import copy
import ast
from hashlib import sha256
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import ge_beam3_g3c_rehearsal_mutations as m
import test_ge_beam3_g3c_restart_preflight as syntax
p=m.p


def fake_origin(graph,motion,scale,prefix):
    value=syntax.syntax_packet(graph); state=value['final_state']
    value['definition']['common_motion']=motion
    value['definition_sha256']=p.digest(value['definition']); state['definition_sha256']=value['definition_sha256']
    value['initial_sha256']=p.digest(state)
    origin=value['initial_sha256']; entries=[]
    for i,command in enumerate(m.design.inherited.commands(motion,scale)[:prefix]):
        accepted=sha256(('SYNTHETIC ONLY '+str(i)).encode()).hexdigest()
        entries.append(dict(epoch=i+1,command=command,origin_sha256=origin,
                            accepted_state_sha256=accepted,previous_entry_sha256=None))
        origin=accepted
    value['history']=entries; state['epoch']=prefix; state['previous_sha256']=entries[-1]['origin_sha256']
    for row in state['native_rows']:
        row['payload'].update(epoch=prefix,previous_state_sha256='5'*64)
    for row in state['adapter_rows']:
        row.update(epoch=prefix,origin_sha256=state['previous_sha256'],previous_sha256='6'*64)
    return p.canonical(m.repair(value))


class MutationStaticTests(unittest.TestCase):
    def test_r06_emits_actual_source_schema_at_both_origins(self):
        contract=m.bridge_contract.expected()
        for graph,motion,scale,prefix in [('J_B2_PAIR','NONE',.01,2),('J_MULTIFAMILY_LOOP','CM3',10.,9)]:
            raw=fake_origin(graph,motion,scale,prefix)
            for member,row in contract['source_schemas'].items():
                source=(ROOT/row['path']).read_bytes().replace(b'\r\n',b'\n')
                self.assertEqual(dict(bytes=len(source),sha256=sha256(source).hexdigest()),row['fingerprint'])
                assignments=[n.value.value for n in ast.parse(source).body
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==row['constant'] for t in n.targets)]
                self.assertEqual(assignments,[row['literal']])
                changed,record=m.packet_attack(raw,'R06_OLD_SCHEMA',member)
                self.assertEqual(p.strict(changed)['schema'],assignments[0])
                self.assertEqual(record['expected_error'],'literal schema')
            changed,_=m.packet_attack(raw,'R06_OLD_SCHEMA','predecessor_G3c')
            self.assertEqual(p.strict(changed)['schema'],contract['hypothetical']['literal'])

    def test_every_packet_member_has_exact_inert_behavior(self):
        for graph,motion,scale,prefix in [('J_B2_PAIR','NONE',.01,2),('J_MULTIFAMILY_LOOP','CM3',10.,9)]:
            raw=fake_origin(graph,motion,scale,prefix)
            p.preflight(raw,sha256(raw).hexdigest(),expected_runtime_sha256='4'*64)
            for category in m.design.parent()['mutation_categories']:
                for member in category['members']:
                    if category['id']=='R10_NORMAL_SOURCE' or member=='changed_implementation_review': continue
                    with self.subTest(origin=graph,category=category['id'],member=member):
                        changed,record=m.packet_attack(raw,category['id'],member)
                        self.assertNotEqual(changed,raw)
                        if record['rejection']=='PREFLIGHT_BEFORE_CONSTRUCTION':
                            with self.assertRaises(ValueError) as got:
                                p.preflight(changed,record['after_sha256'],expected_runtime_sha256='4'*64)
                            self.assertEqual(str(got.exception),record['expected_error'])
                        else:
                            p.preflight(changed,record['after_sha256'],expected_runtime_sha256='4'*64)
            self.assertEqual(raw,fake_origin(graph,motion,scale,prefix))
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)

    def test_deterministic_and_forbidden_members(self):
        raw=fake_origin('J_B2_PAIR','NONE',.01,2)
        a=m.packet_attack(raw,'R20_NATIVE_POSE','proper_but_wrong_Q')
        self.assertEqual(a,m.packet_attack(raw,'R20_NATIVE_POSE','proper_but_wrong_Q'))
        with self.assertRaises(ValueError): m.packet_attack(raw,'unknown','unknown')
        with self.assertRaises(ValueError): m.packet_attack(raw,'R09_RUNTIME','changed_implementation_review')
        self.assertEqual(p.strict(a[0])['history'][:-1],p.strict(raw)['history'][:-1])

    def test_null_beam_candidate_attack_is_deterministic(self):
        raw=fake_origin('J_B2_PAIR','NONE',.01,2)
        value=p.strict(raw)
        adapter=value['final_state']['adapter_rows'][0]
        adapter['diagnostic_sha256']=None
        raw=p.canonical(m.repair(value))

        changed,record=m.packet_attack(raw,'R23_ADAPTER_DIAGNOSTIC','candidate_sha')

        self.assertEqual(p.strict(changed)['final_state']['adapter_rows'][0]['diagnostic_sha256'],'0'*64)
        self.assertEqual(record['expected_error'],'genuine replay prefix mismatch: 2')
        self.assertEqual(record['rejection'],'GENUINE_REPLAY_MISMATCH')
        self.assertEqual(record,m.packet_attack(raw,'R23_ADAPTER_DIAGNOSTIC','candidate_sha')[1])


if __name__=='__main__': unittest.main()
