"""Inert source-bound bridge contract tests, never numerical evidence."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import ge_beam3_g3c_rehearsal_bridge_contract as c


class BridgeContractTests(unittest.TestCase):
    def test_exact_five_sources_and_hypothetical(self):
        values=c.source_literals()
        self.assertEqual(len(values),6)
        self.assertEqual(values['legacy'],'ANYSOLVER_NONLINEAR_CHECKPOINT_V1')
        self.assertEqual(values['G1'],'GE_BEAM3_G1_ELASTIC_RESTART_V1')
        self.assertEqual(c.expected()['hypothetical']['disposition'],'HYPOTHETICAL_NOT_A_PREVIOUSLY_ACCEPTED_IMPORT_FORMAT')

    def test_unique_literal_without_import(self):
        self.assertEqual(c.schema_literal(b'SCHEMA="x"\n','SCHEMA'),'x')
        for raw in (b'SCHEMA=make()\n',b'SCHEMA="x"\nSCHEMA="y"\n',
                    b'SCHEMA="x"\nSCHEMA=make()\n',b'OTHER="x"\n'):
            with self.assertRaises(ValueError): c.schema_literal(raw,'SCHEMA')
        self.assertNotIn('numpy',sys.modules); self.assertNotIn('anyfesolver',sys.modules)

    def test_role_hash_and_scope_mutations(self):
        for key in ('commit','tree','implementation_review_sha256','inputs_sha256','runtime_sha256','archive_manifest_sha256'):
            v=copy.deepcopy(c.expected()); v['producer'][key]='wrong'
            with self.assertRaises(ValueError): c.validate(v)
        for key,value in [('execution_authorized',True),('positive_replays_required',False),('consumer_scope','old_scope')]:
            v=copy.deepcopy(c.expected()); v[key]=value
            with self.assertRaises(ValueError): c.validate(v)
        v=copy.deepcopy(c.expected()); v['consumer_waves'].insert(0,'none')
        with self.assertRaises(ValueError): c.validate(v)

    def test_schema_and_correction_extent_mutations(self):
        v=copy.deepcopy(c.expected()); v['source_schemas']['G2']['literal']='GE_BEAM3_G2_RESTART_V1'
        with self.assertRaises(ValueError): c.validate(v)
        v=copy.deepcopy(c.expected()); v['correction_modify'].append('src/anyfesolver/_ge_beam3_g3c_stable/owner.py')
        with self.assertRaises(ValueError): c.validate(v)
        for raw in (b'{"a":1,"a":2}\n',b'{"a":NaN}\n',b'{"a":1e999}\n'):
            with self.assertRaises(ValueError): c.common.strict(raw)

    def test_bound_producer_and_incident_authority(self):
        self.assertEqual(c.audit(),dict(stage='DESIGN_ONLY_NO_MECHANICS',source_bound_schemas=5,
            hypothetical_schemas=1,producer_roles=2,positive_replays=2,negative_probes=142))


if __name__=='__main__': unittest.main()
