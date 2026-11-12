"""Standard-library-only history/restart contract tests."""
import copy
from hashlib import sha256
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import ge_beam3_g3c_history_contract as contract


class HistoryContractTests(unittest.TestCase):
    def setUp(self):
        self.c = contract.strict(contract.normalized(ROOT/contract.CONTRACT))

    def test_complete_matrix_and_prefix_counts(self):
        rows = contract.matrix()
        self.assertEqual(len(rows), 375)
        self.assertEqual(len({r['case_id'] for r in rows}), 375)
        self.assertEqual(sum(r['accepted_stages'] for r in rows), 3075)
        self.assertEqual(len(contract.prefixes()), 3450)
        self.assertEqual({(r['graph'],r['variant']) for r in rows},
                         {(g,v) for g in contract.GRAPHS for v in contract.VARIANTS})
        self.assertEqual(contract.audit(self.c)['history_probes'], 375)

    def test_genuine_preparation_and_each_prefix(self):
        for row in contract.matrix():
            commands = contract.commands(row['common_motion'], row['force_scale'])
            count = 0 if row['common_motion'] == 'NONE' else 4
            self.assertEqual(commands[:count], [dict(kind='PREPARE_COMMON_MOTION',step=i) for i in range(1,count+1)])
            self.assertEqual([r['root_stage'] for r in commands[count:]], list(range(5)))
            self.assertEqual(sha256(contract.canonical(commands)).hexdigest(), row['commands_sha256'])
        for bad in ('UNREGISTERED', ''):
            with self.assertRaises(ValueError):
                contract.commands(bad, .01)
        with self.assertRaises(ValueError):
            contract.commands('NONE', 1)

    def test_canonical_and_matrix_mutations_rejected(self):
        for raw in (b'{"a":1,"a":1}\n',b'{"a":NaN}\n',b'{"a":1e999}\n',b'{ "a":1}\n'):
            with self.assertRaises(ValueError): contract.strict(raw)
        for field, value in (('execution_authorized',True), ('prefix_matrix_sha256','0'*64)):
            changed=copy.deepcopy(self.c); changed[field]=value
            with self.assertRaises(ValueError): contract.validate_contract(changed)
        changed=copy.deepcopy(self.c); changed['history_matrix'].pop()
        with self.assertRaises(ValueError): contract.validate_contract(changed)
        changed=copy.deepcopy(self.c); changed['history_matrix'][0]['force_scale']=.001
        with self.assertRaises(ValueError): contract.validate_contract(changed)

    def test_bounds_hashes_and_inherited_scope(self):
        for key, value in (('child_seconds',601),('child_seconds',600.0),('numerical_threads',True),
                           ('restart_bytes',9*1024**2),('automatic_retry',True)):
            changed=copy.deepcopy(self.c); changed['limits'][key]=value
            with self.assertRaises(ValueError): contract.validate_contract(changed)
        changed=copy.deepcopy(self.c); changed['bindings'][0]['blob']='0'*40
        with self.assertRaises(ValueError): contract.audit(changed)
        changed=copy.deepcopy(self.c); changed['obligations'].pop()
        with self.assertRaises(ValueError): contract.audit(changed)
        changed=copy.deepcopy(self.c); changed['checkpoint_layout']['extra']='optional'
        with self.assertRaises(ValueError): contract.validate_contract(changed)
        changed=copy.deepcopy(self.c); changed['mutation_categories'].pop()
        with self.assertRaises(ValueError): contract.validate_contract(changed)
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)


if __name__ == '__main__':
    unittest.main()
