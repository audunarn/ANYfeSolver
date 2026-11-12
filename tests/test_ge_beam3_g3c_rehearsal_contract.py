"""Inert inventory tests; these are not numerical qualification evidence."""
import ast
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import ge_beam3_g3c_rehearsal_contract as c


class RehearsalContractTests(unittest.TestCase):
    def test_history_coverage_and_origins(self):
        value = c.expected()
        rows = value['histories']
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(r['accepted_stages'] for r in rows), 70)
        self.assertEqual(sum(r['accepted_stages']+1 for r in rows), 80)
        self.assertEqual({r['graph'] for r in rows}, set(c.inherited.GRAPHS))
        for origin in c.parent()['mutation_origins']:
            row = next(r for r in rows if r['case_id'] == origin['case_id'])
            self.assertLessEqual(origin['prefix'], row['accepted_stages'])

    def test_all_members_and_rejection_inventory(self):
        probes = c.expected()['mutation_probes']
        self.assertEqual(len(probes), 142)
        self.assertEqual(len({(r['origin']['case_id'],r['category'],r['member']) for r in probes}), 142)
        self.assertEqual(sum(r['rejection']=='GENUINE_REPLAY_MISMATCH' for r in probes), 26)
        self.assertEqual(sum(r['rejection']=='VIRGIN_INITIAL_HASH_MISMATCH' for r in probes), 2)
        self.assertEqual(sum(r['rejection']=='RUNNER_AUTHORITY_BEFORE_CONSTRUCTION' for r in probes), 2)
        self.assertEqual(sum(r['rejection']=='PREFLIGHT_BEFORE_CONSTRUCTION' for r in probes), 112)

    def test_strict_schema_and_mutations(self):
        value = c.expected()
        for key, replacement in [('execution_authorized',True),('full_matrix_replaced',True),
                                 ('unknown',None),('mutation_noop_allowed',True)]:
            bad = copy.deepcopy(value); bad[key] = replacement
            with self.assertRaises(ValueError): c.validate(bad)
        for key in ('histories','mutation_probes'):
            bad=copy.deepcopy(value); bad[key].pop()
            with self.assertRaises(ValueError): c.validate(bad)
        bad=copy.deepcopy(value); bad['limits']['max_workers']=3.0
        with self.assertRaises(ValueError): c.validate(bad)
        for raw in (b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{"x":1e999}\n'):
            with self.assertRaises(ValueError): c.inherited.strict(raw)

    def test_inert_authority_and_imports(self):
        self.assertEqual(c.audit()['stage'], 'DESIGN_ONLY_NO_MECHANICS')
        tree=ast.parse((ROOT/c.EXTENT[2]).read_text())
        imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports.update(a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names)
        self.assertEqual(imports, {'hashlib','pathlib','os','subprocess','ge_beam3_g3c_history_contract'})
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('anyfesolver',sys.modules)


if __name__ == '__main__':
    unittest.main()
