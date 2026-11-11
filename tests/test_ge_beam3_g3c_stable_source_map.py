"""Standard-library-only design audit. No copied mechanics are imported."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_map', ROOT/'scripts/ge_beam3_g3c_stable_map.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SourceMapTests(unittest.TestCase):
    def setUp(self):
        self.contract = m.strict(m.normalized(ROOT/m.MAP))

    def test_exact_preview_and_inventory(self):
        self.assertEqual(m.audit(self.contract)['copies'], 13)
        self.assertFalse(self.contract['implementation_authorized'])
        self.assertFalse(self.contract['execution_authorized'])
        for row in self.contract['copies']:
            self.assertFalse((ROOT/row['destination']).exists())
            self.assertEqual(m.generated(row, self.contract['module_map']),
                             m.generated(row, self.contract['module_map']))

    def test_strict_json(self):
        for raw in (b'{"a":1,"a":1}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n', b'{ "a":1}\n'):
            with self.assertRaises((ValueError, UnicodeError)):
                m.strict(raw)

    def test_mutations(self):
        value = copy.deepcopy(self.contract)
        value['copies'][0]['source_blob'] = '0'*40
        with self.assertRaises(ValueError):
            m.audit(value)
        for field in ('source_fingerprint', 'generated_fingerprint'):
            value = copy.deepcopy(self.contract)
            value['copies'][0][field]['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                m.audit(value)
        value = copy.deepcopy(self.contract)
        value['copies'][0]['extra'] = True
        with self.assertRaises(ValueError):
            m.audit(value)
        value = copy.deepcopy(self.contract)
        value['copies'][1]['destination'] = value['copies'][0]['destination']
        with self.assertRaises(ValueError):
            m.audit(value)
        value = copy.deepcopy(self.contract)
        value['module_map']['anyfesolver._ge_beam3_g1_operator'] = 'anyfesolver.forbidden'
        with self.assertRaises(ValueError):
            m.audit(value)

    def test_derivative_only_import_split(self):
        result = m.imports('from ._ge_beam3_mixed_ad import Jet2, so3_exp, rotation_log\n',
                           'anyfesolver.owner', {})
        self.assertIn('from anyfesolver._ge_beam3_mixed_ad import Jet2', result)
        self.assertIn('from anyfesolver._ge_beam3_g3c_so3_numerics import so3_exp', result)
        self.assertIn('from anyfesolver._ge_beam3_mixed_ad import rotation_log', result)
        with self.assertRaises(ValueError):
            m.imports('import anyfesolver.old\n', 'anyfesolver.owner', {'anyfesolver.old':'anyfesolver.new'})


if __name__ == '__main__':
    unittest.main()
