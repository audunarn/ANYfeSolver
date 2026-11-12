"""Inert exact-arithmetic source audit. No numerical environment required."""
import ast
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_ge_beam3_g3c_mo16_obstruction as a


class ObstructionTests(unittest.TestCase):
    def test_source_bindings(self):
        self.assertEqual(set(a.sources()), set(a.BINDINGS))

    def test_exact_clamped_contradiction(self):
        for row in a.audit()['clamped_witnesses']:
            self.assertEqual(F(row['relative_energy_defect']), F(2388, 13))
            self.assertGreater(F(row['energy_defect']), 0)
            self.assertFalse(row['physical_energy_identity'])

    def test_threshold_and_unclamped_controls(self):
        self.assertTrue(all(x['physical_energy_identity'] and not x['clamp_active']
                            for x in a.audit()['unclamped_controls']))

    def test_cancellation_of_axial_strain(self):
        for x in a.audit()['clamped_witnesses']:
            self.assertEqual(F(x['axial_displacement']) + F(x['displacement'])**2/2, 0)

    def test_reject_arbitrary_expression(self):
        with self.assertRaises(ValueError):
            a.rational(ast.parse('open(1)', mode='eval').body, 'open(1)', {})

    def test_no_production_or_numerical_imports(self):
        nodes = ast.walk(ast.parse(Path(a.__file__).read_text()))
        imports = [n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {n.module if isinstance(n, ast.ImportFrom) else n.names[0].name for n in imports}
        self.assertEqual(names, {'ast', 'fractions', 'hashlib', 'json', 'pathlib'})

    def test_determinism_and_no_qualification(self):
        self.assertEqual(a.canonical(a.audit()), a.canonical(a.audit()))
        self.assertFalse(a.audit()['full_g3c_qualified'])

    def test_source_floor_mutation_changes_witness(self):
        source = a.sources()['src/anyfesolver/elements.py']
        original = a.witness(source)
        changed = a.witness(source.replace('_SMALL = 1.0e-12', '_SMALL = 1.0e-30'))
        self.assertNotEqual(original, changed)
        self.assertTrue(changed['physical_energy_identity'])


if __name__ == '__main__':
    unittest.main()
