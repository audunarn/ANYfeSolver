"""Fast source-only regression checks; no solver or scientific worker runs."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('integration_audit', ROOT/'docs/reference_cases/ge_beam3_integration_boundary_audit.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class IntegrationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.sources = {key: (ROOT/path).read_bytes() for key, path in AUDIT.SOURCES.items()}
        self.evidence = {key: (ROOT/data[0]).read_bytes() for key, data in AUDIT.EVIDENCE.items()}

    def change(self, role, old, new):
        self.assertIn(old, self.sources[role])
        self.sources[role] = self.sources[role].replace(old, new)

    def reject(self):
        with self.assertRaises(AUDIT.BoundaryError):
            AUDIT.audit(self.sources, self.evidence)

    def test_current_boundary_and_determinism(self):
        first = AUDIT.audit(self.sources, self.evidence)
        self.assertEqual(first, AUDIT.audit(self.sources, self.evidence))
        self.assertFalse(first['production_activation_authorized'])
        self.assertFalse(first['scientific_execution'])
        self.assertEqual(first['native_external_dofs'], 18)
        self.assertEqual(len(first['inventory']), 9)
        self.assertEqual(len(first['unresolved_programme']), 7)

    def test_factory_redirect_rejected(self):
        self.change('factory', b'from .ge_beam3_element import GeometricallyExactBeam3D3NElement',
                    b'from ._ge_beam3_native_generalized_element import GeometricallyExactBeam3D3NElement')
        self.reject()

    def test_factory_constructor_rejected(self):
        self.change('factory', b'return GeometricallyExactBeam3D3NElement(', b'return UnqualifiedElement(')
        self.reject()

    def test_old_formulation_relabel_rejected(self):
        self.change('facade_state', b'GE_BEAM3_DC_MIXED_K1_MACRO_V2', b'GE_BEAM3_CURVED_NEW')
        self.reject()

    def test_capability_gap_removal_rejected(self):
        self.change('facade', b'"curved_reference",', b'"curved_reference_unconditionally_supported",')
        self.reject()

    def test_premature_private_activation_rejected(self):
        self.change('native', b'production_qualified=False', b'production_qualified=True')
        self.reject()

    def test_static_mass_route_rejected(self):
        self.change('native', b'compute_mass_matrix=_unsupported', b'compute_mass_matrix=compute_stiffness_matrix')
        self.reject()

    def test_inertia_condensation_policy_rejected(self):
        self.change('modal', b'CENTERED_LIFTED_CELL_INERTIA_24_POINT_NO_STATIC_REDUCTION', b'STATIC_GUYAN')
        self.reject()

    def test_modal_history_policy_rejected(self):
        self.change('modal', b'COMMITTED_HISTORY_ELASTIC_INTERIOR_NO_ADVANCE', b'ADVANCE_HISTORY')
        self.reject()

    def test_status_mutation_rejected(self):
        for role in AUDIT.EVIDENCE:
            with self.subTest(role=role):
                original = self.evidence[role]
                self.evidence[role] = original + b' '
                self.reject()
                self.evidence[role] = original

    def test_missing_inventory_rejected(self):
        del self.sources['arc']
        self.reject()

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
                    b'{"a":1e999}', b'{"a":-1e999}', b'[]'):
            with self.subTest(raw=raw), self.assertRaises(AUDIT.BoundaryError):
                AUDIT.strict_json(raw)

    def test_no_mechanics_imported_in_standalone_run(self):
        # Under a wider pytest session unrelated tests may have imported these.
        if __name__ == '__main__':
            for prefix in ('anyfesolver', 'numpy', 'scipy', 'sympy'):
                self.assertFalse(any(n == prefix or n.startswith(prefix+'.') for n in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
