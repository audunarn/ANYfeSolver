"""Static prerequisite evidence, not S01-S08 mechanics qualification."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/reference_cases/ge_beam3_g1_elastic_admission_v1.json"


def source(name):
    return (ROOT / "src/anyfesolver" / name).read_text(encoding="utf-8")


def has_exact_type_guard(name, argument, required_type):
    tree = ast.parse(source(name))
    return any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Name) and node.left.func.id == "type"
        and len(node.left.args) == 1
        and isinstance(node.left.args[0], ast.Name) and node.left.args[0].id == argument
        and len(node.ops) == 1 and isinstance(node.ops[0], ast.IsNot)
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == required_type
        for node in ast.walk(tree)
    )


class G1ElasticAdmissionTests(unittest.TestCase):
    def test_record_is_canonical_and_not_a_qualification(self):
        raw = RECORD.read_bytes().replace(b"\r\n", b"\n")
        record = json.loads(raw)
        self.assertEqual(raw, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
        self.assertEqual(record["terminal"], "BLOCKED_GE_BEAM3_STATIC_AUTHORITY")
        self.assertEqual(record["reason"], "EXACT_ELASTIC_EXTENSION_REQUIRED")
        self.assertEqual(record["unexecuted_cases"], [f"S{i:02d}" for i in range(1, 9)])
        for key in ("g1_complete", "scientific_nogo", "mechanics_changed", "successor_authorized"):
            self.assertIs(record[key], False)

    def test_source_and_contract_bindings(self):
        record = json.loads(RECORD.read_text())
        for row in record["bindings"]:
            with self.subTest(path=row["path"]):
                data = (ROOT / row["path"]).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(len(data), row["bytes_utf8_lf"])
                self.assertEqual(sha256(data).hexdigest(), row["sha256_utf8_lf"])

    def test_generalized_operator_and_cell_require_concrete_plastic_law(self):
        for path in ("_ge_beam3_retained_generalized.py", "_ge_beam3_generalized_cell.py"):
            self.assertTrue(has_exact_type_guard(path, "section", "EllipsoidalGeneralizedSection"))

    def test_static_reduction_requires_existing_operator_and_history(self):
        path = "_ge_beam3_generalized_static_boundary.py"
        self.assertTrue(has_exact_type_guard(path, "operator", "RetainedGeneralizedOperator"))
        self.assertTrue(has_exact_type_guard(path, "origin", "GeneralizedCellHistory"))

    def test_ellipsoid_is_finite_yield_not_unconditional_elastic(self):
        law = source("_ge_beam3_generalized_ellipsoid_section.py")
        self.assertIn("self.yield_force=_number(yield_force)", law)
        self.assertIn("positive yield and hardening required", law)
        self.assertIn("lam=max(D(0),(q-radius)/h)", law)
        self.assertIn("if not isfinite(result):", source("_ge_beam3_fibre_section.py"))

    def test_fibre_background_cannot_be_an_empty_elastic_adapter(self):
        law = source("_ge_beam3_fibre_section.py")
        self.assertIn("not 1 <= len(self.fibres) <= 1024", law)
        self.assertIn("if self.area <= 0 or self.young <= 0:", law)
        self.assertIn("explicit fibre ID and energy curve required", law)

    def test_old_elastic_probe_is_not_an_admitted_substitute(self):
        self.assertTrue(has_exact_type_guard("_ge_beam3_centered_mixed.py", "section", "DirectedHardeningSection"))
        probe = source("_ge_beam3_retained_elastic.py")
        self.assertIn("distributed load not implemented in elastic probe", probe)
        self.assertIn("nonlinear section branch not implemented in elastic probe", probe)

    def test_definition_dispatch_has_no_linear_elastic_family(self):
        tree = ast.parse(source("_ge_beam3_native_definition.py"))
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build")
        families = set()
        for node in ast.walk(build):
            if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript)
                    and isinstance(node.left.value, ast.Name) and node.left.value.id == "data"
                    and isinstance(node.left.slice, ast.Constant) and node.left.slice.value == "family"):
                families.update(n.value for n in node.comparators if isinstance(n, ast.Constant))
        self.assertEqual(families, {"RESULTANT_ELLIPSOID", "PHYSICAL_AXIAL_BIAXIAL_FIBRE"})


if __name__ == "__main__":
    unittest.main()
