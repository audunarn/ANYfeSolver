"""Source-only identity and refusal checks for frozen c2/P0 owner parity."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
parity = importlib.import_module("r8_f03_current_owner_parity_dev")


class FrozenOwnerSourceTests(unittest.TestCase):
    def git_history_available(self):
        """The c64d189e and OLD_PRODUCT objects exist in this checkout."""
        if not (ROOT / ".git").exists():
            return False
        for ref in ("c64d189e", parity.OLD_PRODUCT):
            probe = subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"],
                                   cwd=ROOT, capture_output=True)
            if probe.returncode != 0:
                return False
        return True

    def test_both_representative_preflights(self):
        # Historical refusal: the source guard binds the c64d189e input
        # state and the GE owner/assembly sources changed on main after
        # it, so both representative preflights must refuse fail-closed.
        # A git-free source export refuses the same preflight fail-closed
        # without the committed objects.
        for level in (32, 64):
            with self.subTest(level=level):
                if self.git_history_available():
                    with self.assertRaises(ValueError) as raised:
                        parity.preflight(level)
                    self.assertIn(
                        "current GE owner/source dependencies changed since c64d189e",
                        str(raised.exception))
                else:
                    with self.assertRaises((ValueError, OSError,
                                            subprocess.CalledProcessError)):
                        parity.preflight(level)
        # Reachable positive controls: the frozen c2/P0 cases and the
        # frozen native beam assembly itself remain unchanged.
        for level, rows in ((32, 1632), (64, 5824)):
            with self.subTest(level=level):
                frozen = parity.check_frozen_case(*parity.LEVELS[level])
                self.assertEqual(frozen["frozen_span"], 2)
                self.assertEqual(frozen["frozen_constraint_rows"], 5*level//2)
                self.assertEqual(frozen["frozen_norm_rows"], rows)

    def test_representative_git_drift_inventory(self):
        # Exact drift inventory and frozen native beam identity, explicitly
        # dependent on the c64d189e git objects: skips when they are absent
        # (git-free source export). On an actual checkout the drifted source
        # binding must refuse; it is never forced valid.
        if not self.git_history_available():
            self.skipTest(
                "c64d189e git objects absent (git-free source export)")
        changed = subprocess.run(
            ["git", "diff", "--name-only", "c64d189e", "HEAD", "--",
             *parity.SOURCE_FILES],
            cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
        self.assertEqual(sorted(changed), [
            "src/anyfesolver/_stiffened_shell_ge_experimental.py",
            "src/anyfesolver/_stiffened_shell_l1_assembly_experimental.py"])
        old_owner = subprocess.run(
            ["git", "show",
             f"{parity.OLD_PRODUCT}:src/anyfesolver/_stiffened_shell_ge_experimental.py"],
            cwd=ROOT, check=True, capture_output=True).stdout
        current_owner = (ROOT /
            "src/anyfesolver/_stiffened_shell_ge_experimental.py").read_bytes()
        self.assertEqual(parity.function_ast_hash(old_owner, "_assemble_beam"),
                         parity.function_ast_hash(current_owner, "_assemble_beam"))

    def test_wrong_level_refused(self):
        with self.assertRaises(ValueError):
            parity.preflight(16)

    def test_changed_reference_function_refused(self):
        source = (ROOT / "scripts/b3_e4_bl5_reference_actions_dev.py").read_bytes()
        changed = source.replace(b"return np.r_[beam_u.reshape(-1)",
                                 b"return np.r_[beam_u.reshape(-1)*2", 1)
        self.assertNotEqual(parity.function_ast_hash(source, "physical_loads"),
                            parity.function_ast_hash(changed, "physical_loads"))

    def test_changed_frozen_case_source_bytes_refused(self):
        original = parity.sha

        def changed(path):
            if path.name == "run_case.py":
                return "0"*64
            return original(path)

        with patch.object(parity, "sha", side_effect=changed):
            with self.assertRaises(ValueError):
                parity.check_frozen_case(32, 8, 32)

    def test_signed_swap_basis_and_false_map_refusal(self):
        import numpy as np
        from scipy import sparse
        current = sparse.csc_matrix([[0., -1.], [1., 0.]])
        frozen = sparse.eye(2, format="csc")
        p = parity.signed_physical_basis_map(current, frozen, np, sparse)
        self.assertEqual(parity.max_sparse_abs(current@p-frozen), 0.0)
        current_h = sparse.diags([2., 3.], format="csc")
        expected_h = sparse.diags([3., 2.], format="csc")
        self.assertEqual(parity.max_sparse_abs(p.T@current_h@p-expected_h), 0.0)
        wrong = sparse.csc_matrix([[0., -2.], [1., 0.]])
        with self.assertRaises(ValueError):
            parity.signed_physical_basis_map(wrong, frozen, np, sparse)

    def test_existing_failed_output_preserved_and_versioned_target_fresh(self):
        existing = parity.OUT / "run-nx32-ny8-half32"
        if not existing.exists():
            self.skipTest("saved f03 owner-parity failed-output evidence "
                "(docs/agent_plans/development/r8-f03-current-owner-parity-20261009/"
                "run-nx32-ny8-half32) is absent; the preserved-failed-output "
                "control requires the exact saved failed output")
        self.assertTrue(existing.exists())
        with self.assertRaises(ValueError):
            parity.require_fresh_output(existing)
        self.assertEqual(parity.campaign_folder(32, "v2").name,
                         "run-v2-nx32-ny8-half32")
        with self.assertRaises(ValueError):
            parity.campaign_folder(32, "../v2")

    def test_wrong_predecessor_campaign_or_identity_refused(self):
        binding = {"current_sources_sha256":
                   {"scripts/r8_bl5_actual_primal_work_dev.py": "0"*64}}
        receipt = {"campaign": "v2", "status": "completed", "exit_code": 0,
                   "source_dependencies_sha256": binding["current_sources_sha256"],
                   "process_wall_seconds": 7.0}
        outcome = {"campaign": "v2",
                   "status": "current_owner_frozen_c2_work_parity_passed_development_only",
                   "level": 32, "nx": 32, "ny": 8, "halfcells": 32,
                   "bound_inputs": {"family_result_sha256": parity.FAMILY_SHA}}
        self.assertEqual(parity.validate_prior_32(receipt, outcome, binding, "v2"), 7.0)
        for changed_receipt, changed_outcome in (
            ({**receipt, "campaign": "v1"}, outcome),
            (receipt, {**outcome, "level": 64}),
            ({**receipt, "source_dependencies_sha256": {}}, outcome)):
            with self.assertRaises(ValueError):
                parity.validate_prior_32(changed_receipt, changed_outcome, binding, "v2")


if __name__ == "__main__":
    unittest.main()
