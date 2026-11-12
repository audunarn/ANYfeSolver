"""Static G3a scope/bounds checks; separate from mechanics inventory."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = "e78c1b667694b43141912c1d85f894de2b41a848"
TREE = "ec23282230399da6a5c882f025f1bf53b16bc52b"
EXTENT = {
    "src/anyfesolver/_ge_beam3_g3_analysis.py",
    "src/anyfesolver/_ge_beam3_g3_constraints.py",
    "scripts/run_ge_beam3_g3a.py",
    "tests/test_ge_beam3_g3a_graph.py",
    "tests/test_ge_beam3_g3a_boundary.py",
    "docs/GE_BEAM3_G3A_IMPLEMENTATION_STATUS.md",
    "docs/reference_cases/ge_beam3_g3a_development_v1.json",
}


def git(*args):
    result = subprocess.run(["git", "-c", "safe.directory="+ROOT.as_posix(), *args],
                            cwd=ROOT, capture_output=True, check=True, timeout=15)
    return result.stdout.decode().strip()


def source(path):
    return (ROOT/path).read_bytes().replace(b"\r\n", b"\n")


class G3aBoundaryTests(unittest.TestCase):
    def test_parent_and_additive_extent(self):
        self.assertEqual(git("rev-parse", BASE+"^{tree}"), TREE)
        git("merge-base", "--is-ancestor", BASE, "HEAD")
        for line in git("diff", "--name-status", BASE).splitlines():
            status, path = line.split("\t")
            self.assertEqual(status, "A")
            self.assertIn(path, EXTENT)
        for path in git("ls-files", "--others", "--exclude-standard").splitlines():
            self.assertIn(path, EXTENT)
        git("diff", "--check", BASE)

    def test_parent_authorities_remain_byte_bound(self):
        contract = json.loads(source("docs/reference_cases/ge_beam3_g3_graph_contract_v1.json"))
        for row in contract["source_bindings"]+contract["text_bindings"]:
            with self.subTest(path=row["path"]):
                raw = source(row["path"])
                self.assertEqual(len(raw), row["bytes"])
                self.assertEqual(sha256(raw).hexdigest(), row["sha256"])

    def test_no_public_or_mechanics_implementation_imports(self):
        raw = source("src/anyfesolver/_ge_beam3_g3_analysis.py")
        tree = ast.parse(raw)
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeGraphAnalysis")
        self.assertEqual(owner.bases, [])
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertNotIn("_ge_beam3_g2_analysis", modules)
        self.assertNotIn("elements", modules)
        self.assertNotIn("_ge_beam3_g1_operator", modules)
        for token in (b"pinv(", b"inv(", b"evalf(", b"lstsq("):
            self.assertNotIn(token, raw)

    def test_runner_limits_and_external_exclusive_logs(self):
        raw = source("scripts/run_ge_beam3_g3a.py").decode()
        ast.parse(raw)
        for token in ("24*1024**3", "now-start >= 600", "now-last >= 120",
                      "job.terminate()", "job.close()", 'open("xb")',
                      "tempfile.mkdtemp", '"OMP_NUM_THREADS"', '"OPENBLAS_NUM_THREADS"',
                      '"MKL_NUM_THREADS"', '"NUMEXPR_NUM_THREADS"',
                      'env[key] = "1"', "DEVELOPMENT_ONLY_NOT_QUALIFICATION"):
            self.assertIn(token, raw)

    def test_separate_graph_constraint_bounds(self):
        old = source("src/anyfesolver/_ge_beam3_g2_constraints.py").decode()
        new = source("src/anyfesolver/_ge_beam3_g3_constraints.py").decode()
        self.assertIn("size <= 36", old); self.assertIn("len(node_ids) <= 6", old)
        self.assertIn("size <= 192", new); self.assertIn("len(node_ids) <= 32", new)
        self.assertIn("GE_BEAM3_G3_NATIVE_CONSTRAINTS_V1", new)
        self.assertIn("GE_BEAM3_G3_GRAPH_ELASTIC_RESTART_V1", source("src/anyfesolver/_ge_beam3_g3_analysis.py").decode())


if __name__ == "__main__": unittest.main()
