"""Planning-only G3 authority/fixture tests; standard library, no mechanics."""
import ast
from copy import deepcopy
import subprocess
import unittest
from unittest.mock import patch
from scripts import audit_ge_beam3_g3_contract as a


class G3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=a.strict(a.text(a.FIXTURES))
        cls.wanted=a.expected()

    def test_complete_canonical_authority(self):
        a.validate(a.text(a.MANIFEST), self.wanted)
        self.assertEqual(len(self.wanted["source_bindings"]),31)

    def test_fixture_geometry_and_incidence(self):
        a.validate_fixtures(self.fixture)
        self.assertEqual([a.graph_shape(g) for g in self.fixture["native_graphs"]],[(1,0),(1,1),(1,2),(2,0)])

    def test_duplicate_nonfinite_and_noncanonical_json(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}',b'{"a":1e999}',b'{"a":1}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError): a.strict(raw)

    def test_native_geometry_mutations(self):
        for kind in ("node","midpoint","parallel","orphan","element","support","cycle"):
            f=deepcopy(self.fixture); g=f["native_graphs"][0]
            if kind=="node": g["nodes"][1][0]=g["nodes"][0][0]
            elif kind=="midpoint": g["nodes"][2][1][0]+=.01
            elif kind=="parallel": g["elements"][0]["orientation"]=[1,0,0]
            elif kind=="orphan": g["nodes"].append([999,[0,0,0]])
            elif kind=="element": g["elements"][1]["id"]=g["elements"][0]["id"]
            elif kind=="support": f["native_graphs"][-1]["fixed_nodes"]=[101]
            else: g["cycle_rank"]=2
            with self.subTest(kind=kind),self.assertRaises(ValueError): a.validate_fixtures(f)

    def test_mixed_tie_mutations(self):
        for kind in ("rotation","weights","position","master","shared","family"):
            f=deepcopy(self.fixture); g=f["mixed_graphs"][-1]
            if kind=="rotation": g["tie"]["components"].append(3)
            elif kind=="weights": g["tie"]["masters"][0][1]="1/3"
            elif kind=="position": g["nodes"][-1][1][0]+=.1
            elif kind=="master": g["tie"]["masters"][1][0]=g["tie"]["masters"][0][0]
            elif kind=="shared": g["other_element"]["nodes"][0]=103
            else: g["other_family"]="UNKNOWN"
            with self.subTest(kind=kind),self.assertRaises(ValueError): a.validate_fixtures(f)

    def test_no_implicit_rotational_admission(self):
        for kind in ("allowlist","fallback","positive","s3"):
            f=deepcopy(self.fixture)
            if kind=="allowlist": f["admission"]["accepted_cross_family_rotational_adapters"]=["owned-joint"]
            elif kind=="fallback": f["admission"]["implicit_fallback"]=True
            elif kind=="positive": f["admission"]["s17_positive_status"]="QUALIFIED"
            else: f["families"]["S3"]["class"]="anyfesolver.e4_pl_s3_element.QualifiedE4PLS3ShellElement"
            with self.subTest(kind=kind),self.assertRaises(ValueError): a.validate_fixtures(f)

    def test_limits_and_tolerance_mutations(self):
        for group,key in (("limits","child_seconds"),("limits","elements"),("limits","nodes"),("acceptance","invariant"),("acceptance","derivative_steps")):
            f=deepcopy(self.fixture); f[group][key]=[] if key=="derivative_steps" else 999
            with self.subTest(key=key),self.assertRaises(ValueError): a.validate_fixtures(f)

    def test_unexecuted_inventory(self):
        rows=self.fixture["planned_tests"]
        self.assertEqual(len(rows),14)
        self.assertTrue(all(r["status"]=="PLANNED_NOT_IMPLEMENTED" for r in rows))
        self.assertFalse((a.ROOT/rows[0]["path"]).exists())
        self.assertEqual(len(self.fixture["rotational_rejections"]["families"])*len(self.fixture["rotational_rejections"]["mutations"]),28)

    def test_hash_schema_and_claim_mutations(self):
        for kind in ("source","text","claim","default","missing","extra","parent"):
            m=deepcopy(self.wanted)
            if kind=="source": m["source_bindings"][0]["sha256"]="0"*64
            elif kind=="text": m["text_bindings"][0]["sha256"]="0"*64
            elif kind=="claim": m["qualification_claim"]=True
            elif kind=="default": m["default_changes"]=True
            elif kind=="missing": del m["status"]
            elif kind=="extra": m["unexpected"]=True
            else: m["parent_commit"]="0"*40
            with self.subTest(kind=kind),self.assertRaises(ValueError): a.validate(a.canonical(m),self.wanted)

    def test_no_production_imports_in_auditor(self):
        tree=ast.parse(a.text(a.AUDITOR)); imported=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imported.extend(x.name.split('.')[0] for x in node.names)
            if isinstance(node,ast.ImportFrom): imported.append(node.module.split('.')[0])
        self.assertLessEqual(set(imported),{"argparse","fractions","hashlib","json","math","pathlib","subprocess","sys"})
        self.assertNotIn("import_module",a.text(a.AUDITOR).decode())

    def test_missing_base_is_not_silently_accepted(self):
        with patch.object(a,"git",side_effect=subprocess.CalledProcessError(128,"git")):
            with self.assertRaises(subprocess.CalledProcessError): a.expected()

    def test_source_extent_rejects_production_edit(self):
        with patch.object(a,"git",side_effect=[b'',b'src/anyfesolver/elements.py\n',b'']):
            with self.assertRaises(ValueError): a.check_extent()

    def test_actual_planning_only_extent(self):
        a.check_extent()
        changed=a.git("diff","--name-only",a.BASE,"--","src","pyproject.toml",".github").decode().strip()
        self.assertEqual(changed,"")


if __name__=="__main__": unittest.main()
