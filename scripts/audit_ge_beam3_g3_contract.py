"""Standard-library-only G3 planning audit. No element imports or mechanics."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "d5acb45b0a35d6063d8d075bbb2082c56f26c9e4"
TREE = "d0d67ce2e03893a1e9737299972c520aea923ee4"
DOCUMENT = "docs/GE_BEAM3_G3_GRAPH_JUNCTION_CONTRACT.md"
FIXTURES = "docs/reference_cases/ge_beam3_g3_graph_fixtures_v1.json"
MANIFEST = "docs/reference_cases/ge_beam3_g3_graph_contract_v1.json"
AUDITOR = "scripts/audit_ge_beam3_g3_contract.py"
TEST = "tests/test_ge_beam3_g3_contract.py"
EXTENT = (DOCUMENT, FIXTURES, MANIFEST, AUDITOR, TEST)
SOURCES = (
    "docs/GE_BEAM3_GENERAL_STATIC_INTEGRATION_CONTRACT.md",
    "docs/reference_cases/ge_beam3_general_static_integration_contract_v1.json",
    "docs/GE_BEAM3_LEGACY_PARITY_MATRIX.md",
    "docs/reference_cases/ge_beam3_legacy_parity_matrix_v1.json",
    "docs/GE_BEAM3_G2_CONSTRAINT_CONTRACT.md",
    "docs/GE_BEAM3_G2_CONFIRMATION_STATUS.md",
    "docs/reference_cases/ge_beam3_g2_implementation_review_v2.json",
    "docs/reference_cases/ge_beam3_g2_confirmation_review_v1.json",
    "docs/reference_cases/ge_beam3_g2_confirmation_aggregate_v1.json",
    "src/anyfesolver/_ge_beam3_g1_analysis.py",
    "src/anyfesolver/_ge_beam3_g1_element.py",
    "src/anyfesolver/_ge_beam3_g1_operator.py",
    "src/anyfesolver/_ge_beam3_g2_analysis.py",
    "src/anyfesolver/_ge_beam3_g2_constraints.py",
    "src/anyfesolver/_ge_beam3_pose_joint.py",
    "src/anyfesolver/_ge_beam3_shell_joint_trial.py",
    "src/anyfesolver/_ge_beam3_shell_joint_state.py",
    "src/anyfesolver/elements.py",
    "src/anyfesolver/e4_pl_element.py",
    "src/anyfesolver/e4_pl_s3_v2d_element.py",
    "src/anyfesolver/assembly.py",
    "src/anyfesolver/constraint_audit.py",
    "src/anyfesolver/nonlinear_state.py",
    "tests/test_ge_beam3_g1_elastic.py",
    "tests/test_ge_beam3_g2_constraints.py",
    "tests/test_ge_beam3_g2_corrections.py",
    "tests/test_ge_beam3_pose_joint.py",
    "tests/test_ge_beam3_shell_joint_trial.py",
    "tests/test_ge_beam3_shell_joint_state.py",
    "tests/test_constraint_audit.py",
    "tests/test_generalized_beam_sections.py",
)


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(body):
    return (json.dumps(body, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True)+"\n").encode("ascii")


def strict(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate key")
            result[key] = value
        return result
    def bad(value): raise ValueError("nonfinite JSON")
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=bad)
    require(canonical(value) == data, "noncanonical JSON")
    return value


def text(path): return (ROOT/path).read_bytes().replace(b"\r\n", b"\n")
def digest(data): return dict(bytes=len(data), sha256=sha256(data).hexdigest())


def git(*args):
    return subprocess.run(["git", "--no-replace-objects", "-c", "safe.directory="+ROOT.as_posix(), *args],
        cwd=ROOT, check=True, capture_output=True, timeout=30).stdout


def vector(value):
    require(type(value) is list and len(value) == 3, "three-vector required")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value), "finite numeric coordinates")
    return [Fraction(x) for x in value]


def nodes(rows):
    result = {}
    for node, x in rows:
        require(type(node) is int and node > 0 and node not in result, "unique positive node ID")
        result[node] = vector(x)
    return result


def native_element(e, points):
    require(set(e) == {"id", "nodes", "orientation", "roll", "section"}, "element schema")
    require(type(e["id"]) is int and e["id"] > 0, "element ID")
    conn = e["nodes"]
    require(len(conn) == 3 and len(set(conn)) == 3 and all(n in points for n in conn), "native connectivity")
    a, mid, b = [points[n] for n in conn]
    require(all(2*m == x+y for m,x,y in zip(mid,a,b)), "exact reference midpoint")
    t = [y-x for x,y in zip(a,b)]; v = vector(e["orientation"])
    require(any(t), "nonzero edge")
    cross = [t[1]*v[2]-t[2]*v[1],t[2]*v[0]-t[0]*v[2],t[0]*v[1]-t[1]*v[0]]
    require(any(cross), "physical orientation must not parallel tangent")
    require(type(e["roll"]) in (int,float) and math.isfinite(e["roll"]), "finite roll")
    require(e["section"] == "native_isotropic", "section authority")


def graph_shape(g):
    points = nodes(g["nodes"]); edges = g["elements"]
    require(len(points) <= 32 and len(edges) <= 8, "graph bound")
    require(len({e["id"] for e in edges}) == len(edges), "duplicate element")
    for e in edges: native_element(e, points)
    require({n for e in edges for n in e["nodes"]} == set(points), "orphan node")
    adjacency = {}
    for e in edges:
        a,_,b = e["nodes"]; adjacency.setdefault(a,set()).add(b); adjacency.setdefault(b,set()).add(a)
    unseen = set(adjacency); components = []
    while unseen:
        todo = [min(unseen)]; component = set()
        while todo:
            node = todo.pop()
            if node in component: continue
            component.add(node); todo.extend(adjacency[node]-component)
        unseen -= component; components.append(component)
    require(all(set(g["fixed_nodes"]) & c for c in components), "unsupported component")
    require(all(n in points for n in g["fixed_nodes"]+g["load_nodes"]), "support/load node")
    return len(components), len(edges)-len(adjacency)+len(components)


def mixed_shape(g):
    p = nodes(g["nodes"]); e = g["native_element"]; native_element(e,p)
    other = g["other_element"]; conn = other["nodes"]
    require(g["other_family"] in {"B2","B3","Q4","S3"}, "mixed family")
    require(len(conn) == {"B2":2,"B3":3,"Q4":4,"S3":3}[g["other_family"]], "family node count")
    require(not set(conn) & set(e["nodes"]), "shared cross-family node IDs forbidden")
    require(set(conn)|set(e["nodes"]) == set(p), "mixed graph coverage")
    require(e["id"] != other["id"], "duplicate mixed element ID")
    tie = g["tie"]; require(set(tie) == {"slave","masters","components","offset"}, "tie schema")
    require(tie["components"] == [0,1,2] and tie["offset"] == [0,0,0], "translation only tie")
    require(tie["slave"] in e["nodes"], "native slave")
    masters = tie["masters"]; require(len({m for m,_ in masters}) == len(masters), "duplicate master")
    require(all(m in conn for m,_ in masters), "nonnative masters")
    weights = [Fraction(w) for _,w in masters]
    require(sum(weights) == 1, "partition of unity")
    require(all(sum(w*p[m][j] for (m,_),w in zip(masters,weights)) == p[tie["slave"]][j] for j in range(3)), "reference position reproduction")
    require(set(g["fixed_nodes"]) & set(conn) and set(g["fixed_nodes"]) & set(e["nodes"]), "supported mixed components")


def validate_fixtures(f):
    require(set(f) == {"schema","status","parent","native_graphs","mixed_graphs","families","materials","programs","variants","graph_rejections","rotational_rejections","cache_graphs","mutation_fields","admission","limits","acceptance","planned_tests","slices"}, "fixture schema")
    require(f["schema"] == "GE_BEAM3_G3_FIXTURES_V1" and f["status"] == "FROZEN_DESIGN_NOT_IMPLEMENTED_OR_QUALIFIED" and f["parent"] == BASE, "fixture authority")
    require([g["id"] for g in f["native_graphs"]] == ["N_BRANCH3","N_RING4","N_BRACED5","N_DISJOINT2"], "native inventory")
    require([g["id"] for g in f["mixed_graphs"]] == ["M_B2","M_B3","M_Q4","M_S3","M_Q4_WEIGHTED"], "mixed inventory")
    for g in f["native_graphs"]:
        require(set(g) == {"id","nodes","elements","fixed_nodes","load_nodes","components","cycle_rank","status"}, "graph schema")
        require(graph_shape(g) == (g["components"],g["cycle_rank"]), "component/cycle count")
        require(g["status"] == "PLANNED_NOT_EXECUTED", "no execution claim")
    for g in f["mixed_graphs"]:
        require(set(g) == {"id","nodes","native_element","other_family","other_element","fixed_nodes","load_node","tie","status"}, "mixed schema")
        mixed_shape(g); require(g["status"] == "PLANNED_NOT_EXECUTED", "no execution claim")
    a=f["admission"]
    require(a["accepted_cross_family_rotational_adapters"] == [] and a["implicit_fallback"] is False and
        a["s17_positive_status"] == "DEFERRED_SEPARATE_ADAPTER_CONTRACT_AND_QUALIFICATION", "S17 not qualified")
    require(f["families"]["S3"]["class"].endswith(".NativeParityE4PLS3V2DShellElement"), "accepted V2D family")
    require(f["limits"] == dict(elements=8,nodes=32,external_dofs=192,restart_bytes=8388608,accepted_history=128,
        child_seconds=600,wave_seconds=1800,inactivity_seconds=120,process_tree_memory_bytes=25769803776,
        numerical_threads=1,max_workers=3,formal_cycles=2,automatic_retry=False), "execution bounds")
    require(f["acceptance"] == dict(invariant=1e-11,directional=1e-7,derivative_steps=[1e-4,1e-5,1e-6],
        scales=dict(length=1,force=1,moment=1),normalization="NORM_DIFF_OVER_MAX_1_NORM_A_NORM_B",exact_affine=True), "frozen acceptance")
    rows=f["planned_tests"]
    require(len(rows)==14 and len({r["name"] for r in rows})==14 and {r["gate"] for r in rows}=={"S15","S16","S17","S18"}, "planned test inventory")
    for row in rows:
        require(set(row)=={"gate","name","fixtures","regime","oracle","status","path"} and row["status"]=="PLANNED_NOT_IMPLEMENTED", "planned test schema")
    require(f["slices"][-1]["status"]=="POSITIVE_ADAPTER_DEFERRED", "S17 positive remains deferred")


def expected():
    require(git("rev-parse", BASE+"^{tree}").decode().strip()==TREE, "base tree mismatch")
    fixture=strict(text(FIXTURES)); validate_fixtures(fixture)
    sources=[]
    for path in SOURCES:
        data=git("show", BASE+":"+path)
        require(text(path)==data.replace(b"\r\n",b"\n"), "inherited source changed: "+path)
        sources.append(dict(path=path,commit=BASE,tree=TREE,blob=git("rev-parse",BASE+":"+path).decode().strip(),**digest(data)))
    return dict(schema="GE_BEAM3_G3_GRAPH_CONTRACT_V1",status="FROZEN_DESIGN_NOT_IMPLEMENTED_OR_QUALIFIED",
        parent_commit=BASE,parent_tree=TREE,allowed_extent=list(EXTENT),source_bindings=sources,
        text_bindings=[dict(path=p,encoding="UTF8_LF",**digest(text(p))) for p in (DOCUMENT,FIXTURES,AUDITOR,TEST)],
        native_fixture_count=4,mixed_fixture_count=5,planned_test_families=14,
        rotational_rejection_combinations=28,positive_rotational_adapters=0,
        implementation_started=False,mechanics_runs=0,qualification_claim=False,default_changes=False,
        next_gate="G3a_NATIVE_GRAPH_IMPLEMENTATION_ONLY",
        terminal_precedence=["BLOCKED_GE_BEAM3_G3_AUTHORITY","BLOCKED_GE_BEAM3_G3_PROCESS_OR_EVIDENCE",
            "NO_GO_GE_BEAM3_G3_STATE_OR_RESTART","NO_GO_GE_BEAM3_G3_JUNCTION_OR_WORK",
            "NO_GO_GE_BEAM3_G3_GRAPH_OR_ASSEMBLY","PROVISIONAL_GO_GE_BEAM3_G3_EXPLICIT_SLICE_ONLY"])


def validate(raw, wanted):
    value=strict(raw); require(canonical(value)==canonical(wanted), "authority schema or hash mismatch")


def check_extent():
    # Local full-history audit: a missing base fails closed, never implies clean.
    git("merge-base","--is-ancestor",BASE,"HEAD")
    changed=set(git("diff","--name-only",BASE,"--").decode().splitlines())
    extra=set(git("ls-files","--others","--exclude-standard").decode().splitlines())
    require(not (changed|extra)-set(EXTENT), "outside planning-only extent")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args=parser.parse_args(); wanted=expected(); check_extent()
    if args.render: sys.stdout.buffer.write(canonical(wanted))
    else:
        validate(text(MANIFEST),wanted)
        print("PASS: G3 planning-only freeze; 4 native, 5 mixed fixtures; 14 planned test families; zero mechanics runs")


if __name__=="__main__": main()
