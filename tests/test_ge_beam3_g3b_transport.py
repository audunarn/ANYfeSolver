"""Frozen reference-interface transports; development diagnostics, not an owner.

Fresh transported family operators are compared against the prior candidate
operators and a separately assembled full stationary KKT system. This module
does not widen any mixed helper's runtime admission or restart authority.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Reference
from anyfesolver._ge_beam3_g1_element import ElasticElement
from anyfesolver._ge_beam3_g1_elastic import ElasticSection, canonical
from anyfesolver._ge_beam3_g1_operator import schur
from anyfesolver._ge_beam3_g3b_reference import progress
from anyfesolver.fe_core import FEModel

ROOT = Path(__file__).resolve().parents[1]
PARENT = "7c4c980a8c5b4ccadb6721be177746af58c341ae"
LANES = ("mb2", "mb3", "mq4", "ms3", "weighted")
CASES = ("M_B2", "M_B3", "M_Q4", "M_S3", "M_Q4_WEIGHTED")
VARIANTS = ("reference", "renumber", "reverse", "global", "insertion")
FIXTURE_PATH = ROOT/"docs/reference_cases/ge_beam3_g3_graph_fixtures_v1.json"


def authority():
    contract = json.loads((ROOT/"docs/reference_cases/ge_beam3_g3_graph_contract_v1.json").read_text())
    for item in contract["source_bindings"] + contract["text_bindings"]:
        data = (ROOT/item["path"]).read_bytes().replace(b"\r\n", b"\n")
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("frozen G3 authority mismatch: " + item["path"])
    return json.loads(FIXTURE_PATH.read_text())


def baseline(lane):
    spec = importlib.util.spec_from_file_location("transport_baseline_"+lane,
        ROOT/("tests/test_ge_beam3_g3b_"+lane+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invariant(a, b):
    a, b = np.asarray(a), np.asarray(b)
    error = np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b))
    assert np.isfinite(error) and error <= 1e-11, error


def compare(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b: compare(a[k], b[k])
    elif isinstance(b, (str, bool)) or b is None:
        assert a == b
    elif isinstance(b, (tuple, list)) and any(isinstance(x, (str, dict)) for x in b):
        assert len(a) == len(b)
        for x,y in zip(a,b): compare(x,y)
    else:
        invariant(a, b)


def physical_transport(current, reference, W):
    tensors=("global_membrane_resultant_tensors","global_bending_resultant_tensors")
    vector="global_transverse_shear_resultants"
    labels=((0,0,"xx"),(1,1,"yy"),(2,2,"zz"),(0,1,"xy"),(1,2,"yz"),(0,2,"xz"))
    expected=set(tensors+(vector,))
    expected.update("global_"+label+"_"+surface for _,_,label in labels for surface in ("top","bot"))
    assert {k for k in current if k.startswith("global_")} == expected
    for key in tensors: invariant(current[key],W @ reference[key] @ W.T)
    invariant(current[vector],reference[vector] @ W.T)
    # Six component arrays are samples of symmetric tensors, not vectors.
    # In particular S3's three station vectors form a (3,3) array, not a tensor.
    for surface in ("top","bot"):
        made=[]
        for record in (current,reference):
            tensor=np.zeros((len(record["global_xx_"+surface]),3,3))
            for i,j,label in labels:
                tensor[:,i,j]=record["global_"+label+"_"+surface]
                tensor[:,j,i]=tensor[:,i,j]
            made.append(tensor)
        invariant(made[0],W @ made[1] @ W.T)


def construct(lane, variant):
    if lane not in LANES or variant not in VARIANTS:
        raise ValueError("unregistered transport fixture")
    frozen = authority()  # before creating/evaluating either element
    fixture = next(g for g in frozen["mixed_graphs"] if g["id"] == CASES[LANES.index(lane)])
    module = baseline(lane)
    native, other, material, kw = module.parts()
    original = sorted(kw["nodes"])
    mapping = {n: 1001+7*(len(original)-1-i) if variant == "renumber" else n
               for i,n in enumerate(original)}
    W = (Rotation.from_rotvec(frozen["variants"]["common_rotation_vector"]).as_matrix()
         if variant == "global" else np.eye(3))
    shift = np.array(frozen["variants"]["common_translation"]) if variant == "global" else np.zeros(3)
    positions = {mapping[n]: W @ np.array(x)+shift for n,x in kw["nodes"].items()}
    native_ids = tuple(mapping[n] for n in native.node_ids)
    # The reference contract requires an exact midpoint in binary64.
    positions[native_ids[1]] = (positions[native_ids[0]]+positions[native_ids[2]])/2
    frames = np.einsum("ij,njk->nik", W, native.operator.reference.nodal_triads)
    if variant == "reverse":
        native_ids = native_ids[::-1]
        frames = frames[::-1] @ np.diag([-1.,1.,-1.])
    native = ElasticElement(native.element_id, native_ids,
        Reference([positions[n] for n in native_ids], frames),
        ElasticSection.isotropic(**frozen["materials"]["native_isotropic"]))
    other_ids = [mapping[n] for n in other.node_ids]
    if lane in ("mb2", "mb3"):
        section = deepcopy(other.cross_section)
        section["orientation"] = (W @ np.array(section["orientation"])).tolist()
        # Legacy constructors capture orientation; never mutate only the input
        # dictionary after construction and leave the captured triad stale.
        other = type(other)(other.element_id,other_ids,cross_section=section)
    else:
        other = type(other)(other.element_id,other_ids,thickness=other.thickness,
                            reference_normal=W @ np.array(kw["reference_normal"]))
    ids = tuple(sorted(positions)); size = 6*len(ids)
    B = np.zeros((size,size))
    for i,n in enumerate(original):
        j = ids.index(mapping[n])
        for d in (0,3): B[6*j+d:6*j+d+3,6*i+d:6*i+d+3] = W
    model = FEModel("G3b frozen transport diagnostic")
    insertion = list(ids)[::-1] if variant == "insertion" else list(ids)
    for n in insertion: model.add_node(n,*positions[n])
    elements = (other,native) if variant == "insertion" else (native,other)
    for e in elements: model.add_element(e.element_id,e)
    # FEModel insertion order is intentionally not our canonical assembly order.
    local_dofs = tuple([6*ids.index(n)+d for n in e.node_ids for d in range(6)]
                       for e in (native,other))
    # Element stiffness is returned in its local connectivity order.
    progress("capture")
    full = native.operator.evaluate(native.operator.reference.coordinates, np.zeros((3,3)),
        native.operator.reference.nodal_triads, np.tile(np.eye(3),(2,1,1)), np.zeros(18))
    invariant(full["residual"],np.zeros(42))
    block = other.compute_stiffness_matrix(model.mesh, material)
    progress("local solve")
    return dict(module=module,fixture=fixture,mapping=mapping,W=W,positions=positions,
        ids=ids,size=size,B=B,model=model,native=native,other=other,material=material,
        dofs=local_dofs,full=full,block=block)


def condensed(data, base):
    _, block, lift, _ = schur(data["full"]["residual"],data["full"]["jacobian"])
    K = np.zeros((data["size"],data["size"]))
    for dofs,b in zip(data["dofs"],(block,data["block"])):
        K[np.ix_(dofs,dofs)] += b
    T = data["B"] @ base.T
    invariant(K,K.T)
    u = T @ np.linalg.solve(T.T @ K @ T,T.T @ (data["B"] @ data["module"].load(base)))
    return K,u,lift @ u[data["dofs"][0]]


def full_kkt(data, f):
    # No Schur helper, baseline T/K, candidate global assembly or cached blocks.
    n = data["size"]; H = np.zeros((n+24,n+24))
    full = data["full"]["hessian"]
    for dofs,block in ((data["dofs"][0]+list(range(n,n+24)),full),
                       (data["dofs"][1],data["block"])):
        for i,a in enumerate(dofs):
            for j,b in enumerate(dofs): H[a,b] += block[i,j]
    fixture = data["fixture"]; mapping = data["mapping"]; ids = data["ids"]
    support_count = 6*len(fixture["fixed_nodes"])
    J = np.zeros((support_count+3,n+24))
    row = 0
    for old in sorted(fixture["fixed_nodes"]):
        for d in range(6):
            J[row,6*ids.index(mapping[old])+d] = 1
            row += 1
    tie = fixture["tie"]
    for d in range(3):
        J[row+d,6*ids.index(mapping[tie["slave"]])+d] = 1
        for old,weight in tie["masters"]:
            J[row+d,6*ids.index(mapping[old])+d] -= float(Fraction(weight))
    system = np.block([[H,J.T],[J,np.zeros((len(J),len(J)))]])
    solution = np.linalg.solve(system,np.r_[f,np.zeros(24+len(J))])
    invariant(system @ solution,np.r_[f,np.zeros(24+len(J))])
    progress("assembly")
    return solution,H,J,support_count


def exercise(lane,variant):
    data = construct(lane,variant)
    base = data["module"].make()
    original = base.solve(data["module"].load(base))
    K,u,internal = condensed(data,base)
    B = data["B"]; n = data["size"]; f = B @ data["module"].load(base)
    invariant(K,B @ base.K @ B.T)
    invariant(u,B @ original["u"])
    # Fresh second family construction; KKT does not share candidate caches.
    oracle = construct(lane,variant)
    solution,H,J,ns = full_kkt(oracle,f)
    invariant(u,solution[:n]); invariant(internal,solution[n:n+24])
    invariant(J[:,:n] @ u,np.zeros(len(J)))
    mu = solution[n+24:]
    support = -J[:ns,:n].T @ mu[:ns]
    tie = -J[ns:,:n].T @ mu[ns:]
    invariant(support,B @ original["support_reactions"])
    invariant(tie,B @ original["tie_forces"])
    invariant(K @ u-f,support+tie)
    invariant(u @ K @ u/2,original["energy"])
    invariant(u @ K @ u/2,solution[:n+24] @ H @ solution[:n+24]/2)
    invariant(u @ (K @ u-f),0.)
    force = (support+f).reshape(-1,6)
    x = np.array([data["positions"][node] for node in data["ids"]])
    invariant(force[:,:3].sum(axis=0),np.zeros(3))
    invariant((force[:,3:]+np.cross(x,force[:,:3])).sum(axis=0),np.zeros(3))
    tf = tie.reshape(-1,6)
    invariant(tf[:,:3].sum(axis=0),np.zeros(3))
    invariant(np.cross(x,tf[:,:3]).sum(axis=0),np.zeros(3))
    assert not np.any(J[ns:,:n][:,[6*i+d for i in range(n//6) for d in (3,4,5)]])
    assert not np.any(tf[:,3:])
    recovered = data["native"].operator.cell.recover(internal[6:])
    checked = oracle["native"].operator.cell.recover(solution[n+6:n+24])
    compare(recovered,checked)
    reference_stations = original["native_stations"]
    signs = np.array([1.,-1.,1.,1.,-1.,1.]) if variant == "reverse" else np.ones(6)
    if variant == "reverse": reference_stations = reference_stations[::-1]
    # Reversal changes the directed cut and the proper material triad:
    # both six-vectors transform by -diag(-1,1,-1), once per triple.
    for station,reference in zip(recovered,reference_stations):
        invariant(station["strain"],signs*reference["strain"])
        invariant(station["resultants"],signs*reference["resultants"])
    shell = lane not in ("mb2","mb3")
    kwargs = dict(return_global=True) if shell else {}
    a = data["other"].compute_stresses(data["model"].mesh,u[data["dofs"][1]],data["material"],**kwargs)
    b = oracle["other"].compute_stresses(oracle["model"].mesh,solution[oracle["dofs"][1]],oracle["material"],**kwargs)
    compare(a,b)
    if shell:
        assert a["numerical_fields_excluded"] is True
        director = a["frame"][:,2] if lane == "ms3" else a["physical_director"]
        invariant(director,data["W"] @ np.array([0.,0.,1.]))
        # Check physical tensor/vector transport, not equality of untransformed
        # global components; keep each family's existing recovery schema.
        original_recovery = original["shell_recovery"]
        physical_transport(a,original_recovery,data["W"])
    progress("recovery")
    return dict(case=CASES[LANES.index(lane)],variant=variant,qualification=False,
        external_count=n,internal_count=24,kkt_count=len(solution),
        energy=float(u @ K @ u/2),u=u,support=support,tie=tie,
        native_stations=recovered,other_recovery=a)


@pytest.mark.parametrize("lane",LANES)
@pytest.mark.parametrize("variant",VARIANTS)
def test_frozen_transport(lane,variant):
    exercise(lane,variant)


@pytest.mark.parametrize("lane",LANES)
def test_transport_corruption_is_detected(lane):
    data=construct(lane,"global"); base=data["module"].make()
    K,u,_=condensed(data,base)
    wrong=data["B"].copy(); wrong[0,:]*=-1
    with pytest.raises(AssertionError): invariant(K,wrong @ base.K @ wrong.T)
    with pytest.raises(AssertionError): invariant(u+np.eye(len(u))[0]*.01,data["B"] @ base.solve(data["module"].load(base))["u"])


def test_unknown_case_rejected_before_construction():
    from unittest.mock import patch
    with patch(__name__+".baseline",side_effect=AssertionError("element creation")):
        with pytest.raises(ValueError): construct("legacy","global")
        with pytest.raises(ValueError): construct("mb2","finite")


@pytest.mark.parametrize("lane",("mb2","mb3"))
def test_legacy_orientation_is_captured_by_fresh_constructor(lane):
    data=construct(lane,"global")
    invariant(data["other"]._orientation,data["W"] @ np.array([0.,1.,0.]))


def test_authority_corruption_rejected_before_element_creation():
    from unittest.mock import patch
    original=Path.read_bytes
    def corrupted(path):
        result=original(path)
        return result+b" " if path == FIXTURE_PATH else result
    with patch.object(Path,"read_bytes",corrupted):
        with patch(__name__+".baseline",side_effect=AssertionError("element creation")):
            with pytest.raises(ValueError,match="authority mismatch"): construct("mb2","global")


@pytest.mark.parametrize("key",("global_transverse_shear_resultants","global_xy_top",
                                "global_membrane_resultant_tensors"))
def test_physical_recovery_mutations_are_not_masked(key):
    module=baseline("ms3"); p=module.make()
    recovery=p.solve(module.load(p))["shell_recovery"]
    corrupted=deepcopy(recovery)
    corrupted[key]=np.array(corrupted[key])+0.01
    with pytest.raises(AssertionError): physical_transport(corrupted,recovery,np.eye(3))
    del corrupted[key]
    with pytest.raises(AssertionError): physical_transport(corrupted,recovery,np.eye(3))


def test_deterministic_transport_packet(tmp_path):
    records=[exercise(lane,variant) for lane in LANES for variant in VARIANTS]
    packet=canonical(dict(schema="G3B_TRANSPORT_DEVELOPMENT_V1",qualification=False,records=records))
    path=Path(os.environ.get("G3B_TRANSPORT_DIAGNOSTIC_PAYLOAD",str(tmp_path/"transport.json")))
    with path.open("xb") as stream: stream.write(packet)


def test_transport_extent():
    git=["git","-c","safe.directory="+ROOT.as_posix()]
    allowed={"tests/test_ge_beam3_g3b_transport.py":"A",
        "docs/GE_BEAM3_G3B_TRANSPORT_DEVELOPMENT.md":"A",
        "docs/reference_cases/ge_beam3_g3b_transport_development_v1.json":"A",
        "scripts/run_ge_beam3_g3b.py":"M"}
    allowed.update({"tests/test_ge_beam3_g3b_"+lane+".py":"M" for lane in LANES})
    allowed.update({p:'A' for p in ["src/anyfesolver/_ge_beam3_g3b_owner.py","tests/test_ge_beam3_g3b_owner.py","docs/GE_BEAM3_G3B_OWNER_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_owner_development_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_ENTRY_SAFETY_STATUS.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v2.json"]})
    allowed.update({p:"A" for p in ["scripts/confirm_ge_beam3_g3b.py","scripts/ge_beam3_g3b_environment.py","tests/test_ge_beam3_g3b_confirmation_runner.py","docs/GE_BEAM3_G3B_CONFIRMATION_RUNNER_STATUS.md"]})
    allowed.update({p:"A" for p in ["scripts/ge_beam3_g3b_confirmation_authority.py","tests/test_ge_beam3_g3b_confirmation_authority.py","docs/GE_BEAM3_G3B_FORMAL_CONFIRMATION_CONTRACT.md","docs/reference_cases/ge_beam3_g3b_confirmation_inventory_v1.json"]})
    allowed.update({p:"A" for p in ["docs/GE_BEAM3_G3B_LOCAL_REVIEW_STATUS.md","docs/reference_cases/ge_beam3_g3b_local_safety_review_v1.json","src/anyfesolver/_ge_beam3_g3b_result_schema.py","tests/test_ge_beam3_g3b_owner_corrections.py","docs/GE_BEAM3_G3B_OWNER_CORRECTIONS.md"]})
    for line in subprocess.check_output(git+["diff","--name-status",PARENT,"--"],cwd=ROOT,timeout=20).decode().splitlines():
        status,path=line.split("\t"); assert allowed.get(path)==status
    extra=subprocess.check_output(git+["ls-files","--others","--exclude-standard"],cwd=ROOT,timeout=20).decode().splitlines()
    assert set(extra)<={p for p,s in allowed.items() if s=="A"}
    for lane,marker in zip(LANES,("test_additive_scope_and_immutable_parent_authority",
        "test_successor_extent_preserves_mechanics_and_historical_records",
        "test_mq4_extent_and_frozen_parent_bindings","test_ms3_extent_and_frozen_parent_bindings",
        "test_weighted_extent_and_frozen_parent_bindings")):
        path="tests/test_ge_beam3_g3b_"+lane+".py"; delimiter=("def "+marker+"():").encode()
        old=subprocess.check_output(git+["show",PARENT+":"+path],cwd=ROOT,timeout=20)
        assert old.split(delimiter)[0]==(ROOT/path).read_bytes().replace(b"\r\n",b"\n").split(delimiter)[0]
    authority()
