"""Prepared E03 algebra/deck/parser controls; no executed-reference evidence."""
import copy
import hashlib
import json
import re
import struct

import numpy as np
import pytest

from scripts import r8_e03_curved_reference as ref
from scripts import r8_e03_native_reference as native


@pytest.mark.parametrize("level", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
def test_deterministic_geometry_and_immutable_identity(level, sign):
    case = ref.build_case(level, sign)
    assert case == ref.build_case(level, sign)
    assert json.loads(json.dumps(case, allow_nan=False)) == case
    assert case["production_qualified"] is False
    np.testing.assert_allclose([np.arctan2(r["normal"][1], r["normal"][2]) for r in case["ribs"]],
                               [-.05, .05], atol=1e-15, rtol=0.)
    changed = copy.deepcopy(case)
    changed["young_pa"] *= 1.01
    with pytest.raises(ref.ReferenceParseError, match="identity"):
        ref.validate_case(changed)


def test_independent_fit_retains_deformation_and_recovers_rigid_pose():
    points = np.array(((0., -.2, -.1), (0., .2, -.1), (0., .2, .1), (0., -.2, .1)))
    origin, fit, h = ref.translation_pose_fit(points)
    np.testing.assert_array_equal(origin, np.zeros(3))
    np.testing.assert_allclose(fit @ h, np.eye(6), atol=1e-14)
    q = np.arange(12, dtype=float)
    deformation = (np.eye(12)-h @ fit) @ q
    assert np.linalg.norm(deformation) > 1.
    np.testing.assert_allclose(fit @ deformation, np.zeros(6), atol=1e-13)


@pytest.mark.parametrize("sign", (-1, 1))
def test_fitted_port_transpose_work_and_world_wrench(sign):
    shell = np.array(((0., -.2, -.1), (0., .2, -.1), (0., .2, .1), (0., -.2, .1)))
    beam = .1*shell+np.array((0., 0., sign*.03))
    connection = ref.fitted_connection(shell, beam)
    lam = np.array((7., -3., 5., .1, -.2, .3))
    sf, bf = ref.connection_actions(connection, lam)
    u = np.arange(24, dtype=float)*1e-5
    np.testing.assert_allclose(lam @ connection["constraint"] @ u, np.r_[sf.ravel(), bf.ravel()] @ u,
                               atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(sf.sum(axis=0)+bf.sum(axis=0), np.zeros(3), atol=1e-13)
    np.testing.assert_allclose(np.cross(shell, sf).sum(axis=0)+np.cross(beam, bf).sum(axis=0),
                               np.zeros(3), atol=1e-13)
    e = connection["beam_origin"]-connection["shell_origin"]
    shell_wrench = -connection["transport"].T @ lam
    np.testing.assert_allclose(shell_wrench[3:], -lam[3:]-np.cross(e, lam[:3]), atol=1e-14)


@pytest.mark.parametrize("sign", (-1, 1))
def test_native_b2_world_rows_and_dual_multiplier_map(sign):
    # Local reference-law control only: no owner compilation or global solve.
    from anyfesolver._b2_f10_f11_point_attachment import compile_reference_point

    case = ref.build_case(0, sign)
    definition, _, _, _ = native.b2_definition(case)
    point = definition.points[0]
    member = definition.members[0]
    host = next(shell for shell in definition.shells if shell.shell_id == point.shell_id)
    vertices = np.array([definition.shell_nodes_m[n] for n in host.node_ids])
    law = compile_reference_point(member, vertices, station=.5)
    _, ps, _ = ref.translation_pose_fit(vertices)
    embedded = np.zeros((6, 24))
    for i in range(4):
        embedded[:, 6*i:6*i+3] = ps[:, 3*i:3*i+3]
    trace = member.trace_global(np.zeros(12), stations=(.5,))
    offset = trace.reference_points_m[0]-vertices.mean(axis=0)
    world = np.column_stack((-ref.transport(offset) @ embedded, trace.jacobian[0]))
    t = np.zeros((6, 6))
    t[:3, :3] = law.reference_frames[0].T
    t[3:, 3:] = law.reference_frames[1].T
    np.testing.assert_allclose(law.constraint, t @ world, rtol=2e-12, atol=2e-12)
    q = np.arange(36, dtype=float)*1e-6
    lam_native = np.array((7., -3., 5., .1, -.2, .3))
    np.testing.assert_allclose(lam_native @ law.constraint @ q,
                               (t.T @ lam_native) @ world @ q, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("sign", (-1, 1))
def test_public_definition_geometry_and_distinct_shell_authorities(sign):
    case = ref.build_case(0, sign)
    b2, _, _, _ = native.b2_definition(case)
    ge, loads = native.ge_definition(case, density_kg_m3=7850.)
    assert len(b2.points) == len(ge.points) == 2*case["nx"]
    assert len(loads) == 2
    assert ge.shell_reference_policy == "CURVED_Q4_POINT_REFERENCE_V1_DEV"
    assert len(ge.shell.q4_elements) == 4*len(b2.shells)
    shell_node = ge.points[0].shell_node
    b2_host = next(s for s in b2.shells if s.shell_id == b2.points[0].shell_id)
    origin = np.mean([b2.shell_nodes_m[n] for n in b2_host.node_ids], axis=0)
    np.testing.assert_allclose(ge.shell.coordinates_m[shell_node-1], origin, atol=1e-14)
    assert case["profile_laws"][ref.GE] != case["profile_laws"][ref.B2]


def test_deck_actual_fields_double_contract_and_rounded_rigid_work():
    case = ref.build_case()
    text, metadata = ref.build_deck(case)
    assert text.count("*END STEP") == 1
    assert "*NODE OUTPUT,DOUBLE,NSET=PHYSICAL\nU\n" in text
    assert "*NODE OUTPUT,DOUBLE,NSET=ROOT\nRF\n" in text
    assert "*RIGID BODY" not in text and "SPRING" not in text and "*NODE FILE" not in text
    number = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
    for line in text.splitlines():
        for token in line.split(","):
            if number.fullmatch(token):
                assert len(token) <= 20
    assert metadata["source_nodes"] is not metadata["physical_nodes"]
    t, omega = np.array((.01, -.03, .02)), np.array((.1, -.2, .3))
    for row in metadata["deck_equations"]:
        residual = sum(coefficient*(t+np.cross(omega, metadata["physical_nodes"][node]))[component-1]
                       for node, component, coefficient in row)
        assert abs(residual) < 3e-12
    assert ref.digest(json.loads(json.dumps(metadata))) == ref.digest(metadata)
    assert ref.build_deck(case) == (text, metadata)


def test_ccx_formatter_refuses_nonfinite_and_never_truncates_tokens():
    for value in (-1.234567891234567e-300, 1.234567891234567e300, -.1234567891234567):
        token = ref.ccx_number(value)
        assert len(token) <= 20
        assert abs((float(token)-value)/value) < 1e-12
    for value in (float("nan"), float("inf")):
        with pytest.raises(ref.ReferenceParseError):
            ref.ccx_number(value)


def _synthetic_output(case):
    """Synthetic parser fixture, deliberately not an equilibrium/accuracy oracle."""
    _, metadata = ref.build_deck(case)
    nodes, elements, root = metadata["physical_nodes"], metadata["elements"], metadata["root_nodes"]
    u = {n: (0., 0., 0.) for n in nodes}
    for tip in metadata["tips"]:
        for n in tip["nodes"]:
            u[n] = (0., 0., -1e-6)
    applied = np.zeros(6)
    for n, force in metadata["nodal_loads"].items():
        applied += np.r_[force, np.cross(nodes[n], force)]
    root_map = np.concatenate([np.vstack((np.eye(3), ref.skew(nodes[n]))) for n in root], axis=1)
    reactions = np.linalg.lstsq(root_map, -applied, rcond=None)[0].reshape(-1, 3)
    rf = {n: tuple(force) for n, force in zip(root, reactions)}
    chunks = [b"    1C\n    1UPGM               CalculiX\n    1UVERSION           Version 2.23\n",
              f"    2C {len(nodes)} 3\n".encode("ascii")]
    chunks += [struct.pack("<i3d", n, *p) for n, p in nodes.items()]
    chunks += [f"    3C {len(elements)} 2\n".encode("ascii")]
    order = (*range(12), *range(16, 20), *range(12, 16))
    chunks += [struct.pack("<24i", e, 4, 0, 1, *(ids[j] for j in order)) for e, ids in elements.items()]
    for step, name, prefix, values in ((1, "DISP", "D", u), (2, "FORC", "F", rf)):
        chunks.append((f"    1PSTEP {step} 1 1\n  100CL 101 1.000000000 {len(values)} 0 1 3\n"
                       f" -4 {name} 4 1\n").encode("ascii"))
        chunks += [f" -5 {prefix}{j} 1 2 {j} 0\n".encode("ascii") for j in range(1, 4)]
        chunks += [b" -5 ALL 1 2 0 0 1ALL\n"]
        chunks += [struct.pack("<i3d", n, *v) for n, v in values.items()]
    chunks.append(b" 9999\n")
    outputs = dict(frd=b"".join(chunks), sta=b"STEP INC ATT ITER TOTAL STEP INCREMENT\n1 1 1 2 1.0 1.0 1.0\n",
                   stdout=b"CalculiX Version 2.23\n", stderr=b"")
    receipt = dict(exit_code=0, timed_out=False, version=ref.CCX_VERSION,
        executable_sha256=ref.CCX_SHA256, case_id=case["case_id"], deck_sha256=metadata["deck_sha256"],
        process_wall_seconds=0., execution_scope="SYNTHETIC_PARSER_CONTROL_NOT_EXECUTED")
    receipt.update({name+"_sha256": hashlib.sha256(data).hexdigest() for name, data in outputs.items()})
    return metadata, outputs, receipt


def test_strict_double_parser_saved_metadata_and_owned_observations():
    case = ref.build_case()
    metadata, output, receipt = _synthetic_output(case)
    saved = json.loads(json.dumps(metadata))
    record = ref.parse_result(case, saved, **output, receipt=receipt)
    assert record["external_work_j"] > 0.
    assert record["scaled_balance"] <= ref.BALANCE_LIMIT
    assert record["production_qualified"] is False
    assert len(record["point_gaps"]) == 2*case["nx"]
    assert len(record["tip_poses"]) == 2
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("failure", ("identity", "hash", "timeout", "exit", "incomplete", "diagnostic", "truncated"))
def test_process_output_refusals(failure):
    case = ref.build_case()
    metadata, output, receipt = _synthetic_output(case)
    if failure == "identity":
        metadata["profile"] = ref.GE
    elif failure == "hash":
        receipt["frd_sha256"] = "0"*64
    elif failure == "timeout":
        receipt["timed_out"] = True
    elif failure == "exit":
        receipt["exit_code"] = 1
    elif failure == "incomplete":
        output["sta"] = b"1 1 1 2 .5 .5 .5\n"
    elif failure == "diagnostic":
        output["stderr"] = b"*ERROR unresolved constraint\n"
    else:
        output["frd"] = output["frd"][:-8]
    if failure in ("incomplete", "diagnostic", "truncated"):
        receipt.update({name+"_sha256": hashlib.sha256(data).hexdigest() for name, data in output.items()})
    with pytest.raises(ref.ReferenceParseError):
        ref.parse_result(case, metadata, **output, receipt=receipt)


def _comparison_records(level):
    case = ref.build_case(level)
    common = dict(profile=ref.B2, case_id=case["case_id"], level=level, offset_sign=1,
        external_work_j=1e-6*(1+1e-4*(2-level)),
        tip_poses=[dict(rib_index=i, world_pose=[0., 0., -1e-6, 0., 0., 0.]) for i in range(2)],
        root_reaction_wrench_world_origin=[0., 0., 1., 0., -.254, 0.])
    return (dict(common, status="COMPLETED_UNQUALIFIED_B2_CANDIDATE"),
            dict(common, status="PARSED_UNQUALIFIED_B2_CANDIDATE"))


def test_prepared_comparison_requires_three_levels_and_keeps_full_scope_open():
    pairs = [_comparison_records(i) for i in range(3)]
    result = ref.convergence_candidate([a for a, _ in pairs], [b for _, b in pairs],
                                       mapping_review_id="synthetic-review-control")
    assert result["within_prepared_limits"] is True
    assert result["production_qualified"] is False
    assert result["ge_reference"].startswith("REFUSED")
    with pytest.raises(ref.ReferenceParseError, match="three"):
        ref.convergence_candidate([pairs[0][0]], [pairs[0][1]], mapping_review_id="control")
    a, b = pairs[0]
    with pytest.raises(ref.ReferenceParseError, match="review"):
        ref.comparison_candidate(a, b, mapping_review_id="")
    with pytest.raises(ref.ReferenceParseError, match="GE"):
        ref.comparison_candidate(dict(a, profile=ref.GE), b, mapping_review_id="control")
    with pytest.raises(ValueError, match="drill"):
        native.refuse_ge_reference_scoring()


def test_balance_uses_declared_force_length_scales_not_comparison_floors():
    case = ref.build_case()
    assert ref.balance_measure(case, [1e-8, 0., 0.], [0., 0., 0.]) == pytest.approx(1e-8)
    assert ref.balance_measure(case, [0., 0., 0.], [0., 1e-8*case["length_m"], 0.]) == pytest.approx(1e-8)


def test_receipt_nested_mutation_isolation():
    case = ref.build_case()
    metadata, output, receipt = _synthetic_output(case)
    receipt["effective_settings"] = {"thread_pools": [{"threads": 1}], "notes": []}
    record = ref.parse_result(case, metadata, **output, receipt=receipt)
    receipt["effective_settings"]["thread_pools"][0]["threads"] = 8
    assert record["receipt"]["effective_settings"]["thread_pools"][0]["threads"] == 1
    record["receipt"]["effective_settings"]["notes"].append("record mutation")
    assert receipt["effective_settings"]["notes"] == []
    assert record["receipt"] is not receipt


@pytest.mark.parametrize("extra", (
    np.float64(1.), object(), b"bytes", {"set"},
    float("nan"), float("inf"), {1: "non-string key"}, (1,),
))
def test_receipt_non_json_or_nonfinite_extras_refused(extra):
    case = ref.build_case()
    metadata, output, receipt = _synthetic_output(case)
    receipt["extra"] = extra
    with pytest.raises(ref.ReferenceParseError):
        ref.parse_result(case, metadata, **output, receipt=receipt)


@pytest.mark.parametrize("bound", ("string", "key", "nodes", "depth", "integer", "bytes"))
def test_receipt_explicit_bounds_refused(bound):
    case = ref.build_case()
    metadata, output, receipt = _synthetic_output(case)
    if bound == "string":
        receipt["extra"] = "a"*4097
    elif bound == "key":
        receipt["a"*257] = None
    elif bound == "nodes":
        receipt["extra"] = [0]*4096
    elif bound == "depth":
        nested = None
        for _ in range(17):
            nested = [nested]
        receipt["extra"] = nested
    elif bound == "integer":
        receipt["extra"] = 10**21
    else:
        receipt["extra"] = ["a"*4096]*32
    with pytest.raises(ref.ReferenceParseError):
        ref.parse_result(case, metadata, **output, receipt=receipt)


def test_receipt_cycles_refused():
    case = ref.build_case()
    metadata, output, receipt = _synthetic_output(case)
    receipt["extra"] = receipt
    with pytest.raises(ref.ReferenceParseError, match="cyclic"):
        ref.parse_result(case, metadata, **output, receipt=receipt)


def test_tip_comparison_matches_declared_ids_when_records_reordered():
    a, b = _comparison_records(0)
    # Give the two physical ribs distinct observations so list-order matching fails.
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    a["tip_poses"][0]["world_pose"][2] *= 2
    b["tip_poses"][0]["world_pose"][2] *= 2
    expected = ref.comparison_candidate(a, b, mapping_review_id="synthetic-review-control")
    b["tip_poses"].reverse()
    actual = ref.comparison_candidate(a, b, mapping_review_id="synthetic-review-control")
    assert actual == expected
    assert actual["tip_translation_difference_m"] == [[0., 0., 0.], [0., 0., 0.]]


@pytest.mark.parametrize("side", ("native", "external"))
@pytest.mark.parametrize("ids", ([0, 0], [1, 1], [0, 2], [0], [False, 1], ["0", 1], [None, 1]))
def test_duplicate_missing_or_foreign_rib_ids_refused(side, ids):
    a, b = _comparison_records(0)
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    record = a if side == "native" else b
    tips = []
    for rib in ids:
        tip = {"world_pose": [0., 0., -1e-6, 0., 0., 0.]}
        if rib is not None:
            tip["rib_index"] = rib
        tips.append(tip)
    record["tip_poses"] = tips
    with pytest.raises(ref.ReferenceParseError, match="complete tip"):
        ref.comparison_candidate(a, b, mapping_review_id="synthetic-review-control")
