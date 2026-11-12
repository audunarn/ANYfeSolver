from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "docs" / "reference_cases"
if str(REFERENCE_CASES) not in sys.path:
    sys.path.insert(0, str(REFERENCE_CASES))

import e4_pl_s3_v2_flat_funnel as funnel
import e4_pl_s3_v2_flat_funnel_producer as producer


def _leaf_authority(
    *,
    archive_sha256: str = "A" * 64,
    candidate_commit: str = "1" * 40,
    candidate_tree: str = "2" * 40,
    producer_program_sha256: str | None = None,
) -> dict[str, str]:
    return {
        "candidate_archive_sha256": archive_sha256,
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "producer_program_sha256": (
            funnel.sha256(Path(producer.__file__).read_bytes())
            if producer_program_sha256 is None
            else producer_program_sha256
        ),
    }


def _plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest, raw = funnel.strict_json_load(producer.MANIFEST_PATH)
    records = funnel.validate_manifest(manifest, raw)
    plan = funnel.build_phase_plan(records, "4A")
    path = tmp_path / "phase4a-plan.json"
    path.write_bytes(funnel.canonical_bytes(plan))
    return path, plan


def _tiny_q4_model() -> tuple[object, dict[int, str]]:
    from anyfesolver.boundary import LoadCase
    from anyfesolver.elements import create_shell_element
    from anyfesolver.fe_core import FEModel

    level = 2
    model = FEModel("tiny-phase4a-q4")
    model.add_material(
        "phase4a_steel",
        producer.ELASTIC_MODULUS,
        producer.POISSON_RATIO,
        density=producer.DENSITY,
    )
    for j in range(level + 1):
        for i in range(level + 1):
            model.add_node(
                producer._node_id(i, j, level),
                i / level,
                j / level,
                0.0,
            )
    kinds: dict[int, str] = {}
    element_id = 0
    for j in range(level):
        for i in range(level):
            element_id += 1
            nodes = (
                producer._node_id(i, j, level),
                producer._node_id(i + 1, j, level),
                producer._node_id(i + 1, j + 1, level),
                producer._node_id(i, j + 1, level),
            )
            element = create_shell_element(
                element_id,
                list(nodes),
                "phase4a_steel",
                formulation="e4-pl",
                thickness=producer.THICKNESS,
                reference_normal=np.asarray((0.0, 0.0, 1.0)),
                drilling_stabilization=0.001,
                hourglass_stabilization=0.001,
                pl_stabilization=1.0,
                planar_tolerance=1.0e-10,
                warped_formulation="varying_frame",
            )
            model.add_element(element_id, element)
            kinds[element_id] = "Q4"
    producer._hard_navier_supports(model, level)
    load = LoadCase("tiny-uniform-pressure")
    for registered_id in model.mesh.elements:
        load.add_pressure_load(registered_id, producer.PRESSURE)
    model.add_load_case(load)
    return model, kinds


def _tiny_mixed_model() -> tuple[object, list[object], object]:
    from anyfesolver.elements import create_shell_element
    from anyfesolver.fe_core import FEModel

    model = FEModel("tiny-phase4a-qualified-q4-v2a")
    model.add_material(
        "phase4a_steel",
        producer.ELASTIC_MODULUS,
        producer.POISSON_RATIO,
        density=producer.DENSITY,
    )
    for node_id, coordinates in enumerate(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        start=1,
    ):
        model.add_node(node_id, *coordinates)
    normal = np.asarray((0.0, 0.0, 1.0))
    q4 = create_shell_element(
        1,
        [1, 2, 5, 4],
        "phase4a_steel",
        formulation="e4-pl",
        thickness=producer.THICKNESS,
        reference_normal=normal,
        drilling_stabilization=0.001,
        hourglass_stabilization=0.001,
        pl_stabilization=1.0,
        planar_tolerance=1.0e-10,
        warped_formulation="varying_frame",
    )
    triangles = [
        create_shell_element(
            element_id,
            list(nodes),
            "phase4a_steel",
            formulation=producer.SELECTOR,
            thickness=producer.THICKNESS,
            reference_normal=normal,
        )
        for element_id, nodes in ((2, (2, 3, 6)), (3, (2, 6, 5)))
    ]
    model.add_element(1, q4)
    for triangle in triangles:
        model.add_element(triangle.element_id, triangle)
    return model, triangles, q4


def test_assignment_is_exact_phase4a_diagonal_shard(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    made_plan, shard, digest = producer.load_assignment(
        plan_path,
        shard_index=1,
        selector="e4-pl-s3-v2",
    )
    assert made_plan == plan
    assert shard["diagonal"] == "backslash"
    assert len(shard["records"]) == 27
    assert digest == funnel.sha256(plan_path.read_bytes())
    with pytest.raises(funnel.FlatFunnelError, match="only the exact"):
        producer.load_assignment(plan_path, shard_index=1, selector="e4-pl-s3")


def test_leaf_catalog_is_exact_content_addressed_81_v2_computations(
    tmp_path: Path,
) -> None:
    plan_path, plan = _plan(tmp_path)
    plan_digest = funnel.sha256(plan_path.read_bytes())
    authority = _leaf_authority()
    catalog = producer.build_leaf_catalog(plan, plan_digest, **authority)
    repeated = producer.build_leaf_catalog(plan, plan_digest, **authority)

    assert catalog == repeated
    assert len(catalog) == 81
    assert len({item["leaf_id"] for item in catalog}) == 81
    assert len({item["leaf_assignment_sha256"] for item in catalog}) == 81
    assert len({item["assignment"]["record_id"] for item in catalog}) == 81
    assert sum(
        item["assignment"]["computation_role"]
        == producer.V2_COMPUTATION_ROLE
        for item in catalog
    ) == 81
    assert [item["assignment"]["catalog_index"] for item in catalog] == list(
        range(81)
    )
    assert catalog[0]["assignment"]["record_id"] == "N20:0PCT:none:slash"
    assert catalog[0]["assignment"]["computation_role"] == (
        producer.V2_COMPUTATION_ROLE
    )
    assert catalog[1]["assignment"]["record_id"] == (
        "N20:1PCT:dispersed:slash"
    )
    assert catalog[1]["assignment"]["computation_role"] == (
        producer.V2_COMPUTATION_ROLE
    )
    assert catalog[-1]["assignment"]["record_id"] == (
        "N80:25PCT:chain:alternating"
    )
    assert catalog[-1]["assignment"]["computation_role"] == (
        producer.V2_COMPUTATION_ROLE
    )
    logical_indices = {
        item["assignment"]["record_id"]: item["assignment"][
            "logical_record_index"
        ]
        for item in catalog
    }
    assert sorted(set(logical_indices.values())) == list(range(81))
    for item in catalog:
        assert item["assignment"]["computation_role"] == (
            producer.V2_COMPUTATION_ROLE
        )
        assert item["assignment"]["s3_selector"] == producer.SELECTOR
        assert "v1_diagnostic_expected" not in item["assignment"]
        assert {
            key: item["assignment"][key]
            for key in producer.LEAF_CANDIDATE_AUTHORITY_KEYS
        } == authority
        digest = funnel.sha256(funnel.canonical_bytes(item["assignment"]))
        assert item["leaf_assignment_sha256"] == digest
        assert item["leaf_id"] == f"S3_V2_FLAT_4A_LEAF_{digest}"

    successor_authority = {
        **authority,
        "candidate_commit": "3" * 40,
    }
    successor = producer.build_leaf_catalog(
        plan, plan_digest, **successor_authority
    )
    assert {
        item["leaf_assignment_sha256"] for item in catalog
    }.isdisjoint(item["leaf_assignment_sha256"] for item in successor)

    with pytest.raises(funnel.FlatFunnelError, match="lowercase 40-hex"):
        producer.build_leaf_catalog(
            plan,
            plan_digest,
            **{**authority, "candidate_commit": "A" * 40},
        )

    first_shard = plan["shards"][0]
    all_q4_member = first_shard["records"][0]
    with pytest.raises(funnel.FlatFunnelError, match="must be V2"):
        producer.leaf_assignment_core(
            plan_digest,
            first_shard,
            all_q4_member,
            catalog_index=0,
            logical_record_index=0,
            computation_role=producer.V1_COMPUTATION_ROLE,
            **authority,
        )
    with pytest.raises(funnel.FlatFunnelError, match="must be V2"):
        producer.leaf_assignment_core(
            plan_digest,
            first_shard,
            all_q4_member,
            catalog_index=0,
            logical_record_index=0,
            computation_role="V2_OR_V1_AMBIGUOUS",
            **authority,
        )

    backslash_leaf = next(
        item
        for item in catalog
        if item["assignment"]["diagonal"] == "backslash"
    )
    made_plan, shard, member, leaf, made_digest = producer.load_leaf_assignment(
        plan_path,
        leaf_assignment_sha256=backslash_leaf["leaf_assignment_sha256"],
        selector=producer.SELECTOR,
        **authority,
    )
    assert made_plan == plan
    assert shard["diagonal"] == "backslash"
    assert member["record_id"] == leaf["assignment"]["record_id"]
    assert made_digest == plan_digest

    with pytest.raises(funnel.FlatFunnelError, match="uppercase SHA-256"):
        producer.load_leaf_assignment(
            plan_path,
            leaf_assignment_sha256=catalog[0]["leaf_assignment_sha256"].lower(),
            selector=producer.SELECTOR,
            **authority,
        )
    with pytest.raises(funnel.FlatFunnelError, match="absent or duplicated"):
        producer.load_leaf_assignment(
            plan_path,
            leaf_assignment_sha256="A" * 64,
            selector=producer.SELECTOR,
            **authority,
        )
    with pytest.raises(funnel.FlatFunnelError, match="only the exact"):
        producer.load_leaf_assignment(
            plan_path,
            leaf_assignment_sha256=catalog[0]["leaf_assignment_sha256"],
            selector="e4-pl-s3",
            **authority,
        )


def test_formal_leaves_run_exactly_one_v2_formulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    catalog = producer.build_leaf_catalog(
        plan,
        funnel.sha256(plan_path.read_bytes()),
        **_leaf_authority(archive_sha256="B" * 64),
    )
    baseline = catalog[0]
    mixed_v2 = catalog[1]
    calls: list[tuple[str, str]] = []
    events: list[str] = []

    def fake_case(member: dict[str, object], *, s3_selector: str) -> dict[str, object]:
        calls.append((str(member["record_id"]), s3_selector))
        return {
            "classification": producer.CLASSIFICATION,
            "record_id": str(member["record_id"]),
        }

    monkeypatch.setattr(producer, "produce_case", fake_case)
    real_append = producer._append_reserved_progress

    def observed_append(descriptor: int, record: dict[str, object]) -> None:
        events.append(str(record["phase"]))
        real_append(descriptor, record)

    monkeypatch.setattr(producer, "_append_reserved_progress", observed_append)
    monkeypatch.setattr(
        producer,
        "validate_extracted_candidate_source",
        lambda _root, _archive, _digest: events.append("ARCHIVE_VALIDATED"),
    )
    monkeypatch.setattr(
        producer,
        "activate_frozen_candidate_source",
        lambda _root: events.append("SOURCE_ACTIVATED"),
    )

    for index, leaf in enumerate((baseline, mixed_v2)):
        before = len(calls)
        event_before = len(events)
        output = tmp_path / f"leaf-{index}.json"
        progress = tmp_path / f"leaf-{index}.jsonl"
        document = producer.run_leaf(
            plan_path,
            leaf_assignment_sha256=leaf["leaf_assignment_sha256"],
            selector=producer.SELECTOR,
            candidate_source_root=tmp_path / "candidate-source",
            candidate_archive=tmp_path / "candidate-source.tar",
            **_leaf_authority(archive_sha256="B" * 64),
            output=output,
            progress=progress,
        )
        payload = document["scientific_payload"]
        expected_role = leaf["assignment"]["computation_role"]
        expected_selector = leaf["assignment"]["s3_selector"]
        assert document["schema"] == producer.LEAF_SCIENTIFIC_SCHEMA
        assert document["assignment_sha256"] == leaf["leaf_assignment_sha256"]
        assert document["record_count"] == 1
        assert document["record_ids"] == [leaf["assignment"]["record_id"]]
        assert payload["schema"] == producer.LEAF_PAYLOAD_SCHEMA
        assert payload["leaf_assignment"] == leaf["assignment"]
        assert payload["computation_role"] == expected_role
        assert payload["record"]["record_id"] == leaf["assignment"]["record_id"]
        assert payload["record"]["classification"] == producer.CLASSIFICATION
        assert payload["v1_comparator_disposition"] == (
            producer.FORMAL_V1_DISPOSITION
        )
        assert output.read_bytes() == funnel.canonical_bytes(document)
        assert document["scientific_payload_sha256"] == funnel.sha256(
            funnel.canonical_bytes(payload)
        )
        rows = [json.loads(line) for line in progress.read_text().splitlines()]
        assert [row["phase"] for row in rows] == [
            "INITIALIZATION",
            "AUTHORITY_COMPLETE",
            "CASE_OR_REFINEMENT_OR_STATION",
            "STAGING",
            "VALIDATION",
            "COMPLETION",
        ]
        assert rows[-1]["completed"] == rows[-1]["total"] == 1
        assert events[event_before:] == [
            "INITIALIZATION",
            "ARCHIVE_VALIDATED",
            "SOURCE_ACTIVATED",
            "AUTHORITY_COMPLETE",
            "CASE_OR_REFINEMENT_OR_STATION",
            "STAGING",
            "VALIDATION",
            "COMPLETION",
        ]
        selectors = [selector for _record_id, selector in calls[before:]]
        assert selectors == [expected_selector]
        assert expected_role == producer.V2_COMPUTATION_ROLE
        assert expected_selector == producer.SELECTOR


def test_leaf_v2_failure_never_launches_v1_or_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    mixed = next(
        item
        for item in producer.build_leaf_catalog(
            plan,
            funnel.sha256(plan_path.read_bytes()),
            **_leaf_authority(archive_sha256="C" * 64),
        )
        if item["assignment"]["record_id"]
        == "N20:1PCT:dispersed:slash"
    )
    selectors: list[str] = []

    def fail_v2(_member: object, *, s3_selector: str) -> object:
        selectors.append(s3_selector)
        raise RuntimeError("synthetic leaf V2 contradiction")

    monkeypatch.setattr(producer, "produce_case", fail_v2)
    monkeypatch.setattr(
        producer,
        "validate_extracted_candidate_source",
        lambda _root, _archive, _digest: None,
    )
    monkeypatch.setattr(
        producer, "activate_frozen_candidate_source", lambda _root: None
    )
    output = tmp_path / "must-not-exist-leaf.json"
    with pytest.raises(RuntimeError, match="synthetic leaf V2"):
        producer.run_leaf(
            plan_path,
            leaf_assignment_sha256=mixed["leaf_assignment_sha256"],
            selector=producer.SELECTOR,
            candidate_source_root=tmp_path / "candidate-source",
            candidate_archive=tmp_path / "candidate-source.tar",
            **_leaf_authority(archive_sha256="C" * 64),
            output=output,
            progress=tmp_path / "failed-leaf-progress.jsonl",
        )
    assert selectors == [producer.SELECTOR]
    assert not output.exists()


def test_formal_leaf_catalog_never_registers_v1_diagnostic() -> None:
    plan = {
        "phase": "4A",
        "record_count": 81,
        "scope": "full",
        "selector": producer.SELECTOR,
        "shards": [],
    }
    assert producer.V1_COMPUTATION_ROLE not in {
        producer.V2_COMPUTATION_ROLE
    }
    assert producer.FORMAL_V1_DISPOSITION.endswith("NO_FALLBACK")
    with pytest.raises(funnel.FlatFunnelError, match="three diagonal"):
        producer.build_leaf_catalog(plan, "A" * 64, **_leaf_authority())


def test_leaf_rejects_wrong_program_identity_before_scientific_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    authority = _leaf_authority(archive_sha256="D" * 64)
    leaf = producer.build_leaf_catalog(
        plan, funnel.sha256(plan_path.read_bytes()), **authority
    )[0]
    launched: list[str] = []
    monkeypatch.setattr(
        producer,
        "produce_case",
        lambda *_args, **_kwargs: launched.append("CASE"),
    )
    progress = tmp_path / "wrong-program-progress.jsonl"
    with pytest.raises(funnel.FlatFunnelError, match="program hash differs"):
        producer.run_leaf(
            plan_path,
            leaf_assignment_sha256=leaf["leaf_assignment_sha256"],
            selector=producer.SELECTOR,
            candidate_source_root=tmp_path / "candidate-source",
            candidate_archive=tmp_path / "candidate-source.tar",
            **{**authority, "producer_program_sha256": "F" * 64},
            output=tmp_path / "wrong-program.json",
            progress=progress,
        )
    assert launched == []
    assert not progress.exists()


def test_leaf_authority_completes_only_after_archive_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    authority = _leaf_authority(archive_sha256="E" * 64)
    leaf = producer.build_leaf_catalog(
        plan, funnel.sha256(plan_path.read_bytes()), **authority
    )[0]
    reached: list[str] = []

    def reject_archive(*_args: object) -> None:
        reached.append("ARCHIVE_REJECTED")
        raise funnel.FlatFunnelError("synthetic archive rejection")

    monkeypatch.setattr(
        producer, "validate_extracted_candidate_source", reject_archive
    )
    monkeypatch.setattr(
        producer,
        "activate_frozen_candidate_source",
        lambda _root: reached.append("SOURCE_ACTIVATED"),
    )
    monkeypatch.setattr(
        producer,
        "produce_case",
        lambda *_args, **_kwargs: reached.append("CASE"),
    )
    output = tmp_path / "rejected-archive.json"
    progress = tmp_path / "rejected-archive.jsonl"
    with pytest.raises(funnel.FlatFunnelError, match="synthetic archive"):
        producer.run_leaf(
            plan_path,
            leaf_assignment_sha256=leaf["leaf_assignment_sha256"],
            selector=producer.SELECTOR,
            candidate_source_root=tmp_path / "candidate-source",
            candidate_archive=tmp_path / "candidate-source.tar",
            **authority,
            output=output,
            progress=progress,
        )
    rows = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["INITIALIZATION"]
    assert reached == ["ARCHIVE_REJECTED"]
    assert not output.exists()


def test_leaf_progress_is_reserved_as_exclusive_regular_file(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress.jsonl"
    descriptor = producer._reserve_progress_exclusive(progress)
    try:
        information = os.fstat(descriptor)
        assert stat.S_ISREG(information.st_mode)
        assert not (
            getattr(information, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    finally:
        os.close(descriptor)
    assert progress.read_bytes() == b""
    with pytest.raises(funnel.FlatFunnelError, match="overwrite progress"):
        producer._reserve_progress_exclusive(progress)

    dangling = tmp_path / "dangling-progress.jsonl"
    try:
        dangling.symlink_to(tmp_path / "absent-target.jsonl")
    except OSError:
        return
    with pytest.raises(funnel.FlatFunnelError, match="overwrite progress"):
        producer._reserve_progress_exclusive(dangling)


def test_mindlin_nodal_input_is_deterministic_and_satisfies_hard_trace() -> None:
    vector, digest, center = producer.mindlin_nodal_reference(2)
    repeated, repeated_digest, repeated_center = producer.mindlin_nodal_reference(2)
    np.testing.assert_array_equal(vector, repeated)
    assert digest == repeated_digest
    assert center == repeated_center
    assert len(digest) == 64 and digest == digest.upper()
    rows = vector.reshape(9, 6)
    for j in range(3):
        for i in range(3):
            row = rows[producer._node_id(i, j, 2) - 1]
            if i in (0, 2) or j in (0, 2):
                np.testing.assert_array_equal(row[:3], np.zeros(3))
            if i in (0, 2):
                assert row[3] == 0.0
            if j in (0, 2):
                assert row[4] == 0.0
    assert center != 0.0


def test_reference_uses_correct_shell_embedding_and_v3_protocol() -> None:
    vector, _digest, _center = producer.mindlin_nodal_reference(2)
    document, _checker_center = __import__(
        "e4_pl_s3_v2_flat_funnel_checker"
    ).reference_vector_document(2)
    np.testing.assert_array_equal(vector, np.asarray(document["values"]))
    assert producer.REFERENCE_IDENTITY.endswith("SHELL_EMBEDDED_V3")
    assert producer.SUPPORT_IDENTITY.endswith("SHELL_ROTATIONS_V3")


def test_tiny_mixed_q4_v2a_model_has_authoritative_directors_and_passes_guard() -> None:
    model, triangles, q4 = _tiny_mixed_model()
    normal = np.asarray((0.0, 0.0, 1.0))
    np.testing.assert_array_equal(q4.reference_normal, normal)
    material = model.get_material("phase4a_steel")
    for triangle in triangles:
        components = triangle.compute_stiffness_components(model.mesh, material)
        assert {"physical", "pl", "total"} <= set(components)
        assert all(
            components[key].shape == (18, 18)
            and np.all(np.isfinite(components[key]))
            for key in ("physical", "pl", "total")
        )


def test_tiny_q4_production_solve_closes_discrete_energy_identity() -> None:
    model, kinds = _tiny_q4_model()
    reference, _digest, _center = producer.mindlin_nodal_reference(2)
    solver, measured, solution = producer._solve_and_measure(model, kinds, reference)
    assert solver["status"] == "CONVERGED_DIRECT_SPARSE"
    assert solver["total_dofs"] == 54
    assert solver["free_dofs"] > 0
    assert solver["residual_relative"] < 1.0e-10
    assert np.all(np.isfinite(solution))
    recomputed = (
        measured["solution_total"]
        + measured["reference_total"]
        - 2.0 * measured["solution_reference_cross"]
    )
    assert measured["error_total"] == pytest.approx(max(recomputed, 0.0), rel=2.0e-13)
    assert measured["strain_total"] == pytest.approx(
        measured["strain_physical"]
        + measured["strain_q4_pl"]
        + measured["strain_s3_pl"]
        + measured["strain_q4_hourglass"],
        rel=2.0e-12,
    )
    assert measured["strain_s3_pl"] == 0.0


def test_three_fake_shards_cover_exact_81_classifying_and_72_v1_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan = _plan(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_case(member: dict[str, object], *, s3_selector: str) -> dict[str, object]:
        calls.append((str(member["record_id"]), s3_selector))
        return {
            "classification": (
                producer.CLASSIFICATION
                if s3_selector == producer.SELECTOR
                else "NONCLASSIFYING_V1_COMPARATOR_ONLY"
            ),
            "record_id": str(member["record_id"]),
        }

    monkeypatch.setattr(producer, "produce_case", fake_case)
    monkeypatch.setattr(
        producer,
        "validate_extracted_candidate_source",
        lambda _root, _archive, _digest: None,
    )
    monkeypatch.setattr(
        producer, "activate_frozen_candidate_source", lambda _root: None
    )
    documents = []
    for shard_index in range(3):
        documents.append(
            producer.run_assignment(
                plan_path,
                shard_index=shard_index,
                selector=producer.SELECTOR,
                candidate_source_root=tmp_path / "candidate-source",
                candidate_archive=tmp_path / "candidate-source.tar",
                candidate_archive_sha256="A" * 64,
                output=tmp_path / f"scientific-{shard_index}.json",
                progress=tmp_path / f"progress-{shard_index}.jsonl",
            )
        )
    payloads = [document["scientific_payload"] for document in documents]
    assert sum(len(payload["classifying_records"]) for payload in payloads) == 81
    assert sum(len(payload["v1_comparator_diagnostics"]) for payload in payloads) == 72
    assert len({row["record_id"] for payload in payloads for row in payload["classifying_records"]}) == 81
    assert {payload["diagonal"] for payload in payloads} == set(funnel.DIAGONALS)
    assert all(
        payload["v1_comparator_disposition"] == producer.V1_DISPOSITION
        for payload in payloads
    )
    assert sum(selector == producer.SELECTOR for _record, selector in calls) == 81
    assert sum(selector == "e4-pl-s3" for _record, selector in calls) == 72
    for shard_index, document in enumerate(documents):
        raw = (tmp_path / f"scientific-{shard_index}.json").read_bytes()
        assert raw == funnel.canonical_bytes(document)
        assert document["assignment_sha256"] == plan["shards"][shard_index]["assignment_sha256"]
        progress_rows = [
            json.loads(line)
            for line in (tmp_path / f"progress-{shard_index}.jsonl").read_text().splitlines()
        ]
        assert len(progress_rows) == 32
        assert progress_rows[0]["completed"] == 0
        assert progress_rows[-1]["completed"] == 27
        required = [
            "INITIALIZATION",
            "AUTHORITY_COMPLETE",
            "CASE_OR_REFINEMENT_OR_STATION",
            "STAGING",
            "VALIDATION",
            "COMPLETION",
        ]
        phases = [row["phase"] for row in progress_rows]
        assert all(phase in phases for phase in required)
        assert [phases.index(phase) for phase in required] == sorted(
            phases.index(phase) for phase in required
        )


def test_v2_failure_never_launches_v1_or_publishes_canonical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _plan_value = _plan(tmp_path)
    selectors: list[str] = []

    def fail_v2(_member: object, *, s3_selector: str) -> object:
        selectors.append(s3_selector)
        raise RuntimeError("synthetic V2 contradiction")

    monkeypatch.setattr(producer, "produce_case", fail_v2)
    monkeypatch.setattr(
        producer,
        "validate_extracted_candidate_source",
        lambda _root, _archive, _digest: None,
    )
    monkeypatch.setattr(
        producer, "activate_frozen_candidate_source", lambda _root: None
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match="synthetic V2"):
        producer.run_assignment(
            plan_path,
            shard_index=0,
            selector=producer.SELECTOR,
            candidate_source_root=tmp_path / "candidate-source",
            candidate_archive=tmp_path / "candidate-source.tar",
            candidate_archive_sha256="B" * 64,
            output=output,
            progress=tmp_path / "failed-progress.jsonl",
        )
    assert selectors == [producer.SELECTOR]
    assert not output.exists()


def test_cli_requires_exact_shard_or_leaf_interface() -> None:
    parser = producer._parser()
    arguments = parser.parse_args(
        [
            "--run-flat-assignment",
            "plan.json",
            "--shard-index",
            "2",
            "--selector",
            producer.SELECTOR,
            "--candidate-source-root",
            "candidate-source",
            "--candidate-archive",
            "candidate-source.tar",
            "--candidate-archive-sha256",
            "C" * 64,
            "--output",
            "scientific.json",
            "--progress",
            "progress.jsonl",
        ]
    )
    assert arguments.shard_index == 2
    assert arguments.selector == producer.SELECTOR
    assert arguments.candidate_source_root == Path("candidate-source")
    assert arguments.candidate_archive == Path("candidate-source.tar")
    assert arguments.candidate_archive_sha256 == "C" * 64
    leaf_arguments = parser.parse_args(
        [
            "--run-flat-leaf",
            "plan.json",
            "--leaf-assignment-sha256",
            "D" * 64,
            "--selector",
            producer.SELECTOR,
            "--candidate-source-root",
            "candidate-source",
            "--candidate-archive",
            "candidate-source.tar",
            "--candidate-archive-sha256",
            "E" * 64,
            "--candidate-commit",
            "1" * 40,
            "--candidate-tree",
            "2" * 40,
            "--producer-program-sha256",
            "F" * 64,
            "--output",
            "leaf.json",
            "--progress",
            "leaf.jsonl",
        ]
    )
    assert leaf_arguments.run_flat_leaf == Path("plan.json")
    assert leaf_arguments.shard_index is None
    assert leaf_arguments.leaf_assignment_sha256 == "D" * 64
    assert leaf_arguments.candidate_commit == "1" * 40
    assert leaf_arguments.candidate_tree == "2" * 40
    assert leaf_arguments.producer_program_sha256 == "F" * 64

    with pytest.raises(funnel.FlatFunnelError, match="complete authority"):
        producer.main(
            [
                "--run-flat-leaf",
                "plan.json",
                "--leaf-assignment-sha256",
                "D" * 64,
                "--selector",
                producer.SELECTOR,
                "--candidate-source-root",
                "candidate-source",
                "--candidate-archive",
                "candidate-source.tar",
                "--candidate-archive-sha256",
                "E" * 64,
                "--output",
                "leaf.json",
                "--progress",
                "leaf.jsonl",
            ]
        )


def test_candidate_archive_must_exactly_match_extracted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate-source"
    package = candidate / "src" / "anyfesolver"
    package.mkdir(parents=True)
    source = b'FORMULATION = "S3_V2A"\n'
    (package / "__init__.py").write_bytes(source)
    generator = candidate / "docs" / "reference_cases" / "e4_pl_s3_mixed_mesh_manifest.py"
    generator.parent.mkdir(parents=True)
    generator_source = b'ORIGIN = "EXTRACTED_CANDIDATE"\n'
    generator.write_bytes(generator_source)
    archive = tmp_path / "candidate-source.tar"
    with tarfile.open(archive, mode="w:") as bundle:
        for name in ("src", "src/anyfesolver", "docs", "docs/reference_cases"):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            bundle.addfile(directory)
        member = tarfile.TarInfo("src/anyfesolver/__init__.py")
        member.size = len(source)
        bundle.addfile(member, io.BytesIO(source))
        member = tarfile.TarInfo(
            "docs/reference_cases/e4_pl_s3_mixed_mesh_manifest.py"
        )
        member.size = len(generator_source)
        bundle.addfile(member, io.BytesIO(generator_source))
    digest = funnel.sha256(archive.read_bytes())

    producer.validate_extracted_candidate_source(candidate, archive, digest)
    monkeypatch.setattr(producer, "_ACTIVE_CANDIDATE_ROOT", candidate)
    loaded = producer._load_manifest_generator()
    assert loaded.ORIGIN == "EXTRACTED_CANDIDATE"
    assert Path(loaded.__file__).resolve() == generator.resolve()
    (package / "__init__.py").write_bytes(b'FORMULATION = "MUTATED"\n')
    with pytest.raises(funnel.FlatFunnelError, match="differs"):
        producer.validate_extracted_candidate_source(candidate, archive, digest)


def test_scientific_output_publication_is_atomic_and_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "scientific.json"
    raw = funnel.canonical_bytes({"terminal": "TEST_ONLY"})
    producer._publish_exclusive(output, raw)
    assert output.read_bytes() == raw
    assert not list(tmp_path.glob(".scientific.json.pending-*"))
    with pytest.raises(funnel.FlatFunnelError, match="overwrite"):
        producer._publish_exclusive(output, b"different\n")
    assert output.read_bytes() == raw
    assert not list(tmp_path.glob(".scientific.json.pending-*"))
