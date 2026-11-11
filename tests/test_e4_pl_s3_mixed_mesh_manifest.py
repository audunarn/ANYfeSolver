from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "docs" / "reference_cases" / "e4_pl_s3_mixed_mesh_manifest.py"
MANIFEST = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_mesh_connectivity_manifest.json"
)
MANIFEST_SHA256 = "3EA7ABD0B332831D62B30B3CD52E0DB85EC951B125340FFAF40A891DC37BD589"
QUALIFICATION_CONTRACT = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_s3_mixed_mesh_qualification_contract.json"
)
QUALIFICATION_CONTRACT_SHA256 = (
    "3F933786A334A876C1B53EF90891237F18C0970DE6B95F885CF5B775791E0A99"
)
QUALIFIED_Q4_MECHANICS = ROOT / "src" / "anyfesolver" / "e4_pl_element.py"
Q4_REPLAY_HOTFIX = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_q4_0_4_1_replay_hotfix.json"
)
Q4_REPLAY_CANONICALIZATION = (
    ROOT
    / "docs"
    / "reference_cases"
    / "e4_pl_q4_0_4_2_replay_canonicalization.json"
)
Q4_MECHANICS_AUTHORITY_SCHEMA = "anysolver.python-ast-mechanics-closure-v1"
Q4_MECHANICS_CANONICALIZATION = (
    "AST_NODE_ORDERED_FIELDS_COMPACT_JSON_WITHOUT_LOCATIONS_DOCSTRINGS_"
    "EMPTY_TYPE_PARAMS_OR_NONE_TYPE_COMMENTS_WITH_ELLIPSIS_TAG"
)


Q4_SOURCE_BINDING = ROOT / 'docs' / 'reference_cases' / 'e4_pl_q4_source_binding_20261019.json'
Q4_SOURCE_BINDING_SHA256 = '64B07F1A535E6D6EFCEAE0B2EC01BA539DF69BE7956C03DBDA65C2ACF7C7E74F'


def _load_generator():
    spec = importlib.util.spec_from_file_location("e4_pl_s3_mixed_mesh_manifest", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


def _strict_json(raw: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> object:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_mechanics_ast(value: object) -> object:
    if isinstance(value, ast.AST):
        fields: list[list[object]] = []
        for name, item in ast.iter_fields(value):
            if name == "type_comment" and item is None:
                continue
            if name == "type_params" and item == []:
                continue
            if (
                name == "body"
                and isinstance(item, list)
                and item
                and isinstance(item[0], ast.Expr)
                and isinstance(item[0].value, ast.Constant)
                and isinstance(item[0].value.value, str)
            ):
                item = item[1:]
            fields.append([name, _canonical_mechanics_ast(item)])
        return [type(value).__name__, fields]
    if isinstance(value, (list, tuple)):
        return [_canonical_mechanics_ast(item) for item in value]
    if value is Ellipsis:
        return ["Ellipsis"]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported AST field value {type(value).__name__}")


def _mechanics_nodes(source: str) -> dict[str, list[ast.AST]]:
    nodes: dict[str, list[ast.AST]] = {}

    def add(name: str, node: ast.AST) -> None:
        nodes.setdefault(name, []).append(node)

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}{node.name}"
                add(qualified, node)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{qualified}.")
                continue
            targets: list[str] = []
            if isinstance(node, ast.Assign):
                targets = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            for target in targets:
                add(f"{prefix}{target}", node)

    visit(ast.parse(source).body)
    return nodes


def _q4_mechanics_symbol_hashes(
    source: str,
    selected: set[str],
) -> dict[str, str]:
    inventory = _mechanics_nodes(source)
    missing = sorted(name for name in selected if name not in inventory)
    duplicate = sorted(
        name for name in selected if len(inventory.get(name, ())) != 1
    )
    if missing:
        raise ValueError(f"missing selected Q4 mechanics symbols: {missing}")
    if duplicate:
        raise ValueError(f"duplicate selected Q4 mechanics symbols: {duplicate}")
    result: dict[str, str] = {}
    for name in sorted(selected):
        encoded = json.dumps(
            _canonical_mechanics_ast(inventory[name][0]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        result[name] = hashlib.sha256(encoded).hexdigest().upper()
    return result


def _q4_mechanics_local_dependency_edges(
    source: str,
    roots: set[str],
    *,
    required_symbols: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return the deterministic transitive local dependency closure."""

    inventory = _mechanics_nodes(source)
    required = set(roots)
    if required_symbols is not None:
        required.update(required_symbols)
    missing = sorted(name for name in required if name not in inventory)
    duplicate = sorted(
        name for name in required if len(inventory.get(name, ())) != 1
    )
    if missing:
        raise ValueError(
            f"unresolved selected Q4 mechanics local dependencies: {missing}"
        )
    if duplicate:
        raise ValueError(
            f"duplicate selected Q4 mechanics local dependencies: {duplicate}"
        )

    module_symbols = {
        name for name in inventory if "." not in name
    }
    class_prefix = "QualifiedE4PLShellElement."
    class_owners = {
        "QualifiedE4PLShellElement",
        "__class__",
        "cls",
        "self",
    }

    def direct_dependencies(symbol: str) -> list[str]:
        dependencies: set[str] = set()
        node = inventory[symbol][0]
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id in module_symbols
            ):
                dependencies.add(item.id)
                continue
            if not (
                isinstance(item, ast.Attribute)
                and isinstance(item.ctx, ast.Load)
                and isinstance(item.value, ast.Name)
                and item.value.id in class_owners
            ):
                continue
            qualified = f"{class_prefix}{item.attr}"
            if qualified in inventory:
                dependencies.add(qualified)
        dependencies.discard(symbol)
        return sorted(dependencies)

    closure = set(roots)
    pending = sorted(roots)
    edges: dict[str, list[str]] = {}
    while pending:
        symbol = pending.pop(0)
        dependencies = direct_dependencies(symbol)
        edges[symbol] = dependencies
        for dependency in dependencies:
            if len(inventory.get(dependency, ())) != 1:
                raise ValueError(
                    "unresolved selected Q4 mechanics local dependency: "
                    f"{symbol} -> {dependency}"
                )
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
        pending.sort()
    return {name: edges[name] for name in sorted(edges)}


def _require_acyclic_mechanics_graph(edges: dict[str, list[str]]) -> None:
    """Reject a closure that cannot be represented as a hash DAG."""

    visiting: set[str] = set()
    complete: set[str] = set()

    def visit(name: str) -> None:
        if name in complete:
            return
        if name in visiting:
            raise ValueError(f"cyclic Q4 mechanics local dependency at {name}")
        visiting.add(name)
        for dependency in edges[name]:
            if dependency not in edges:
                raise ValueError(
                    "unresolved selected Q4 mechanics local dependency: "
                    f"{name} -> {dependency}"
                )
            visit(dependency)
        visiting.remove(name)
        complete.add(name)

    for symbol in sorted(edges):
        visit(symbol)


def _q4_mechanics_authority_payload(
    source: str,
    roots: set[str],
    *,
    schema: str,
    path: str,
    required_symbols: set[str] | None = None,
) -> dict[str, object]:
    edges = _q4_mechanics_local_dependency_edges(
        source,
        roots,
        required_symbols=required_symbols,
    )
    _require_acyclic_mechanics_graph(edges)
    symbols = _q4_mechanics_symbol_hashes(source, set(edges))
    return {
        "edges": edges,
        "path": path,
        "roots": sorted(roots),
        "schema": schema,
        "symbols": symbols,
    }


def _q4_mechanics_aggregate(payload: dict[str, object]) -> str:
    if set(payload) != {"edges", "path", "roots", "schema", "symbols"}:
        raise ValueError("Q4 mechanics aggregate payload keys are incompatible")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _node_id(i: int, j: int, level: int) -> int:
    return j * (level + 1) + i + 1


def _independent_connectivity_sha256(
    level: int,
    split_cells: set[tuple[int, int]],
    diagonal: str,
) -> str:
    digest = hashlib.sha256(f"level:{level}\n".encode("ascii"))
    element_id = 0
    for j in range(level):
        for i in range(level):
            n00 = _node_id(i, j, level)
            n10 = _node_id(i + 1, j, level)
            n11 = _node_id(i + 1, j + 1, level)
            n01 = _node_id(i, j + 1, level)
            if (i, j) not in split_cells:
                elements = (("Q4", (n00, n10, n11, n01)),)
            else:
                made = diagonal
                if made == "alternating":
                    made = "backslash" if (i + j) % 2 == 0 else "slash"
                if made == "backslash":
                    elements = (
                        ("S3", (n00, n10, n11)),
                        ("S3", (n00, n11, n01)),
                    )
                elif made == "slash":
                    elements = (
                        ("S3", (n00, n10, n01)),
                        ("S3", (n10, n11, n01)),
                    )
                else:  # pragma: no cover - the manifest schema excludes this
                    raise AssertionError(made)
            for kind, node_ids in elements:
                element_id += 1
                nodes = ",".join(str(node_id) for node_id in node_ids)
                digest.update(f"{element_id}:{kind}:{nodes}\n".encode("ascii"))
    return digest.hexdigest().upper()


def _signed_area_twice(node_ids: tuple[int, ...], level: int) -> int:
    stride = level + 1
    points = tuple(((node_id - 1) % stride, (node_id - 1) // stride) for node_id in node_ids)
    return sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )


def test_generator_is_standard_library_only_and_mechanics_free() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots == {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "pathlib",
        "typing",
    }


def test_manifest_is_strict_canonical_and_bound_to_generator() -> None:
    raw = MANIFEST.read_bytes()
    payload = _strict_json(raw)
    assert hashlib.sha256(raw).hexdigest().upper() == MANIFEST_SHA256
    assert raw == _canonical_bytes(payload)
    assert payload == GEN.build_manifest()
    assert payload["schema"] == GEN.SCHEMA
    assert payload["connectivity_hash_encoding"] == GEN.HASH_ENCODING


def test_qualification_contract_is_canonical_and_binds_connectivity() -> None:
    raw = QUALIFICATION_CONTRACT.read_bytes()
    contract = _strict_json(raw)
    assert hashlib.sha256(raw).hexdigest().upper() == QUALIFICATION_CONTRACT_SHA256

    canonical_pretty = (
        json.dumps(
            contract,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert raw == canonical_pretty
    authority = contract["connectivity_authority"]
    manifest_raw = MANIFEST.read_bytes()
    manifest = _strict_json(manifest_raw)
    assert authority == {
        "bytes": len(manifest_raw),
        "gated_record_count": len(manifest["records"]),
        "path": (
            "docs/reference_cases/e4_pl_s3_mixed_mesh_connectivity_manifest.json"
        ),
        "research_control_record_count": len(
            manifest["research_control"]["records"]
        ),
        "schema": manifest["schema"],
        "sha256": hashlib.sha256(manifest_raw).hexdigest().upper(),
    }


def _require_prospective_q4_source_binding(
    record_raw: bytes,
    source_raw: bytes,
    actual: dict[str, object],
    authority: dict[str, object],
) -> dict[str, object]:
    assert hashlib.sha256(record_raw).hexdigest().upper() == Q4_SOURCE_BINDING_SHA256
    record = _strict_json(record_raw)
    assert record['schema'] == 'anysolver.q4-source-binding-successor-v1'
    assert record['qualification_boundary'] == 'SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION'
    predecessor = record['previous_source_binding']
    assert predecessor['path'] == 'docs/reference_cases/e4_pl_q4_source_binding_20261016.json'
    assert predecessor['sha256'].upper() == 'BE903D0381248BEC56B9C48E8EF90F119B0F15E133F1AEBB9A5B89C341B8AA1B'
    assert hashlib.sha256((ROOT / predecessor['path']).read_bytes()).hexdigest() == predecessor['sha256']
    integration = record['source_binding_integration_review']
    assert integration['external_carrier_changed_since_predecessor'] is True
    assert integration['product_edited_in_this_binding_slice'] is False
    assert all(integration[key] is False for key in ('physical_acceptance_transfer', 'C5_exception_transfer', 'CI_credit'))
    assert hashlib.sha256((ROOT / integration['review_path']).read_bytes()).hexdigest() == integration['review_sha256']
    lifecycle = integration['lifecycle_delta']
    assert hashlib.sha256((ROOT / lifecycle['path']).read_bytes()).hexdigest() == lifecycle['sha256']
    assert record['source']['path'] == QUALIFIED_Q4_MECHANICS.relative_to(ROOT).as_posix()
    assert hashlib.sha256(source_raw).hexdigest().upper() == record['source']['sha256']
    source = source_raw.decode('utf-8')
    assert actual['roots'] == authority['roots'] == record['mechanics']['roots']
    assert len(actual['roots']) == 47
    assert set(authority['symbols']) <= set(actual['symbols'])
    assert len(authority['symbols']) == 129
    assert actual == record['mechanics']
    assert _q4_mechanics_aggregate(actual) == record['mechanics_aggregate_sha256']
    assert record['historical_current_source_matches'] is False
    for relative, digest in record['historical_authorities'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper() == digest
    assert set(record['historical_authorities']) == {
        QUALIFICATION_CONTRACT.relative_to(ROOT).as_posix(),
        Q4_REPLAY_HOTFIX.relative_to(ROOT).as_posix(),
        Q4_REPLAY_CANONICALIZATION.relative_to(ROOT).as_posix(),
    }
    inventory = _mechanics_nodes(source)
    expected_external = {
        '_elastic_increment_module': 'src/anyfesolver/vectorized_nonlinear.py',
        '_local_contractions': 'src/anyfesolver/_compensated_contractions.py',
        '_elements_module': 'src/anyfesolver/elements.py',
    }
    assert set(record['external_arithmetic_dependencies']) == set(expected_external)
    for alias, relative in expected_external.items():
        row = record['external_arithmetic_dependencies'][alias]
        assert row['path'] == relative and row['changed_this_slice'] is False
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper() == row['sha256'].upper()
        used = sorted({
            node.attr
            for symbol in actual['symbols']
            for node in ast.walk(inventory[symbol][0])
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == alias
        })
        assert used == row['used_attributes']
    observed = record['observed_focused_pre_reader_integration']
    assert observed['status'] == 'PASS'
    assert observed['case_count'] == 41 and observed['phase_count'] == 123
    assert set(observed['all_source_symbol_delta']['files']) == {
        'src/anyfesolver/e4_pl_element.py',
        'tests/test_e4_pl_s3_mixed_mesh_manifest.py',
        'tests/test_general_contact_c5_elastic_increment.py',
        'tests/test_general_contact_c5_numpy_coordinate_admission.py',
    }
    # The later reader-only test edit is independently pinned by its whole-module carrier.
    for relative in (
        'tests/test_general_contact_c5_elastic_increment.py',
        'tests/test_general_contact_c5_numpy_coordinate_admission.py',
    ):
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == observed[
            'all_source_symbol_delta'
        ]['files'][relative]['after_source_sha256']
    return record


def test_qualification_contract_and_replay_hotfix_bind_q4_mechanics_symbols() -> None:
    contract = _strict_json(QUALIFICATION_CONTRACT.read_bytes())
    candidate = contract["candidate"]
    assert candidate["qualified_q4_activation_authority"] == {
        "blob": "bc0de942573e12089ee5a3619249eeb9b4412cb3",
        "disposition": "HISTORICAL_ACTIVATION_BASELINE_NOT_LIVE_MECHANICS_GATE",
        "sha256": (
            "EE49BAE1C9439C41EC2D61798A8A8B88CBA9081DCAD2DFDC857FE313C6C0D4D1"
        ),
    }
    authority = candidate["qualified_q4_mechanics_authority"]
    assert set(authority) == {
        "aggregate_sha256",
        "canonicalization",
        "edges",
        "path",
        "root_symbol_count",
        "roots",
        "schema",
        "symbol_count",
        "symbols",
    }
    assert authority["path"] == QUALIFIED_Q4_MECHANICS.relative_to(ROOT).as_posix()
    assert authority["schema"] == Q4_MECHANICS_AUTHORITY_SCHEMA
    assert authority["canonicalization"] == Q4_MECHANICS_CANONICALIZATION
    roots = authority["roots"]
    assert authority["root_symbol_count"] == len(roots) == 47
    assert roots == sorted(set(roots))
    expected = authority["symbols"]
    expected_edges = authority["edges"]
    assert authority["symbol_count"] == len(expected) == len(expected_edges) == 129
    assert list(expected) == sorted(expected)
    assert list(expected_edges) == sorted(expected_edges)
    assert all(
        dependencies == sorted(set(dependencies))
        and set(dependencies) <= set(expected)
        for dependencies in expected_edges.values()
    )
    assert all(
        isinstance(name, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and set(digest) <= set("0123456789ABCDEF")
        for name, digest in expected.items()
    )
    actual = _q4_mechanics_authority_payload(
        QUALIFIED_Q4_MECHANICS.read_text(encoding="utf-8"),
        set(roots),
        schema=authority["schema"],
        path=authority["path"],
        required_symbols=set(expected),
    )
    hotfix = _strict_json(Q4_REPLAY_HOTFIX.read_bytes())
    correction = hotfix["source_correction"]
    successor = _strict_json(Q4_REPLAY_CANONICALIZATION.read_bytes())
    successor_correction = successor["correction"]
    changed = sorted(
        name
        for name, digest in expected.items()
        if actual["symbols"][name] != digest
    )
    added = sorted(set(actual["symbols"]) - set(expected))
    # The immutable historical chain retains its original correction scope.
    assert correction["updated_existing_symbols"] == successor_correction[
        "cumulative_updated_existing_symbols"
    ]
    assert set(correction["added_symbols"]) < set(
        successor_correction["cumulative_added_symbols"]
    )
    assert authority["aggregate_sha256"] == correction[
        "prior_mechanics_aggregate_sha256"
    ]
    assert correction["current_mechanics_aggregate_sha256"] == successor_correction[
        "prior_mechanics_aggregate_sha256"
    ]
    assert hashlib.sha256(Q4_REPLAY_HOTFIX.read_bytes()).hexdigest().upper() == successor[
        "prior_evidence"
    ]["sha256"]
    # Current source genuinely differs from that historical authority; no fallback.
    assert _q4_mechanics_aggregate(actual) != successor_correction[
        "current_mechanics_aggregate_sha256"
    ]
    binding = _require_prospective_q4_source_binding(
        Q4_SOURCE_BINDING.read_bytes(), QUALIFIED_Q4_MECHANICS.read_bytes(), actual, authority
    )
    assert binding["historical_difference"] == {
        "changed_symbols": changed,
        "added_symbols": added,
        "removed_symbols": sorted(set(expected) - set(actual["symbols"])),
    }


def test_prospective_q4_source_binding_rejects_unknown_record_and_outside_closure() -> None:
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    raw = QUALIFIED_Q4_MECHANICS.read_bytes()
    source = raw.decode("utf-8")
    actual = _q4_mechanics_authority_payload(
        source, set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    record = Q4_SOURCE_BINDING.read_bytes()
    with pytest.raises(AssertionError):
        _require_prospective_q4_source_binding(record + b" ", raw, actual, authority)
    for field in ("roots", "symbols"):
        refused = _strict_json(json.dumps(actual).encode("utf-8"))
        if field == "roots":
            refused[field] = refused[field][1:]
        else:
            refused[field].pop(next(iter(authority["symbols"])))
        with pytest.raises(AssertionError):
            _require_prospective_q4_source_binding(record, raw, refused, authority)
    outside = "QualifiedE4PLShellElement._compute_prepared_nonlinear_response"
    assert outside not in actual["symbols"]
    definition = "    def _compute_prepared_nonlinear_response("
    assert source.count(definition) == 1
    changed = source.replace(definition, definition.replace("response", "response_changed"), 1)
    unchanged_closure = _q4_mechanics_authority_payload(
        changed, set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    assert unchanged_closure == actual
    with pytest.raises(AssertionError):
        _require_prospective_q4_source_binding(
            record, changed.encode("utf-8"), unchanged_closure, authority
        )


def test_prospective_q4_source_binding_rejects_external_dependency_drift(monkeypatch) -> None:
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    raw = QUALIFIED_Q4_MECHANICS.read_bytes()
    actual = _q4_mechanics_authority_payload(
        raw.decode("utf-8"), set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    record = Q4_SOURCE_BINDING.read_bytes()
    dependency = ROOT / "src" / "anyfesolver" / "vectorized_nonlinear.py"
    original = Path.read_bytes
    def changed_bytes(path):
        value = original(path)
        return value + b"\n# injected source drift\n" if path == dependency else value
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    with pytest.raises(AssertionError):
        _require_prospective_q4_source_binding(record, raw, actual, authority)

def test_q4_mechanics_symbol_authority_rejects_missing_duplicate_and_change() -> None:
    contract = _strict_json(QUALIFICATION_CONTRACT.read_bytes())
    authority = contract["candidate"]["qualified_q4_mechanics_authority"]
    selected = set(authority["roots"])
    source = QUALIFIED_Q4_MECHANICS.read_text(encoding="utf-8")
    signature = "def _shape(r: float, s: float) -> np.ndarray:"
    assert source.count(signature) == 1

    missing = source.replace(signature, signature.replace("_shape", "_shape_removed"), 1)
    with pytest.raises(ValueError, match="missing selected Q4 mechanics symbols"):
        _q4_mechanics_symbol_hashes(missing, selected)

    duplicate = source + "\n\ndef _shape(r: float, s: float) -> np.ndarray:\n    return r + s\n"
    with pytest.raises(ValueError, match="duplicate selected Q4 mechanics symbols"):
        _q4_mechanics_symbol_hashes(duplicate, selected)

    changed = source.replace(
        signature,
        "def _shape(r: float, s: float, marker: int = 0) -> np.ndarray:",
        1,
    )
    baseline_hashes = _q4_mechanics_symbol_hashes(source, selected)
    changed_hashes = _q4_mechanics_symbol_hashes(changed, selected)
    assert {
        name
        for name in selected
        if changed_hashes[name] != baseline_hashes[name]
    } == {"_shape"}


def test_q4_mechanics_closure_binds_physical_director_dependency() -> None:
    contract = _strict_json(QUALIFICATION_CONTRACT.read_bytes())
    authority = contract["candidate"]["qualified_q4_mechanics_authority"]
    source = QUALIFIED_Q4_MECHANICS.read_text(encoding="utf-8")
    roots = set(authority["roots"])
    expected_symbols = set(authority["symbols"])
    sign = "        sign = 1 if alignment > 0.0 else -1"
    reversed_sign = "        sign = -1 if alignment > 0.0 else 1"
    assert source.count(sign) == 1

    mutated = source.replace(sign, reversed_sign, 1)
    baseline = _q4_mechanics_authority_payload(
        source,
        roots,
        schema=authority["schema"],
        path=authority["path"],
        required_symbols=expected_symbols,
    )
    actual = _q4_mechanics_authority_payload(
        mutated,
        roots,
        schema=authority["schema"],
        path=authority["path"],
        required_symbols=expected_symbols,
    )
    changed = {
        name for name, digest in baseline["symbols"].items()
        if actual["symbols"][name] != digest
    }
    assert changed == {
        "QualifiedE4PLShellElement._physical_director_context"
    }
    assert _q4_mechanics_aggregate(actual) != _q4_mechanics_aggregate(baseline)

    definition = "    def _physical_director_context("
    assert source.count(definition) == 1
    unresolved = source.replace(
        definition,
        "    def _physical_director_context_removed(",
        1,
    )
    with pytest.raises(
        ValueError,
        match="unresolved selected Q4 mechanics local dependencies",
    ):
        _q4_mechanics_authority_payload(
            unresolved,
            roots,
            schema=authority["schema"],
            path=authority["path"],
            required_symbols=expected_symbols,
        )


def test_qualification_contract_freezes_all_gate_families_without_activation() -> None:
    contract = _strict_json(QUALIFICATION_CONTRACT.read_bytes())
    gates = contract["acceptance_gates"]
    assert set(gates) == {
        "batch",
        "buckling",
        "convergence",
        "interface_resultants",
        "locking",
        "modal",
        "patch_and_equilibrium",
        "performance",
        "pl_participation",
        "symmetry_and_covariance_residual_maximum",
    }
    assert contract["execution_policy"] == {
        "automatic_retry": False,
        "canonical_cycles": 2,
        "deterministic_cycle_equality": "BYTE_IDENTICAL_CANONICAL_AGGREGATES",
        "external_diagnostics": True,
        "maximum_memory_gib_per_process": 24,
        "maximum_seconds_per_process": 600,
        "numerical_library_threads_per_process": 1,
        "partial_output_is_canonical": False,
        "worker_directories": "FRESH_AND_DISTINCT",
    }
    assert contract["result_policy"]["present_status"] == (
        "PREREGISTERED_NOT_EXECUTED"
    )
    assert contract["result_policy"]["research_control_can_classify"] is False
    assert contract["ecosystem"]["default_activation"].startswith("FORBIDDEN_")


def test_campaign_has_exact_order_counts_and_fractions() -> None:
    payload = _strict_json(MANIFEST.read_bytes())
    records = payload["records"]
    assert isinstance(records, list)
    assert len(records) == 252
    expected_keys: list[tuple[int, str, str, int]] = []
    for split_count in GEN.SPLIT_COUNTS:
        masks = ("none",) if split_count == 0 else GEN.MASKS
        for mask in masks:
            for diagonal in GEN.DIAGONALS:
                for level in GEN.LEVELS:
                    expected_keys.append((split_count, mask, diagonal, level))
    actual_keys = [
        (
            record["split_base_cell_count"],
            record["mask"],
            record["diagonal"],
            record["level"],
        )
        for record in records
    ]
    assert actual_keys == expected_keys
    assert len(actual_keys) == len(set(actual_keys))

    for record in records:
        level = record["level"]
        split_count = record["split_base_cell_count"]
        factor = level // GEN.BASE_GRID
        split_refined = split_count * factor * factor
        assert record["node_count"] == (level + 1) ** 2
        assert record["split_refined_cell_count"] == split_refined
        assert record["q4_element_count"] == level**2 - split_refined
        assert record["s3_element_count"] == 2 * split_refined
        assert record["element_count"] == level**2 + split_refined
        assert record["s3_area_fraction_percent"] == split_count // 4

    research = payload["research_control"]
    assert research["classification"] == "RESEARCH_CONTROL_NOT_A_PRODUCTION_GATE"
    assert research["split_base_cell_count"] == GEN.BASE_GRID**2
    assert research["s3_area_fraction_percent"] == 100
    control_records = research["records"]
    assert [
        (record["diagonal"], record["level"]) for record in control_records
    ] == [
        (diagonal, level)
        for diagonal in GEN.DIAGONALS
        for level in GEN.LEVELS
    ]
    assert len(control_records) == 12
    for record in control_records:
        level = record["level"]
        assert record["mask"] == "all_cells"
        assert record["node_count"] == (level + 1) ** 2
        assert record["split_refined_cell_count"] == level**2
        assert record["q4_element_count"] == 0
        assert record["s3_element_count"] == 2 * level**2
        assert record["element_count"] == 2 * level**2
        assert record["s3_area_fraction_percent"] == 100


def test_masks_are_exact_nested_permutations_and_refine_one_to_four() -> None:
    all_cells = {(i, j) for j in range(GEN.BASE_GRID) for i in range(GEN.BASE_GRID)}
    for mask in GEN.MASKS:
        ordered = GEN.ordered_mask_cells(mask)
        assert len(ordered) == GEN.BASE_GRID**2
        assert set(ordered) == all_cells
        previous: set[tuple[int, int]] = set()
        for split_count in GEN.SPLIT_COUNTS[1:]:
            selected = set(GEN.selected_base_cells(mask, split_count))
            assert len(selected) == split_count
            assert previous < selected
            previous = selected
            for level in GEN.LEVELS:
                factor = level // GEN.BASE_GRID
                expanded = GEN.expanded_split_cells(tuple(selected), level)
                assert len(expanded) == split_count * factor * factor
                for i, j in selected:
                    assert {
                        (factor * i + di, factor * j + dj)
                        for dj in range(factor)
                        for di in range(factor)
                    } <= expanded


def test_all_connectivity_hashes_are_independently_reproducible() -> None:
    payload = _strict_json(MANIFEST.read_bytes())
    all_records = [
        *payload["records"],
        *payload["research_control"]["records"],
    ]
    for record in all_records:
        mask = record["mask"]
        split_count = record["split_base_cell_count"]
        level = record["level"]
        if mask == "none":
            base_cells = ()
        elif mask == "all_cells":
            base_cells = GEN._all_cells()
        else:
            base_cells = GEN.selected_base_cells(mask, split_count)
        split_cells = set(GEN.expanded_split_cells(base_cells, level))
        assert record["selected_base_cells_sha256"] == hashlib.sha256(
            _canonical_bytes(base_cells)
        ).hexdigest().upper()
        assert record["connectivity_sha256"] == _independent_connectivity_sha256(
            level,
            split_cells,
            record["diagonal"],
        )


@pytest.mark.parametrize("diagonal", GEN.DIAGONALS)
def test_q4_and_s3_connectivities_are_counterclockwise(diagonal: str) -> None:
    level = GEN.BASE_GRID
    for split in (False, True):
        for i, j in ((0, 0), (7, 11), (19, 19)):
            elements = GEN._cell_connectivity(
                i,
                j,
                level,
                split=split,
                diagonal=diagonal,
            )
            assert len(elements) == (2 if split else 1)
            for kind, node_ids in elements:
                assert kind == ("S3" if split else "Q4")
                assert _signed_area_twice(node_ids, level) > 0


def test_cli_check_accepts_canonical_and_rejects_mutation(tmp_path: Path) -> None:
    accepted = subprocess.run(
        [sys.executable, str(GENERATOR), "--check", str(MANIFEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr

    stale = tmp_path / "stale.json"
    payload = _strict_json(MANIFEST.read_bytes())
    payload["base_grid"] = 21
    stale.write_bytes(_canonical_bytes(payload))
    rejected = subprocess.run(
        [sys.executable, str(GENERATOR), "--check", str(stale)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "not canonical or is stale" in rejected.stderr
