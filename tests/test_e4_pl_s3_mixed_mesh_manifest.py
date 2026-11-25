from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import os
import json
from pathlib import Path
import re
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


# Two authentic states of the October 19 record.  The restored original is the
# only bytes the original reader pin 64B07F... describes; the live file at the
# original pathname carries six later in-place rebinds of external sha256 leaves
# (exact Git genealogy cited in the successor record).  Neither is rewritten.
Q4_SOURCE_BINDING_AS_REVIEWED = (
    ROOT / 'docs' / 'reference_cases' / 'e4_pl_q4_source_binding_20261019_as_reviewed.json'
)
Q4_SOURCE_BINDING_AS_REVIEWED_SHA256 = '64B07F1A535E6D6EFCEAE0B2EC01BA539DF69BE7956C03DBDA65C2ACF7C7E74F'
Q4_SOURCE_BINDING_AS_REVIEWED_BLOB = '99b59279e9702b509a84f1932874e2b24680adcc'
Q4_SOURCE_BINDING = ROOT / 'docs' / 'reference_cases' / 'e4_pl_q4_source_binding_20261019.json'
Q4_SOURCE_BINDING_SHA256 = '06679901687621BD6696DB9C8777A80131EE08631BFCCA706FD03FA15042FB38'
Q4_SOURCE_BINDING_BLOB = '3052505a185198142ffeee60fbee8ad46b3d968c'
EXTERNAL_LEAVES = {
    f'external_arithmetic_dependencies/{alias}/sha256'
    for alias in ('_elastic_increment_module', '_local_contractions', '_elements_module')
}

# Source-binding successor (2026-11-11).  SOURCE BINDING ONLY: it grants no CI,
# scientific or physical acceptance and replaces no immutable record.  The pin
# stays None until (1) the collector scripts/build_q4_source_binding_20261111.py
# has been run with the historical commits present and (2) an independent
# reviewer outside the generating invocation has reviewed the record, this
# reader and the collector.  Do not fill it from the generating invocation.
Q4_SOURCE_BINDING_SUCCESSOR = (
    ROOT / 'docs' / 'reference_cases' / 'e4_pl_source_binding_20261111.json'
)
Q4_SOURCE_BINDING_SUCCESSOR_SHA256: str | None = '2716B80F2792574F4720CC282C4368E3E936E5C1353FFF8B63439E72B30F42B1'
# Source-binding delta (2026-11-24, warped-Q4 drilling floor).  SOURCE BINDING
# ONLY.  The 2026-11-11 record keeps binding the bytes it was generated for:
# its reader runs on predecessor bytes rebuilt from live bytes by this record's
# reverse hunks, authenticated by the 2026-11-11 pinned identities.  This record
# binds the live bytes and the delta.  The pin stays None until (1) the
# collector scripts/build_q4_source_binding_20261124.py has been run with the
# base revision present and (2) an independent reviewer outside the generating
# invocation has reviewed the record, this reader and the collector.  Do not
# fill it from the generating invocation.
Q4_SOURCE_BINDING_DELTA = (
    ROOT / 'docs' / 'reference_cases' / 'e4_pl_source_binding_20261124.json'
)
# Pinned 2026-11-24 after the collector --check ran with the historical objects
# present and an independent Codex (gpt-6.2-sol, high) review of the record,
# reader and collector returned PIN APPROVED for these bytes.
Q4_SOURCE_BINDING_DELTA_SHA256: str | None = (
    '340CF7EB6B15B01799FA5838CC558BE74C1D6033C267120C5A142F6C257E50A7'
)
# Source-binding delta chained from the 2026-11-24 delta (2026-11-25, S2
# bit-identical beam frame-math trim in elements.py).  SOURCE BINDING ONLY.
# Both predecessor readers keep running on predecessor bytes rebuilt from the
# live bytes: this record's reverse hunks rebuild the 2026-11-24-bound bytes
# (authenticated by the 2026-11-24 pinned record), whose own hunks rebuild the
# 2026-11-11-bound bytes.  The pin stays None until (1) the collector
# scripts/build_q4_source_binding_20261125.py has been run with the base
# revision present and (2) an independent reviewer outside the generating
# invocation has reviewed the record, this reader and the collector.  Do not
# fill it from the generating invocation.
Q4_SOURCE_BINDING_DELTA2 = (
    ROOT / 'docs' / 'reference_cases' / 'e4_pl_source_binding_20261125.json'
)
Q4_SOURCE_BINDING_DELTA2_SHA256: str | None = None
SIX_CARRIERS = (
    'e4_pl_element.py',
    'e4_pl_s3_element.py',
    '_q4_assumed_strain_plasticity.py',
    'vectorized_nonlinear.py',
    'elements.py',
    '_compensated_contractions.py',
)
# Explicit recorded-to-live map.  Recorded bytes/paths are never rewritten; the
# map is applied only when resolving a recorded path to a live file.
RECORDED_TO_LIVE_PATH = {
    f'src/anysolver/{name}': f'src/anyfesolver/{name}' for name in SIX_CARRIERS
}
TEXT_LIVE_TO_RECORDED = (('ANYfeSolver', 'ANYsolver'), ('anyfesolver', 'anysolver'))
LIVE_ENTRY = 'src/anyfesolver/e4_pl_element.py'
# Independently measured Git ids, identical at 5c34dc2e4, a0781c0bc and e0c5f32ab.
UPSTREAM_GIT_BLOBS = {
    'e4_pl_element.py': '47efceacb07a9be0ab92542164e5dd3ca1ea0c63',
    'e4_pl_s3_element.py': '8047c66f677be5214fb7457a0312492481fb7e93',
    '_q4_assumed_strain_plasticity.py': 'b78cf67beb8f11d1e0a239c05ee098417ecdfad7',
    'vectorized_nonlinear.py': 'b2ed521ff2b16710d50b7ee7b407447eb455ec4e',
    'elements.py': '0654d419641bf591ee21ddbe9ddf0c98bc8890c8',
    '_compensated_contractions.py': 'e82e10b1f74aeee44ccc9e3009444b668f1ed53d',
}
V6_TERMINAL_BLOBS = {
    'e4_pl_element.py': 'bd6c0dc2cb3661eb73942fc0fea56e527aa8d0b4',
    'e4_pl_s3_element.py': 'adc6562cbd3fb768d7f0d0ab69011f21ff7c9392',
}
OCT19_CLOSURE = (159, '996C94581594E958545D5D64F25D1864942941A5B766BE88A79A4991BD4DE95C')
UPSTREAM_CLOSURE = (161, 'CF0026B4621EA9180B0DFCE1A483AFB8A762A218CB5CC3F538517C1A8D0843AA')
# Auto-text sources (index LF, Windows working copy CRLF, no eol attribute).
# Identity is the canonical LF form; exactly two physical forms are permitted
# (uniform LF or uniform CRLF).  -text carriers and everything else stay exact.
PORTABLE_AUTO_TEXT = frozenset({
    'e4_pl_s3_element.py', '_q4_assumed_strain_plasticity.py',
    '_qualified_authority_epoch.py', 'shell_sections.py', 'e4_pl_s3_state.py',
    'plasticity.py',
})
LINEAGE_JSON_SHA256 = '1cef66aedcd866ed1feaa1ee28497f30a96966a78d01e1cb472ea5d58dbbd7ac'
LINEAGE_README_SHA256 = '0605298d0ec64b1f005e0139513da7a588cab256e58b36ab3140d04295bef523'
# Independently confirmed Git-free digest of the whole nine-node canonical payload
# (all nine records and ten parents, with blobs, digests, lengths and changes).
LINEAGE_CHAIN_SHA256 = 'C75FA5607256A911999174EE7475B07FDE691ECD5A935F725E2B9F93F06E3BA5'
# Independently audited topology of the nine nodes: (commit, record blob, parents).
LINEAGE_TOPOLOGY = (
    ('af0344922b565fe8424e2c98dd28280d01339eda', 'fc61d1cb69156d950aa35c6318153dfa06ddb006',
     ('4074eb8b163c1d04cd5e1faed742a237680c01fd',)),
    ('36a5d5d8c1ae3816fe98d9049e79dec7b39787e9', '99b59279e9702b509a84f1932874e2b24680adcc',
     ('af0344922b565fe8424e2c98dd28280d01339eda',)),
    ('f75aed6a4188cc0f86cefcaf6e15af806e3ea108', '7024980f2f8e2b917aaa0626a375da9a2fd43e0b',
     ('57dd01f047a4ea4e26d518aeddbf03997c3bd641',)),
    ('842c38926890521efbdbe33e9c5c72222fa37596', '7b46c036dfdece1f96e7f81a273143cbddd963a9',
     ('f6a9de9f815c684e883e73212571764ae6b07665',)),
    ('a42dc0a5c94c40f59ba781306aee833aa9d4a3ff', '249fb5e38917dd1ba77c6d18a34cba7868d82cde',
     ('842c38926890521efbdbe33e9c5c72222fa37596',)),
    ('504dfb6578c2aacef31d66c6e48fb2f0d3a42af7', '0ffb6689d5a4ddfef15ff7de2ea872a3eec1e170',
     ('a42dc0a5c94c40f59ba781306aee833aa9d4a3ff',)),
    ('b7863d9d864a6df2f8a15ae9ca91791b390627c0', '6b8b616559f53f1af2339a5f981e14f086eec3fd',
     ('504dfb6578c2aacef31d66c6e48fb2f0d3a42af7',)),
    ('cb33056a44c22486843473964a51045cabf1ab68', '3052505a185198142ffeee60fbee8ad46b3d968c',
     ('b7863d9d864a6df2f8a15ae9ca91791b390627c0',)),
    ('5842caf6c9384009f1d9e3b1308ffd77e61a626a', '3052505a185198142ffeee60fbee8ad46b3d968c',
     ('953d189a379ddc56cb4fbddc33d7ae6945e50eb8', '0758b87c98a95c27fdf0e912300865feb8f6ec7e')),
)
SUCCESSOR_FLAG_KEYS = {
    'C5_exception_transfer', 'CI_credit', 'historical_observation_rerun',
    'historical_records_modified', 'new_physical_acceptance',
    'new_scientific_acceptance', 'numerical_tolerances_changed',
    'physical_acceptance_transfer', 'production_mechanics_changed_by_this_record',
}


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
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
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


def _require_historical_q4_source_binding_record(
    record_raw: bytes,
    authority: dict[str, object],
) -> dict[str, object]:
    """Read the immutable October 19 record on its own recorded terms.

    The record binds one historical byte state; it is deliberately NOT compared
    with live bytes here (``historical_current_source_matches`` is False).  Live
    bytes are bound only by the 2026-11-11 successor reader below.
    """

    assert hashlib.sha256(record_raw).hexdigest().upper() == Q4_SOURCE_BINDING_AS_REVIEWED_SHA256
    assert _blob_id(record_raw) == Q4_SOURCE_BINDING_AS_REVIEWED_BLOB
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
    # The recorded fe_core lifecycle bytes are a whole-file carrier of the
    # historical tree; they are a recorded historical gap, not a live check.
    lifecycle = integration['lifecycle_delta']
    assert lifecycle['path'] == 'src/anysolver/fe_core.py' and len(lifecycle['sha256']) == 64
    # Recorded spelling is src/anysolver/...; resolve through the explicit map.
    assert _live_path(record['source']['path']) == QUALIFIED_Q4_MECHANICS.relative_to(ROOT).as_posix()
    assert record['mechanics']['roots'] == authority['roots']
    assert len(record['mechanics']['roots']) == 47
    assert set(authority['symbols']) <= set(record['mechanics']['symbols'])
    assert len(authority['symbols']) == 129
    assert _q4_mechanics_aggregate(record['mechanics']) == record['mechanics_aggregate_sha256']
    assert record['historical_current_source_matches'] is False
    for relative, digest in record['historical_authorities'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper() == digest
    assert set(record['historical_authorities']) == {
        QUALIFICATION_CONTRACT.relative_to(ROOT).as_posix(),
        Q4_REPLAY_HOTFIX.relative_to(ROOT).as_posix(),
        Q4_REPLAY_CANONICALIZATION.relative_to(ROOT).as_posix(),
    }
    # The three recorded aliases and their recorded (historical) paths are kept
    # exactly; their recorded digests describe the October 19 bytes and are not
    # compared with live bytes.  Live bytes are checked by the successor reader.
    expected_external = {
        '_elastic_increment_module': 'src/anyfesolver/vectorized_nonlinear.py',
        '_local_contractions': 'src/anyfesolver/_compensated_contractions.py',
        '_elements_module': 'src/anyfesolver/elements.py',
    }
    assert set(record['external_arithmetic_dependencies']) == set(expected_external)
    for alias, relative in expected_external.items():
        row = record['external_arithmetic_dependencies'][alias]
        assert _live_path(row['path']) == relative and row['changed_this_slice'] is False
        assert len(row['sha256']) == 64
    observed = record['observed_focused_pre_reader_integration']
    assert observed['status'] == 'PASS'
    assert observed['case_count'] == 41 and observed['phase_count'] == 123
    assert {
        _live_path(path) if path.startswith('src/') else path
        for path in observed['all_source_symbol_delta']['files']
    } == {
        'src/anyfesolver/e4_pl_element.py',
        'tests/test_e4_pl_s3_mixed_mesh_manifest.py',
        'tests/test_general_contact_c5_elastic_increment.py',
        'tests/test_general_contact_c5_numpy_coordinate_admission.py',
    }
    # The two C5 test carriers' recorded whole-module digests describe the
    # historical tree; they are historical gaps (see the successor record).
    for relative in (
        'tests/test_general_contact_c5_elastic_increment.py',
        'tests/test_general_contact_c5_numpy_coordinate_admission.py',
    ):
        assert len(observed['all_source_symbol_delta']['files'][relative]['after_source_sha256']) == 64
    return record


def _leaves(value: object, path: tuple[str, ...] = ()) -> dict[str, object]:
    if isinstance(value, dict):
        return {k: v for key in value for k, v in _leaves(value[key], path + (key,)).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in _leaves(item, path + (str(i),)).items()}
    return {'/'.join(path): value}


# Root whose files the import-form check inspects (tests redirect it to a copy).
IMPORT_FORM_ROOT = ROOT


def _import_form_entries(directory: Path) -> list[tuple[str, bool]]:
    """(name, is_directory) of every entry beside a bound file."""

    return [(entry.name, entry.is_dir()) for entry in directory.iterdir()]


def _require_sole_import_form(relative: str) -> None:
    """Fail closed unless the bound ``.py`` file is the only importable form of
    its module: no package or namespace directory of that name (with or without
    ``__init__.py``) and no other import suffix (bytecode, extension module,
    or another spelling of the ``.py`` name) beside it, compared without regard
    to case but never against the exact bound file name; an ordinary
    extensionless regular file of that name does not shadow; for package modules, the path finder must resolve to
    exactly the bound file.

    Scope (owner decision 2026-11-25, whole chain 20261111 -> 20261124 -> 20261125):
    the source binding covers the bytes of the tracked source files in this
    checkout plus this same-directory import-form uniqueness.  Runtime import
    identity is out of scope: ``sys.path`` order (another checkout's ``src`` or a
    checkout-root package ahead of this ``src``), bytecode caches (including a
    timestamp cache that survives an edit-and-restore within the same second) and
    installed copies are not checked here.
    """

    if not relative.endswith(".py"):
        return
    path = IMPORT_FORM_ROOT / relative
    assert path.is_file(), relative
    directory, stem = path.parent, path.name[:-3].casefold()
    suffixes = {suffix.casefold() for suffix in importlib.machinery.all_suffixes()}
    for entry_name, entry_is_dir in _import_form_entries(directory):
        if entry_name == path.name:
            continue  # exactly the bound file; every OTHER entry is judged below
        name = entry_name.casefold()
        if name == stem:
            # A directory (package or namespace, with or without __init__.py)
            # shadows; an ordinary extensionless regular file does not.  Another
            # spelling of the bound file's own name (case-fold) is another form.
            assert not entry_is_dir, f"{relative}: {entry_name} shadows the bound module"
        assert name != path.name.casefold(), (
            f"{relative}: {entry_name} is another spelling of the bound module"
        )
        # Only a regular file with an importable suffix is another module form; a
        # directory merely named like one (e.g. ``elements.pyc/``) is not.
        assert entry_is_dir or not any(
            name.endswith(suffix) and name[: -len(suffix)] == stem for suffix in suffixes
        ), f"{relative}: {entry_name} is another importable form of the bound module"
    if relative.startswith("src/"):
        fullname = ".".join(Path(relative[len("src/"):-len(".py")]).parts)
        importlib.invalidate_caches()
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(directory)])
        assert spec is not None and spec.origin is not None, relative
        assert os.path.normcase(os.path.realpath(spec.origin)) == os.path.normcase(
            os.path.realpath(path)
        ), f"{relative}: Python imports {spec.origin}"


def _require_physical(path: str, raw: bytes, row: dict[str, object]) -> None:
    """Canonical LF identity everywhere; raw exact unless explicitly auto-text."""

    _require_sole_import_form(path)
    portable = path.rsplit('/', 1)[1] in PORTABLE_AUTO_TEXT
    assert row['portable_text'] is portable
    canonical = _lf(raw)
    assert row['live_lf'] == _identity(canonical)
    if portable:
        assert 'live_raw' not in row
        assert b'\r' not in canonical  # no lone or doubled CR survives normalization
        crlf = canonical.replace(b'\n', b'\r\n')
        assert raw == canonical or raw == crlf  # uniform LF or uniform CRLF only
        assert row['permitted_raw'] == {
            'crlf': _identity(crlf), 'lf': _identity(canonical),
        }
    else:
        assert 'permitted_raw' not in row
        assert row['live_raw'] == _identity(raw)


def _authenticated_lineage_transitions(lineage: dict[str, object]) -> list[dict[str, object]]:
    """Verify all nine nodes statically (no Git, no external files)."""

    nodes = lineage['nodes']
    assert lineage['lineage_json_sha256'].lower() == LINEAGE_JSON_SHA256
    assert lineage['readme_sha256'].lower() == LINEAGE_README_SHA256
    assert lineage['anchor'] == 'e0c5f32ab40e9548ba0ea8981cf5e288ed375a1c'
    assert len(nodes) == 9
    assert all(set(n) == {'blob', 'commit', 'length', 'parents', 'sha256'} for n in nodes)
    assert [
        (n['commit'], n['blob'], tuple(p['commit'] for p in n['parents'])) for n in nodes
    ] == list(LINEAGE_TOPOLOGY)
    # Bind the exact whole payload: restamping invented sha/length/blobs/changes
    # cannot pass even when commit topology matches.
    assert _digest_upper(nodes) == LINEAGE_CHAIN_SHA256
    assert lineage['chain_sha256'] == LINEAGE_CHAIN_SHA256
    assert (nodes[1]['blob'], nodes[1]['sha256']) == (
        Q4_SOURCE_BINDING_AS_REVIEWED_BLOB, Q4_SOURCE_BINDING_AS_REVIEWED_SHA256,
    )
    assert all((n['blob'], n['sha256']) == (Q4_SOURCE_BINDING_BLOB, Q4_SOURCE_BINDING_SHA256) for n in nodes[7:])
    steps = []
    for node in nodes[2:8]:
        assert len(node['parents']) == 1 and 'changes' in node['parents'][0]
        parent = node['parents'][0]
        steps.append({
            'blob': node['blob'], 'changes': parent['changes'],
            'parent_blob': parent['blob'], 'sha256': node['sha256'],
        })
    assert nodes[8]['parents'][0]['blob'] == Q4_SOURCE_BINDING_BLOB
    assert nodes[8]['parents'][0]['changes'] == {}
    return steps


def _require_current_historical_record(
    current_raw: bytes, reviewed_raw: bytes, successor: dict[str, object], read
) -> None:
    """Pin the live historical file independently of its predecessor.

    Cites the six measured in-place edits (external sha256 leaves only) and the
    five whole-file comparisons that no longer describe live bytes.  Endorses no
    scientific transfer and does not equate the file with its predecessor or with
    current execution.
    """

    assert hashlib.sha256(current_raw).hexdigest().upper() == Q4_SOURCE_BINDING_SHA256
    assert _blob_id(current_raw) == Q4_SOURCE_BINDING_BLOB
    assert current_raw != reviewed_raw
    reviewed, current = _leaves(_strict_json(reviewed_raw)), _leaves(_strict_json(current_raw))
    assert reviewed.keys() == current.keys()
    assert {k for k in current if current[k] != reviewed[k]} == EXTERNAL_LEAVES
    lineage = successor['historical_record_lineage']
    assert lineage['scientific_transfer_endorsed'] is False
    state, previous = dict(reviewed), Q4_SOURCE_BINDING_AS_REVIEWED_BLOB
    transitions = _authenticated_lineage_transitions(lineage)
    assert len(transitions) == 6
    states = {Q4_SOURCE_BINDING_AS_REVIEWED_BLOB: dict(reviewed)}  # exact state per blob
    for step in transitions:
        assert step['parent_blob'] == previous and set(step['changes']) <= EXTERNAL_LEAVES
        for leaf, change in step['changes'].items():
            assert state[leaf] == change['before']
            state[leaf] = change['after']
        previous = step['blob']
        states[previous] = dict(state)
    assert state == current and previous == Q4_SOURCE_BINDING_BLOB
    # Merge: each parent's changes must equal its own replayed blob state -> current.
    # (0758b87c is the object after the FIRST edit, not the original pin.)
    for parent in lineage['nodes'][8]['parents']:
        assert parent['blob'] in states
        assert parent['changes'] == {
            leaf: {'before': states[parent['blob']][leaf], 'after': current[leaf]}
            for leaf in sorted(current)
            if states[parent['blob']][leaf] != current[leaf]
        }
        assert set(parent['changes']) <= EXTERNAL_LEAVES
    assert transitions[-1]['sha256'] == Q4_SOURCE_BINDING_SHA256
    assert lineage['pin_node']['sha256'] == Q4_SOURCE_BINDING_AS_REVIEWED_SHA256
    assert lineage['pin_node']['blob'] == Q4_SOURCE_BINDING_AS_REVIEWED_BLOB

    gaps = successor['historical_gaps_of_current_historical_record']
    integration = _strict_json(current_raw)['source_binding_integration_review']
    assert all(integration[k] is False for k in ('physical_acceptance_transfer', 'C5_exception_transfer', 'CI_credit'))
    assert len(gaps) == 5 and len({g['check'] for g in gaps}) == 5
    for gap in gaps:
        assert gap['historical_gap'] is True
        assert gap['live_path'] == gap['recorded_path'].replace('src/anysolver/', 'src/anyfesolver/', 1)
        gap_raw = read(gap['live_path'])
        _require_sole_import_form(gap['live_path'])
        live = hashlib.sha256(_lf(gap_raw)).hexdigest().upper()
        assert gap['live_lf_sha256'] == live
        # Protected (-text) carriers: raw current bytes are pinned as well as LF.
        assert gap['live_path'].rsplit('/', 1)[1] not in PORTABLE_AUTO_TEXT
        assert gap['live_raw'] == _identity(gap_raw)
        assert gap['live_lf'] == _identity(_lf(gap_raw))
        # Each exact historical leaf links to this observation.
        assert str(current[gap['historical_leaf']]).upper() == gap['recorded_sha256']
        assert gap['live_lf_matches_recorded'] is (live == gap['recorded_sha256']) is False
        assert gap['recorded_sha256'] in {str(v).upper() for v in current.values()}


def _live_path(recorded: str) -> str:
    assert recorded in RECORDED_TO_LIVE_PATH, f"unmapped recorded path {recorded!r}"
    return RECORDED_TO_LIVE_PATH[recorded]


def _read_live(relative: str) -> bytes:
    return (ROOT / relative).read_bytes()


_DELTA_LINE = re.compile(rb"[^\n]*\n|[^\n]+\Z")


def _rebuild_predecessor(live: bytes, hunks: list[dict[str, object]]) -> bytes:
    """Apply recorded reverse line hunks (bottom-up, non-overlapping)."""

    lines = _DELTA_LINE.findall(live)
    assert b"".join(lines) == live
    limit = len(lines)
    for hunk in sorted(hunks, key=lambda h: h["live_line_start"], reverse=True):
        start, end = hunk["live_line_start"], hunk["live_line_end"]
        assert type(start) is int and type(end) is int and 0 <= start <= end <= limit
        assert hashlib.sha256(b"".join(lines[start:end])).hexdigest().upper() == (
            hunk["live_lines_sha256"]
        )
        lines[start:end] = [hunk["predecessor_text"].encode("utf-8")]
        limit = start
    return b"".join(lines)


def _delta_record() -> dict[str, object]:
    raw = Q4_SOURCE_BINDING_DELTA.read_bytes()
    if Q4_SOURCE_BINDING_DELTA_SHA256 is not None:
        assert hashlib.sha256(raw).hexdigest().upper() == Q4_SOURCE_BINDING_DELTA_SHA256
    record = _strict_json(raw)
    predecessor = Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()
    assert record["predecessor"] == {
        "path": Q4_SOURCE_BINDING_SUCCESSOR.relative_to(ROOT).as_posix(),
        **_identity(predecessor),
    }
    return record


def _delta2_record() -> dict[str, object]:
    raw = Q4_SOURCE_BINDING_DELTA2.read_bytes()
    if Q4_SOURCE_BINDING_DELTA2_SHA256 is not None:
        assert hashlib.sha256(raw).hexdigest().upper() == Q4_SOURCE_BINDING_DELTA2_SHA256
    record = _strict_json(raw)
    predecessor = Q4_SOURCE_BINDING_DELTA.read_bytes()
    assert Q4_SOURCE_BINDING_DELTA_SHA256 is not None
    assert hashlib.sha256(predecessor).hexdigest().upper() == Q4_SOURCE_BINDING_DELTA_SHA256
    assert record["predecessor"] == {
        "path": Q4_SOURCE_BINDING_DELTA.relative_to(ROOT).as_posix(),
        **_identity(predecessor),
    }
    return record


def _read_delta_bound(relative: str, read=None, record=None) -> bytes:
    """Bytes the 2026-11-24 delta binds: live bytes, or for elements.py (the
    only file the 2026-11-25 delta may change), the 2026-11-24-bound bytes
    rebuilt from the live bytes.

    ``record`` is the 2026-11-25 record the rebuild uses; a checker passes the
    record it has authenticated, never a second copy.  The rebuild is
    authenticated by the record's predecessor identity, which must equal the
    live identity the pinned 2026-11-24 record binds; the 2026-11-24 reader
    then checks the bytes on its own terms.  A record that changes any other
    file fails closed.
    """

    files = (_delta2_record() if record is None else record)["files"]
    assert set(files) == {"src/anyfesolver/elements.py"}
    raw = (_read_live if read is None else read)(relative)
    row = files.get(relative)
    if row is None:
        return raw
    rebuilt = _rebuild_predecessor(raw, row["reverse_hunks"])
    assert _identity(rebuilt) == row["predecessor_raw"]
    assert row["predecessor_raw"] == _delta_record()["files"][relative]["live_raw"]
    return rebuilt


def _read_bound(relative: str, read=None) -> bytes:
    """Bytes the 2026-11-11 record binds: the 2026-11-24-bound bytes (``read``,
    by default the live bytes rebuilt through the 2026-11-25 delta), or for a
    file changed by the 2026-11-24 delta, its predecessor bytes rebuilt from them.

    The rebuild is authenticated twice: by the delta's predecessor identity and,
    independently, by the 2026-11-11 reader's own pinned identities of the bytes
    it is given.  Any mutation of the live bytes breaks the rebuild.
    """

    raw = (_read_delta_bound if read is None else read)(relative)
    row = _delta_record()["files"].get(relative)
    if row is None:
        return raw
    rebuilt = _rebuild_predecessor(raw, row["reverse_hunks"])
    assert _identity(rebuilt) == row["predecessor_raw"]
    return rebuilt


def _read_with(replacements: dict[str, bytes]):
    def read(relative: str) -> bytes:
        return replacements[relative] if relative in replacements else _read_bound(relative)

    return read


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _identity(data: bytes) -> dict[str, object]:
    return {
        "bytes": len(data),
        "git_blob": _blob_id(data),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
    }


def _reverse_spelling(data: bytes) -> bytes:
    for live, recorded in TEXT_LIVE_TO_RECORDED:
        data = data.replace(live.encode("ascii"), recorded.encode("ascii"))
    return data


def _digest_upper(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()


def _module_symbol_aggregate(source: str) -> str:
    return _digest_upper({
        name: _digest_upper([_canonical_mechanics_ast(node) for node in nodes])
        for name, nodes in sorted(_mechanics_nodes(source).items())
    })


def _internal_imports(source: str, closure_symbols: list[str]) -> list[dict[str, object]]:
    """Independent scan of every sibling-package import of the entry module."""

    inventory = _mechanics_nodes(source)
    rows: list[dict[str, object]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom):
            continue
        absolute = bool(node.level == 0 and node.module) and node.module.split(".")[0] in (
            "anyfesolver", "anysolver",
        )
        if node.level == 0 and not absolute:
            continue
        parts = node.module.split(".")[1:] if absolute else (
            node.module.split(".") if node.module else []
        )
        for alias in node.names:
            bound = alias.asname or alias.name
            used: set[str] = set()
            loaded = False
            for symbol in closure_symbols:
                for item in ast.walk(inventory[symbol][0]):
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == bound:
                        loaded = True
                    if (
                        isinstance(item, ast.Attribute)
                        and isinstance(item.value, ast.Name)
                        and item.value.id == bound
                    ):
                        used.add(item.attr)
            kind = "name" if parts else "module"
            rows.append({
                "bound_name": bound,
                "file": ("/".join(parts) if parts else alias.name) + ".py"
                if node.level <= 1 else None,
                "imported": alias.name,
                "kind": kind,
                "relative_level": node.level,
                "used_attributes": sorted(used) if kind == "module" else [],
                "used_in_closure": loaded or bool(used),
            })
    return sorted(rows, key=lambda r: (r["bound_name"], str(r["file"]), r["imported"]))


def _require_pin() -> None:
    if Q4_SOURCE_BINDING_SUCCESSOR_SHA256 is None:
        pytest.fail(
            "source-binding successor is not adopted: run "
            "scripts/build_q4_source_binding_20261111.py --write, have the record, "
            "this reader and the collector independently reviewed outside the "
            "generating invocation, then pin Q4_SOURCE_BINDING_SUCCESSOR_SHA256",
            pytrace=False,
        )


def _require_successor_mechanics(record, actual, authority) -> None:
    mech = record["mechanics"]
    assert actual["roots"] == authority["roots"] == mech["roots"]
    assert len(actual["roots"]) == 47
    assert set(authority["symbols"]) <= set(actual["symbols"])
    assert len(authority["symbols"]) == 129
    assert actual["symbols"] == mech["symbols"]
    assert mech["symbol_count"] == len(actual["symbols"])
    assert actual["path"] == mech["logical_path"] == authority["path"]
    assert actual["schema"] == mech["schema"] == authority["schema"]
    assert _q4_mechanics_aggregate(actual) == mech["aggregate_sha256"]


def _require_successor_imports(record, source: str, closure_symbols, read) -> None:
    rows = record["sibling_imports_of_entry_module"]
    recomputed = _internal_imports(source, sorted(closure_symbols))
    live_keys = {
        "live_path", "recorded_path", "live_raw", "live_lf", "portable_text", "permitted_raw",
    }
    assert [
        {k: v for k, v in row.items() if k not in live_keys} for row in rows
    ] == recomputed
    for row in rows:
        assert row["file"] is not None and row["relative_level"] <= 1
        assert row["live_path"] == f"src/anyfesolver/{row['file']}"
        assert row["recorded_path"] == f"src/anysolver/{row['file']}"
        _require_physical(row["live_path"], read(row["live_path"]), row)
    # Preserve the October 19 external checks: each recorded alias still resolves
    # (through the explicit map) to exactly one live import of that file.
    oct19 = _strict_json(Q4_SOURCE_BINDING_AS_REVIEWED.read_bytes())
    for alias, old in oct19["external_arithmetic_dependencies"].items():
        matches = [row for row in rows if row["bound_name"] == alias]
        assert len(matches) == 1
        assert matches[0]["live_path"] == _live_path(old["path"])
        assert matches[0]["used_in_closure"] is True
    # The new assumed-strain runtime source is a bound dependency of the closure.
    assert any(
        row["file"] == "_q4_assumed_strain_plasticity.py" and row["used_in_closure"]
        for row in rows
    )


def _require_q4_source_binding_successor(
    record_raw: bytes,
    read,
    actual: dict[str, object],
    authority: dict[str, object],
) -> dict[str, object]:
    """Bind live (renamed) source to the immutable history; SOURCE BINDING ONLY."""

    assert Q4_SOURCE_BINDING_SUCCESSOR_SHA256 is not None
    assert hashlib.sha256(record_raw).hexdigest().upper() == Q4_SOURCE_BINDING_SUCCESSOR_SHA256
    record = _strict_json(record_raw)
    assert record_raw == _canonical_bytes(record)
    assert record["schema"] == "anyfesolver.q4-source-binding-successor-v2"
    assert record["qualification_boundary"] == "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"
    adoption = record["adoption"]
    assert adoption["reviewer_approval_claimed"] is False
    assert adoption["owner_adoption_recorded"] is False
    assert adoption["status"] == "UNREVIEWED_CANDIDATE_RECORD"
    assert adoption["independent_review_required_outside_generating_invocation"] is True
    flags = record["boundary_flags"]
    assert set(flags) == SUCCESSOR_FLAG_KEYS
    assert all(value is False for value in flags.values())

    # Immutable predecessor, read on its own recorded terms.
    reviewed_raw = Q4_SOURCE_BINDING_AS_REVIEWED.read_bytes()
    predecessor = record["predecessor"]
    assert predecessor["oct19_as_reviewed"] == {
        "bytes": len(reviewed_raw),
        "git_blob": Q4_SOURCE_BINDING_AS_REVIEWED_BLOB,
        "path": Q4_SOURCE_BINDING_AS_REVIEWED.relative_to(ROOT).as_posix(),
        "sha256": Q4_SOURCE_BINDING_AS_REVIEWED_SHA256,
    }
    current_raw = Q4_SOURCE_BINDING.read_bytes()
    assert predecessor["oct19_current_historical"] == {
        "bytes": len(current_raw),
        "git_blob": Q4_SOURCE_BINDING_BLOB,
        "path": Q4_SOURCE_BINDING.relative_to(ROOT).as_posix(),
        "sha256": Q4_SOURCE_BINDING_SHA256,
    }
    assert predecessor["current_historical_equals_as_reviewed"] is False
    assert predecessor["current_historical_describes_current_execution"] is False
    historical = _require_historical_q4_source_binding_record(reviewed_raw, authority)
    _require_current_historical_record(current_raw, reviewed_raw, record, read)
    assert record["revisions"]["c31a223f"] == "c31a223fe63f886a1a9b0b47c926bbe748ad4554"
    assert record["predecessor"]["v6_runtime_guard_successor_commit"] == record["revisions"]["c31a223f"]
    cited = record["historical_evidence_cited"]
    assert cited["oct19_nonacceptance_flags_retained"] == {
        key: False for key in ("physical_acceptance_transfer", "C5_exception_transfer", "CI_credit")
    }
    assert all(
        historical["source_binding_integration_review"][key] is False
        for key in cited["oct19_nonacceptance_flags_retained"]
    )
    assert cited["oct19_observation_retained_historical_only"] == {
        "case_count": 41, "phase_count": 123, "rerun": False,
    }
    assert cited["q4_r4_grade"]["grade"] == "READY-WITH-EXCEPTIONS"
    assert len(cited["q4_r4_grade"]["exceptions"]) == 4
    files = record["historical_files_sha256"]
    assert files[Q4_SOURCE_BINDING.relative_to(ROOT).as_posix()] == Q4_SOURCE_BINDING_SHA256
    assert files[Q4_SOURCE_BINDING_AS_REVIEWED.relative_to(ROOT).as_posix()] == (
        Q4_SOURCE_BINDING_AS_REVIEWED_SHA256
    )
    assert {
        QUALIFICATION_CONTRACT.relative_to(ROOT).as_posix(),
        Q4_REPLAY_HOTFIX.relative_to(ROOT).as_posix(),
        Q4_REPLAY_CANONICALIZATION.relative_to(ROOT).as_posix(),
        "docs/reference_cases/qualified_shell_runtime_guard_performance_successor.json",
        "docs/STATIC_RUNTIME_PERFORMANCE.md",
    } <= set(files)
    for relative, digest in files.items():
        _require_sole_import_form(relative)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper() == digest
    report = record["static_runtime_performance_report"]
    assert report["path"] == "docs/STATIC_RUNTIME_PERFORMANCE.md" and report["bytes"] == 4561
    assert files[report["path"]] == report["sha256"]

    # Explicit recorded-to-live spelling map.
    spelling = record["recorded_to_live_spelling_map"]
    assert spelling["path_map"] == RECORDED_TO_LIVE_PATH
    assert spelling["text_live_to_recorded"] == [list(pair) for pair in TEXT_LIVE_TO_RECORDED]

    # Six carriers: live raw/LF identity, upstream Git ids, reverse-map evidence.
    carriers = record["carriers"]
    assert set(carriers) == set(SIX_CARRIERS)
    for name in SIX_CARRIERS:
        row = carriers[name]
        assert row["recorded_path"] == f"src/anysolver/{name}"
        assert row["live_path"] == _live_path(row["recorded_path"])
        raw = read(row["live_path"])
        _require_physical(row["live_path"], raw, row)
        reverse = _reverse_spelling(_lf(raw))
        assert row["live_lf_after_reverse_spelling_map"] == _identity(reverse)
        assert row["upstream_raw"]["git_blob"] == UPSTREAM_GIT_BLOBS[name]
        assert row["upstream_lf"] == _identity(reverse)
        assert row["reverse_spelling_map_reproduces_upstream_lf"] is True
        assert row["module_symbols"]["candidate_aggregate_sha256"] == _module_symbol_aggregate(
            _lf(raw).decode("utf-8")
        )
        if name in V6_TERMINAL_BLOBS:
            assert row["v6_terminal_git_blob"] == V6_TERMINAL_BLOBS[name]
            assert row["live_lf_equals_v6_terminal"] is (
                _blob_id(_lf(raw)) == V6_TERMINAL_BLOBS[name]
            )

    # Links: mechanics changed after October 19 / the V6 terminal; no extension.
    link1 = record["links"]["L1_OCT19_AND_V6_TERMINAL_TO_QUALIFIED_Q4_R4"]
    assert link1["mechanics_changed"] is True
    assert link1["v6_terminal_mechanics_unchanged_claim_extended"] is False
    assert link1["v6_terminal_blobs"] == V6_TERMINAL_BLOBS
    assert link1["external_kernel_added"] == "_q4_assumed_strain_plasticity.py"
    assert (
        link1["oct19_closure"]["symbol_count"], link1["oct19_closure"]["aggregate_sha256"]
    ) == OCT19_CLOSURE
    assert (
        link1["qualified_target_closure"]["symbol_count"],
        link1["qualified_target_closure"]["aggregate_sha256"],
    ) == UPSTREAM_CLOSURE
    assert {
        "QualifiedE4PLShellElement._assumed_strain_kernel", "_initial_offsets_or_none",
    } <= set(link1["symbol_delta_oct19_to_qualified_target"]["added"])
    assert "unbound_scope" in record["links"]["L2_UPSTREAM_E0C5F32AB_TO_RELEASE_CANDIDATE"]

    _require_successor_mechanics(record, actual, authority)
    _require_sole_import_form(LIVE_ENTRY)
    source = _lf(read(LIVE_ENTRY)).decode("utf-8")
    _require_successor_imports(record, source, actual["symbols"], read)
    return record


def test_historical_oct19_record_reads_on_its_own_terms() -> None:
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    raw = Q4_SOURCE_BINDING_AS_REVIEWED.read_bytes()
    record = _require_historical_q4_source_binding_record(raw, authority)
    assert record["historical_current_source_matches"] is False
    with pytest.raises(AssertionError):
        _require_historical_q4_source_binding_record(raw + b" ", authority)
    # The original pin is never attributed to the pathname's later bytes.
    with pytest.raises(AssertionError):
        _require_historical_q4_source_binding_record(Q4_SOURCE_BINDING.read_bytes(), authority)
    with pytest.raises(AssertionError):
        _live_path("src/anysolver/not_a_recorded_carrier.py")


def test_recorded_to_live_spelling_map_is_explicit_and_checked() -> None:
    assert set(RECORDED_TO_LIVE_PATH) == {f"src/anysolver/{n}" for n in SIX_CARRIERS}
    for recorded, live in RECORDED_TO_LIVE_PATH.items():
        assert (ROOT / live).is_file()
        assert recorded != live and recorded.replace("anysolver", "anyfesolver", 1) == live
    assert _reverse_spelling(b"ANYfeSolver anyfesolver.elements") == b"ANYsolver anysolver.elements"


def test_successor_record_binds_live_source_without_new_acceptance() -> None:
    _require_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    actual = _q4_mechanics_authority_payload(
        _lf(_read_bound(LIVE_ENTRY)).decode("utf-8"), set(authority["roots"]),
        schema=authority["schema"], path=authority["path"],
        required_symbols=set(authority["symbols"]),
    )
    record = _require_q4_source_binding_successor(
        Q4_SOURCE_BINDING_SUCCESSOR.read_bytes(), _read_bound, actual, authority
    )
    assert record["boundary_flags"]["CI_credit"] is False


GAP_PATHS = (
    "src/anyfesolver/e4_pl_element.py",
    "src/anyfesolver/fe_core.py",
    "src/anyfesolver/elements.py",
    "tests/test_general_contact_c5_elastic_increment.py",
    "tests/test_general_contact_c5_numpy_coordinate_admission.py",
)


def _successor_context():
    _require_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    actual = _q4_mechanics_authority_payload(
        _lf(_read_bound(LIVE_ENTRY)).decode("utf-8"), set(authority["roots"]),
        schema=authority["schema"], path=authority["path"],
        required_symbols=set(authority["symbols"]),
    )
    return authority, actual, Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()


@pytest.mark.parametrize("relative", GAP_PATHS)
def test_successor_rejects_content_and_eol_only_mutation_of_protected_carriers(
    relative: str,
) -> None:
    authority, actual, record = _successor_context()
    raw = _read_bound(relative)
    assert relative.rsplit("/", 1)[1] not in PORTABLE_AUTO_TEXT
    eol_only = _lf(raw) if b"\r\n" in raw else _lf(raw).replace(b"\n", b"\r\n")
    assert eol_only != raw and _lf(eol_only) == _lf(raw)
    for mutated in (raw + b"\n# content mutation\n", eol_only):
        with pytest.raises(AssertionError):
            _require_q4_source_binding_successor(
                record, _read_with({relative: mutated}), actual, authority
            )


@pytest.mark.parametrize("name", sorted(PORTABLE_AUTO_TEXT))
def test_portable_auto_text_accepts_only_uniform_lf_or_crlf(name: str) -> None:
    authority, actual, record = _successor_context()
    relative = f"src/anyfesolver/{name}"
    canonical = _lf(_read_bound(relative))
    assert b"\r" not in canonical
    crlf = canonical.replace(b"\n", b"\r\n")
    for good in (canonical, crlf):
        _require_q4_source_binding_successor(
            record, _read_with({relative: good}), actual, authority
        )
    first = canonical.index(b"\n") + 1
    for bad in (
        canonical[:first].replace(b"\n", b"\r\n") + canonical[first:],  # mixed endings
        canonical.replace(b"\n", b"\r\r\n"),  # double CR conversion
        canonical + b"# content mutation\n",
        crlf + b"# content mutation\r\n",
    ):
        with pytest.raises(AssertionError):
            _require_q4_source_binding_successor(
                record, _read_with({relative: bad}), actual, authority
            )


def test_lineage_authentication_rejects_forged_intermediate_parents_and_missing_rows() -> None:
    _require_pin()
    lineage = _strict_json(Q4_SOURCE_BINDING_SUCCESSOR.read_bytes())["historical_record_lineage"]
    assert len(_authenticated_lineage_transitions(lineage)) == 6

    def forge(edit, *, restamp: bool = True) -> dict[str, object]:
        copy = json.loads(json.dumps(lineage))
        edit(copy["nodes"])
        if restamp:  # a forger who also recomputes the chain digest is still refused
            copy["chain_sha256"] = _digest_upper(copy["nodes"])
        return copy

    edits = (
        lambda n: n[3].__setitem__("commit", "0" * 40),  # forged intermediate commit
        lambda n: n[4]["parents"][0].__setitem__("commit", "1" * 40),  # wrong parent
        lambda n: n[8]["parents"].pop(),  # incomplete merge parents
        lambda n: n.pop(5),  # missing row
        lambda n: n.reverse(),
    )
    for edit in edits:
        with pytest.raises(AssertionError):
            _authenticated_lineage_transitions(forge(edit))
    # Invented sha/length/parent blob/changes with matching topology, restamped.
    for invented in (
        lambda n: n[3].__setitem__("sha256", "0" * 64),
        lambda n: n[3].__setitem__("length", n[3]["length"] + 1),
        lambda n: n[3]["parents"][0].__setitem__("blob", "2" * 40),
        lambda n: n[3]["parents"][0]["changes"].clear(),
    ):
        with pytest.raises(AssertionError):
            _authenticated_lineage_transitions(forge(invented))
    with pytest.raises(AssertionError):  # stale chain digest
        _authenticated_lineage_transitions(
            forge(lambda n: n[3].__setitem__("commit", "0" * 40), restamp=False)
        )


@pytest.mark.parametrize("name", SIX_CARRIERS)
def test_successor_rejects_any_unrecorded_carrier_file_mutation(name: str) -> None:
    _require_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    actual = _q4_mechanics_authority_payload(
        _lf(_read_bound(LIVE_ENTRY)).decode("utf-8"), set(authority["roots"]),
        schema=authority["schema"], path=authority["path"],
        required_symbols=set(authority["symbols"]),
    )
    record = Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()
    relative = RECORDED_TO_LIVE_PATH[f"src/anysolver/{name}"]
    read = _read_with({relative: _read_bound(relative) + b"\n# unrecorded mutation\n"})
    with pytest.raises(AssertionError):
        _require_q4_source_binding_successor(record, read, actual, authority)


def test_successor_mechanics_and_import_checks_reject_unrecorded_changes() -> None:
    _require_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    record = _strict_json(Q4_SOURCE_BINDING_SUCCESSOR.read_bytes())
    source = _lf(_read_bound(LIVE_ENTRY)).decode("utf-8")
    actual = _q4_mechanics_authority_payload(
        source, set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    _require_successor_mechanics(record, actual, authority)
    _require_successor_imports(record, source, actual["symbols"], _read_bound)

    # Symbol mutation inside the closure is visible.
    sign = "        sign = 1 if alignment > 0.0 else -1"
    assert source.count(sign) == 1
    mutated = _q4_mechanics_authority_payload(
        source.replace(sign, "        sign = -1 if alignment > 0.0 else 1", 1),
        set(authority["roots"]), schema=authority["schema"], path=authority["path"],
        required_symbols=set(authority["symbols"]),
    )
    with pytest.raises(AssertionError):
        _require_successor_mechanics(record, mutated, authority)

    # A new, unrecorded sibling import is visible even if unused.
    with pytest.raises(AssertionError):
        _require_successor_imports(
            record, source + "\nfrom . import fe_core as _unrecorded_dependency\n",
            actual["symbols"], _read_bound,
        )

    # Mutating any recorded sibling dependency file is visible.
    for row in record["sibling_imports_of_entry_module"]:
        read = _read_with({row["live_path"]: _read_bound(row["live_path"]) + b"\n#x\n"})
        with pytest.raises(AssertionError):
            _require_successor_imports(record, source, actual["symbols"], read)


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
    # Frozen contract spelling is src/anysolver/...; resolve via the explicit map.
    assert _live_path(authority["path"]) == QUALIFIED_Q4_MECHANICS.relative_to(ROOT).as_posix()
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
        _read_bound(LIVE_ENTRY).decode("utf-8"),
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
    _require_pin()
    binding = _require_q4_source_binding_successor(
        Q4_SOURCE_BINDING_SUCCESSOR.read_bytes(), _read_bound, actual, authority
    )
    # Original qualification contract (129 expected symbols) -> candidate live
    # closure.  This is NOT the October 19 -> qualified-R4 link (links.L1).  The
    # successor record has no historical_difference field; its candidate closure
    # is binding["mechanics"]["symbols"], which the reader above already proved
    # equal to the independently recomputed live `actual`.
    record_symbols = binding["mechanics"]["symbols"]
    assert record_symbols == actual["symbols"]
    derived = {
        "changed_symbols": sorted(
            name for name, digest in expected.items()
            if name in record_symbols and record_symbols[name] != digest
        ),
        "added_symbols": sorted(set(record_symbols) - set(expected)),
        "removed_symbols": sorted(set(expected) - set(record_symbols)),
    }
    assert derived == {
        "changed_symbols": changed,
        "added_symbols": added,
        "removed_symbols": sorted(set(expected) - set(actual["symbols"])),
    }
    assert derived["removed_symbols"] == []  # contract symbols all remain required
    assert derived["changed_symbols"] and derived["added_symbols"]


def test_prospective_q4_source_binding_rejects_unknown_record_and_outside_closure() -> None:
    _require_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    raw = _read_bound(LIVE_ENTRY)
    source = raw.decode("utf-8")
    actual = _q4_mechanics_authority_payload(
        source, set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    record = Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()
    with pytest.raises(AssertionError):
        _require_q4_source_binding_successor(record + b" ", _read_bound, actual, authority)
    for field in ("roots", "symbols"):
        refused = _strict_json(json.dumps(actual).encode("utf-8"))
        if field == "roots":
            refused[field] = refused[field][1:]
        else:
            refused[field].pop(next(iter(authority["symbols"])))
        with pytest.raises(AssertionError):
            _require_q4_source_binding_successor(record, _read_bound, refused, authority)
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
        _require_q4_source_binding_successor(
            record, _read_with({LIVE_ENTRY: changed.encode("utf-8")}),
            unchanged_closure, authority,
        )


def test_prospective_q4_source_binding_rejects_external_dependency_drift(monkeypatch) -> None:
    _require_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    raw = _read_bound(LIVE_ENTRY)
    actual = _q4_mechanics_authority_payload(
        raw.decode("utf-8"), set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    record = Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()
    dependency = ROOT / "src" / "anyfesolver" / "vectorized_nonlinear.py"
    original = Path.read_bytes
    def changed_bytes(path):
        value = original(path)
        return value + b"\n# injected source drift\n" if path == dependency else value
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    with pytest.raises(AssertionError):
        _require_q4_source_binding_successor(record, _read_bound, actual, authority)

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


# ---------------------------------------------------------------------------
# 2026-11-24 source-binding delta (warped-Q4 drilling floor).  SOURCE BINDING
# ONLY: the live bytes of the files the 2026-11-11 record binds, the exact
# reverse hunks to those predecessor bytes, the per-module symbol deltas and
# the live qualified-Q4 mechanics closure.  No acceptance is granted.

DELTA_FILE_KEYS = {
    "live_lf", "live_raw", "module_symbols", "predecessor_raw", "reverse_hunks",
}


def _require_delta_pin() -> None:
    if Q4_SOURCE_BINDING_DELTA_SHA256 is None:
        pytest.fail(
            "source-binding delta is not adopted: run "
            "scripts/build_q4_source_binding_20261124.py --write, have the record, "
            "this reader and the collector independently reviewed outside the "
            "generating invocation, then pin Q4_SOURCE_BINDING_DELTA_SHA256",
            pytrace=False,
        )


def _module_symbol_digests(source: str) -> dict[str, str]:
    return {
        name: _digest_upper([_canonical_mechanics_ast(node) for node in nodes])
        for name, nodes in sorted(_mechanics_nodes(source).items())
    }


def _symbol_delta(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(after) - set(before)),
        "changed": sorted(n for n in set(before) & set(after) if before[n] != after[n]),
        "removed": sorted(set(before) - set(after)),
    }


def _predecessor_live_identities(predecessor: dict[str, object]) -> dict[str, object]:
    """Every live path the 2026-11-11 record binds by exact raw identity."""

    bound: dict[str, object] = {}
    for row in predecessor["carriers"].values():
        if "live_raw" in row:
            bound[row["live_path"]] = row["live_raw"]
    for row in predecessor["sibling_imports_of_entry_module"]:
        if "live_raw" in row:
            bound.setdefault(row["live_path"], row["live_raw"])
    for row in predecessor["historical_gaps_of_current_historical_record"]:
        bound.setdefault(row["live_path"], row["live_raw"])
    return bound


def _require_q4_source_binding_delta(record_raw: bytes, read, authority) -> dict[str, object]:
    """Chain reader.  Scope (owner decision 2026-11-25, chain 20261111 -> 20261124 ->
    20261125): tracked-source bytes and same-directory import-form uniqueness only;
    runtime import identity (sys.path order, bytecode caches, installed copies) is
    out of scope.  See _require_sole_import_form.
    """
    assert Q4_SOURCE_BINDING_DELTA_SHA256 is not None
    assert hashlib.sha256(record_raw).hexdigest().upper() == Q4_SOURCE_BINDING_DELTA_SHA256
    record = _strict_json(record_raw)
    assert record_raw == _canonical_bytes(record)
    assert record["schema"] == "anyfesolver.q4-source-binding-delta-v1"
    assert record["qualification_boundary"] == "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"
    assert record["adoption"] == {
        "independent_review_required_outside_generating_invocation": True,
        "owner_adoption_recorded": False,
        "reviewer_approval_claimed": False,
        "status": "UNREVIEWED_CANDIDATE_RECORD",
    }
    assert set(record["boundary_flags"]) == SUCCESSOR_FLAG_KEYS
    assert all(value is False for value in record["boundary_flags"].values())
    change = record["change"]
    assert change["mechanics_changed"] is True
    assert change["warped_qualified_q4_results_changed"] is True
    assert change["planar_and_other_element_results_changed"] is False
    assert (ROOT / change["regression_test"]).is_file()
    base = record["base_revision"]
    assert len(base) == 40 and set(base) <= set("0123456789abcdef")

    # Predecessor: exactly the pinned 2026-11-11 record, read on its own terms
    # against the predecessor bytes rebuilt from the bytes ``read`` serves.
    predecessor_raw = Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()
    assert Q4_SOURCE_BINDING_SUCCESSOR_SHA256 is not None
    assert hashlib.sha256(predecessor_raw).hexdigest().upper() == Q4_SOURCE_BINDING_SUCCESSOR_SHA256
    assert record["predecessor"] == {
        "path": Q4_SOURCE_BINDING_SUCCESSOR.relative_to(ROOT).as_posix(),
        **_identity(predecessor_raw),
    }
    predecessor = _strict_json(predecessor_raw)
    bound = _predecessor_live_identities(predecessor)
    files = record["files"]
    assert files and set(files) <= set(bound)

    def bound_read(relative: str) -> bytes:
        return _read_bound(relative, read)

    predecessor_actual = _q4_mechanics_authority_payload(
        _lf(bound_read(LIVE_ENTRY)).decode("utf-8"), set(authority["roots"]),
        schema=authority["schema"], path=authority["path"],
        required_symbols=set(authority["symbols"]),
    )
    _require_q4_source_binding_successor(predecessor_raw, bound_read, predecessor_actual, authority)

    # Changed files: live identity, exact rebuild, per-module symbol deltas.
    for relative, row in files.items():
        assert set(row) == DELTA_FILE_KEYS
        live = read(relative)
        assert row["live_raw"] == _identity(live)
        assert row["live_lf"] == _identity(_lf(live))
        assert row["predecessor_raw"] == bound[relative]
        previous = _rebuild_predecessor(live, row["reverse_hunks"])
        assert _identity(previous) == bound[relative]
        assert previous != live
        before = _module_symbol_digests(_lf(previous).decode("utf-8"))
        after = _module_symbol_digests(_lf(live).decode("utf-8"))
        symbols = row["module_symbols"]
        assert {k: symbols[k] for k in ("added", "changed", "removed")} == _symbol_delta(before, after)
        assert symbols["predecessor_aggregate_sha256"] == _digest_upper(before)
        assert symbols["live_aggregate_sha256"] == _digest_upper(after)
        assert symbols["live_aggregate_sha256"] == _module_symbol_aggregate(_lf(live).decode("utf-8"))
    # Every other predecessor-bound file is byte-for-byte unchanged, and every
    # bound file is the only importable form of its module.
    for relative, identity in bound.items():
        _require_sole_import_form(relative)
        if relative not in files:
            assert _identity(read(relative)) == identity

    # Live qualified-Q4 mechanics closure and its delta from the predecessor.
    _require_sole_import_form(LIVE_ENTRY)
    source = _lf(read(LIVE_ENTRY)).decode("utf-8")
    actual = _q4_mechanics_authority_payload(
        source, set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    mech = record["mechanics"]
    assert actual["roots"] == authority["roots"] == mech["roots"]
    assert set(authority["symbols"]) <= set(actual["symbols"])
    assert actual["symbols"] == mech["symbols"]
    assert mech["symbol_count"] == len(actual["symbols"])
    assert actual["path"] == mech["logical_path"] == authority["path"]
    assert actual["schema"] == mech["schema"] == authority["schema"]
    assert _q4_mechanics_aggregate(actual) == mech["aggregate_sha256"]
    assert mech["predecessor_aggregate_sha256"] == predecessor["mechanics"]["aggregate_sha256"]
    assert mech["delta_from_predecessor"] == _symbol_delta(
        predecessor["mechanics"]["symbols"], actual["symbols"]
    )
    # No new sibling import of the entry module.
    live_keys = {
        "live_path", "recorded_path", "live_raw", "live_lf", "portable_text", "permitted_raw",
    }
    assert record["sibling_imports_of_entry_module_unchanged"] is True
    assert _internal_imports(source, sorted(actual["symbols"])) == [
        {k: v for k, v in row.items() if k not in live_keys}
        for row in predecessor["sibling_imports_of_entry_module"]
    ]
    return record


def test_predecessor_bytes_rebuild_exactly_and_refuse_tampering() -> None:
    """Runs without the delta pin: the rebuild is authenticated by 2026-11-11."""

    _require_pin()
    delta = _delta_record()
    bound = _predecessor_live_identities(_strict_json(Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()))
    assert delta["files"] and set(delta["files"]) <= set(bound)
    for relative, row in delta["files"].items():
        # The 2026-11-24-bound bytes (live, rebuilt through 2026-11-25).
        live = _read_delta_bound(relative)
        assert _identity(_read_bound(relative)) == bound[relative]
        hunks = row["reverse_hunks"]
        assert hunks
        tampered = json.loads(json.dumps(hunks))
        tampered[0]["predecessor_text"] += "#"
        with pytest.raises(AssertionError):
            assert _identity(_rebuild_predecessor(live, tampered)) == bound[relative]
        shifted = json.loads(json.dumps(hunks))
        shifted[0]["live_line_start"] += 1
        shifted[0]["live_line_end"] += 1
        with pytest.raises(AssertionError):
            assert _identity(_rebuild_predecessor(live, shifted)) == bound[relative]

        def drifted(path: str, _relative=relative, _live=live) -> bytes:
            return _live + b"\n# drift\n" if path == _relative else _read_delta_bound(path)

        with pytest.raises(AssertionError):
            _read_bound(relative, drifted)


def test_source_binding_delta_binds_live_source_without_new_acceptance() -> None:
    _require_pin()
    _require_delta_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    record = _require_q4_source_binding_delta(
        Q4_SOURCE_BINDING_DELTA.read_bytes(), _read_delta_bound, authority
    )
    assert set(record["files"]) == {
        "src/anyfesolver/e4_pl_element.py", "src/anyfesolver/elements.py",
    }


@pytest.mark.parametrize("relative", GAP_PATHS)
def test_source_binding_delta_rejects_content_and_eol_mutation(relative: str) -> None:
    _require_pin()
    _require_delta_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    record = Q4_SOURCE_BINDING_DELTA.read_bytes()
    raw = _read_delta_bound(relative)
    eol_only = _lf(raw) if b"\r\n" in raw else _lf(raw).replace(b"\n", b"\r\n")
    for mutated in (raw + b"\n# content mutation\n", eol_only):
        def read(path: str, _mutated=mutated) -> bytes:
            return _mutated if path == relative else _read_delta_bound(path)
        with pytest.raises(AssertionError):
            _require_q4_source_binding_delta(record, read, authority)
    with pytest.raises(AssertionError):
        _require_q4_source_binding_delta(record + b" ", _read_delta_bound, authority)


# ---------------------------------------------------------------------------
# 2026-11-25 source-binding delta, chained from the 2026-11-24 delta (S2,
# bit-identical beam frame-math trim in elements.py).  SOURCE BINDING ONLY: the
# live bytes of the files the chain binds, the exact reverse hunks to the
# 2026-11-24-bound bytes, the per-module symbol deltas and the live
# qualified-Q4 mechanics closure (unchanged).  No acceptance is granted.

DELTA2_ELEMENTS = "src/anyfesolver/elements.py"
DELTA2_MODULE_SYMBOLS = {
    "added": ["_contiguous_vector_norm", "_frame_columns"],
    "changed": [
        "BeamElement",
        "BeamElement._beam_frame_and_transform",
        "BeamElement._linear_rigid_arm_map",
        "_beam_rotation_matrix",
    ],
    "removed": [],
}


def _require_delta2_pin() -> None:
    if Q4_SOURCE_BINDING_DELTA2_SHA256 is None:
        pytest.fail(
            "chained source-binding delta (2026-11-25) is not adopted: run "
            "scripts/build_q4_source_binding_20261125.py --write, have the record, "
            "this reader and the collector independently reviewed outside the "
            "generating invocation, then pin Q4_SOURCE_BINDING_DELTA2_SHA256",
            pytrace=False,
        )


def _chain_live_identities() -> dict[str, object]:
    """Every live path the 2026-11-24 chain binds by exact raw identity."""

    bound = _predecessor_live_identities(_strict_json(Q4_SOURCE_BINDING_SUCCESSOR.read_bytes()))
    for relative, row in _delta_record()["files"].items():
        bound[relative] = row["live_raw"]
    return bound


def _require_q4_source_binding_delta2(record_raw: bytes, read, authority) -> dict[str, object]:
    """Chain reader.  Scope (owner decision 2026-11-25, chain 20261111 -> 20261124 ->
    20261125): tracked-source bytes and same-directory import-form uniqueness only;
    runtime import identity (sys.path order, bytecode caches, installed copies) is
    out of scope.  See _require_sole_import_form.
    """
    assert Q4_SOURCE_BINDING_DELTA2_SHA256 is not None
    assert hashlib.sha256(record_raw).hexdigest().upper() == Q4_SOURCE_BINDING_DELTA2_SHA256
    record = _strict_json(record_raw)
    assert record_raw == _canonical_bytes(record)
    assert record["schema"] == "anyfesolver.q4-source-binding-delta-v1"
    assert record["qualification_boundary"] == "SOURCE_BINDING_ONLY_NOT_SCIENTIFIC_QUALIFICATION"
    assert record["adoption"] == {
        "independent_review_required_outside_generating_invocation": True,
        "owner_adoption_recorded": False,
        "reviewer_approval_claimed": False,
        "status": "UNREVIEWED_CANDIDATE_RECORD",
    }
    assert set(record["boundary_flags"]) == SUCCESSOR_FLAG_KEYS
    assert all(value is False for value in record["boundary_flags"].values())
    change = record["change"]
    assert change["mechanics_changed"] is False
    assert change["warped_qualified_q4_results_changed"] is False
    assert change["planar_and_other_element_results_changed"] is False
    assert change["bit_identical_results_claimed"] is True
    assert change["upstream_reproduction_claim_replaced_by_successor_lineage"] is True
    assert change["changed_module_symbols"] == {DELTA2_ELEMENTS: DELTA2_MODULE_SYMBOLS}
    assert (ROOT / change["regression_test"]).is_file()
    base = record["base_revision"]
    assert len(base) == 40 and set(base) <= set("0123456789abcdef")

    # Predecessor: exactly the pinned 2026-11-24 record, read on its own terms
    # against the 2026-11-24-bound bytes rebuilt from the bytes ``read`` serves.
    predecessor_raw = Q4_SOURCE_BINDING_DELTA.read_bytes()
    assert Q4_SOURCE_BINDING_DELTA_SHA256 is not None
    assert record["predecessor"] == {
        "path": Q4_SOURCE_BINDING_DELTA.relative_to(ROOT).as_posix(),
        **_identity(predecessor_raw),
    }
    predecessor = _strict_json(predecessor_raw)
    bound = _chain_live_identities()
    files = record["files"]
    assert set(files) == {DELTA2_ELEMENTS}

    def delta_bound_read(relative: str) -> bytes:
        # Rebuild only from the record authenticated above, never a second copy.
        return _read_delta_bound(relative, read, record)

    _require_q4_source_binding_delta(predecessor_raw, delta_bound_read, authority)

    # Changed files: live identity, exact rebuild, per-module symbol deltas.
    for relative, row in files.items():
        assert set(row) == DELTA_FILE_KEYS
        live = read(relative)
        assert row["live_raw"] == _identity(live)
        assert row["live_lf"] == _identity(_lf(live))
        assert row["predecessor_raw"] == bound[relative]
        previous = _rebuild_predecessor(live, row["reverse_hunks"])
        assert _identity(previous) == bound[relative]
        assert previous != live
        before = _module_symbol_digests(_lf(previous).decode("utf-8"))
        after = _module_symbol_digests(_lf(live).decode("utf-8"))
        symbols = row["module_symbols"]
        delta = _symbol_delta(before, after)
        assert {k: symbols[k] for k in ("added", "changed", "removed")} == delta
        assert delta == change["changed_module_symbols"][relative]
        assert symbols["predecessor_aggregate_sha256"] == _digest_upper(before)
        assert symbols["live_aggregate_sha256"] == _digest_upper(after)
        assert symbols["live_aggregate_sha256"] == _module_symbol_aggregate(_lf(live).decode("utf-8"))
    # Every other chain-bound file is byte-for-byte unchanged, and every bound
    # file is the only importable form of its module.
    for relative, identity in bound.items():
        _require_sole_import_form(relative)
        if relative not in files:
            assert _identity(read(relative)) == identity

    # Live qualified-Q4 mechanics closure: unchanged from the predecessor.
    _require_sole_import_form(LIVE_ENTRY)
    source = _lf(read(LIVE_ENTRY)).decode("utf-8")
    actual = _q4_mechanics_authority_payload(
        source, set(authority["roots"]), schema=authority["schema"],
        path=authority["path"], required_symbols=set(authority["symbols"]),
    )
    mech = record["mechanics"]
    assert actual["roots"] == authority["roots"] == mech["roots"]
    assert actual["symbols"] == mech["symbols"] == predecessor["mechanics"]["symbols"]
    assert mech["symbol_count"] == len(actual["symbols"])
    assert actual["path"] == mech["logical_path"] == authority["path"]
    assert actual["schema"] == mech["schema"] == authority["schema"]
    assert _q4_mechanics_aggregate(actual) == mech["aggregate_sha256"]
    assert mech["aggregate_sha256"] == mech["predecessor_aggregate_sha256"]
    assert mech["predecessor_aggregate_sha256"] == predecessor["mechanics"]["aggregate_sha256"]
    assert mech["delta_from_predecessor"] == {"added": [], "changed": [], "removed": []}
    live_keys = {
        "live_path", "recorded_path", "live_raw", "live_lf", "portable_text", "permitted_raw",
    }
    assert record["sibling_imports_of_entry_module_unchanged"] is True
    assert _internal_imports(source, sorted(actual["symbols"])) == [
        {k: v for k, v in row.items() if k not in live_keys}
        for row in _strict_json(Q4_SOURCE_BINDING_SUCCESSOR.read_bytes())[
            "sibling_imports_of_entry_module"
        ]
    ]
    return record


def test_delta_bound_bytes_rebuild_exactly_and_refuse_tampering() -> None:
    """Runs without the 2026-11-25 pin: the rebuild is authenticated by the
    pinned 2026-11-24 record's live identities."""

    _require_pin()
    delta2 = _delta2_record()
    bound = _chain_live_identities()
    assert set(delta2["files"]) == {DELTA2_ELEMENTS}
    for relative, row in delta2["files"].items():
        live = _read_live(relative)
        assert _identity(_read_delta_bound(relative)) == bound[relative]
        hunks = row["reverse_hunks"]
        assert hunks
        tampered = json.loads(json.dumps(hunks))
        tampered[0]["predecessor_text"] += "#"
        with pytest.raises(AssertionError):
            assert _identity(_rebuild_predecessor(live, tampered)) == bound[relative]
        shifted = json.loads(json.dumps(hunks))
        shifted[0]["live_line_start"] += 1
        shifted[0]["live_line_end"] += 1
        with pytest.raises(AssertionError):
            assert _identity(_rebuild_predecessor(live, shifted)) == bound[relative]

        def drifted(path: str, _relative=relative, _live=live) -> bytes:
            return _live + b"\n# drift\n" if path == _relative else _read_live(path)

        with pytest.raises(AssertionError):
            _read_delta_bound(relative, drifted)
        with pytest.raises(AssertionError):
            _read_bound(relative, lambda path: _read_delta_bound(path, drifted))


def test_chained_source_binding_delta_binds_live_source_without_new_acceptance() -> None:
    _require_pin()
    _require_delta_pin()
    _require_delta2_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    record = _require_q4_source_binding_delta2(
        Q4_SOURCE_BINDING_DELTA2.read_bytes(), _read_live, authority
    )
    assert set(record["files"]) == {DELTA2_ELEMENTS}


@pytest.mark.parametrize("relative", GAP_PATHS)
def test_chained_source_binding_delta_rejects_content_and_eol_mutation(relative: str) -> None:
    _require_pin()
    _require_delta_pin()
    _require_delta2_pin()
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    record = Q4_SOURCE_BINDING_DELTA2.read_bytes()
    raw = _read_live(relative)
    eol_only = _lf(raw) if b"\r\n" in raw else _lf(raw).replace(b"\n", b"\r\n")
    for mutated in (raw + b"\n# content mutation\n", eol_only):
        def read(path: str, _mutated=mutated) -> bytes:
            return _mutated if path == relative else _read_live(path)
        with pytest.raises(AssertionError):
            _require_q4_source_binding_delta2(record, read, authority)
    with pytest.raises(AssertionError):
        _require_q4_source_binding_delta2(record + b" ", _read_live, authority)


def _record_with_extra_reverse_hunk(relative: str, drifted: bytes, original: bytes):
    """The 2026-11-25 record plus a reverse hunk that hides drift in ``relative``."""

    record = json.loads(Q4_SOURCE_BINDING_DELTA2.read_bytes())
    lines = _DELTA_LINE.findall(drifted)
    start = len(_DELTA_LINE.findall(original))
    record["files"][relative] = {
        "predecessor_raw": _identity(original),
        "reverse_hunks": [{
            "live_line_start": start,
            "live_line_end": len(lines),
            "live_lines_sha256": hashlib.sha256(b"".join(lines[start:])).hexdigest().upper(),
            "predecessor_text": "",
        }],
    }
    return record


def test_chained_reader_rebuilds_only_from_the_authenticated_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review P2 (fd830773e): with authentic ``record_raw`` the nested
    reader rebuilt from a second, unauthenticated copy of the 2026-11-25
    record, so a reverse hunk added there hid drift in another carrier."""

    _require_pin()
    _require_delta_pin()
    authentic = Q4_SOURCE_BINDING_DELTA2.read_bytes()
    monkeypatch.setattr(
        sys.modules[__name__], "Q4_SOURCE_BINDING_DELTA2_SHA256",
        hashlib.sha256(authentic).hexdigest().upper(),
    )
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    relative = "src/anyfesolver/_q4_assumed_strain_plasticity.py"
    original = _read_live(relative)
    drifted = original + b"# unrecorded drift\n"
    tampered = _record_with_extra_reverse_hunk(relative, drifted, original)
    monkeypatch.setattr(sys.modules[__name__], "_delta2_record", lambda: tampered)

    def read(path: str) -> bytes:
        return drifted if path == relative else _read_live(path)

    with pytest.raises(AssertionError):
        _require_q4_source_binding_delta2(authentic, read, authority)
    # The adapter fails closed on a 2026-11-25 record changing any other file.
    with pytest.raises(AssertionError):
        _read_delta_bound(relative, read, tampered)
    with pytest.raises(AssertionError):
        _read_delta_bound(relative, read)


# ---------------------------------------------------------------------------
# Codex re-review of fa6d84176.  P2: a sibling importable form (package,
# namespace directory, bytecode or extension module) beside a bound carrier
# changes what Python imports while the readers hash the bound ``.py`` bytes.
# P3: a falsey reader callback fell back to the working-tree reader.

SHADOWED_CARRIER = "src/anyfesolver/_q4_assumed_strain_plasticity.py"


@pytest.fixture(scope="module")
def _import_form_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of every file the import-form check inspects."""

    import shutil

    root = tmp_path_factory.mktemp("import_form_root")
    shutil.copytree(
        ROOT / "src" / "anyfesolver", root / "src" / "anyfesolver",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for relative in _chain_live_identities():
        target = root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
    return root


def _shadow(root: Path, form: str):
    module = root / SHADOWED_CARRIER[: -len(".py")]
    if form == "package":
        module.mkdir()
        (module / "__init__.py").write_bytes(b"")
        return lambda: (os.remove(module / "__init__.py"), module.rmdir())
    if form == "namespace":
        module.mkdir()
        return module.rmdir
    suffix = {"sourceless": ".pyc", "extension": importlib.machinery.EXTENSION_SUFFIXES[0]}[form]
    path = module.with_name(module.name + suffix)
    path.write_bytes(b"\0")
    return lambda: os.remove(path)


def _run_reader(which: str, read=None) -> None:
    authority = _strict_json(QUALIFICATION_CONTRACT.read_bytes())["candidate"][
        "qualified_q4_mechanics_authority"
    ]
    if which == "20261111":
        actual = _q4_mechanics_authority_payload(
            _lf(_read_bound(LIVE_ENTRY)).decode("utf-8"), set(authority["roots"]),
            schema=authority["schema"], path=authority["path"],
            required_symbols=set(authority["symbols"]),
        )
        _require_q4_source_binding_successor(
            Q4_SOURCE_BINDING_SUCCESSOR.read_bytes(),
            _read_bound if read is None else (lambda path: _read_bound(path, lambda p: _read_delta_bound(p, read))),
            actual, authority,
        )
    elif which == "20261124":
        _require_q4_source_binding_delta(
            Q4_SOURCE_BINDING_DELTA.read_bytes(),
            _read_delta_bound if read is None else (lambda path: _read_delta_bound(path, read)),
            authority,
        )
    else:
        _require_q4_source_binding_delta2(
            Q4_SOURCE_BINDING_DELTA2.read_bytes(), _read_live if read is None else read, authority,
        )


def _pin_delta2_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The authentic on-disk bytes; the committed pin stays None until review.
    monkeypatch.setattr(
        sys.modules[__name__], "Q4_SOURCE_BINDING_DELTA2_SHA256",
        hashlib.sha256(Q4_SOURCE_BINDING_DELTA2.read_bytes()).hexdigest().upper(),
    )


@pytest.mark.parametrize(
    "which, form",
    [
        ("20261111", "package"),
        ("20261111", "namespace"),
        ("20261111", "sourceless"),
        ("20261111", "extension"),
        ("20261124", "package"),
        ("20261125", "package"),
    ],
)
def test_readers_refuse_another_importable_form_beside_a_bound_carrier(
    monkeypatch: pytest.MonkeyPatch, _import_form_copy: Path, which: str, form: str,
) -> None:
    _require_pin()
    _require_delta_pin()
    _pin_delta2_in_memory(monkeypatch)
    monkeypatch.setattr(sys.modules[__name__], "IMPORT_FORM_ROOT", _import_form_copy)
    _run_reader(which)  # control: the unshadowed copy is accepted
    undo = _shadow(_import_form_copy, form)
    try:
        with pytest.raises(AssertionError):
            _run_reader(which)
    finally:
        undo()


def test_sole_import_form_holds_for_every_live_bound_carrier() -> None:
    _require_pin()
    for relative in _chain_live_identities():
        _require_sole_import_form(relative)
    _require_sole_import_form(SHADOWED_CARRIER)


class _FalseyDriftingReader:
    """A callable whose truth value is False and which serves drifted bytes."""

    def __bool__(self) -> bool:
        return False

    def __call__(self, path: str) -> bytes:
        raw = _read_live(path)
        return raw + b"# drift\n" if path == SHADOWED_CARRIER else raw


@pytest.mark.parametrize("which", ["20261111", "20261124", "20261125"])
def test_falsey_reader_callback_is_used_not_replaced(
    monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    _require_pin()
    _require_delta_pin()
    _pin_delta2_in_memory(monkeypatch)
    read = _FalseyDriftingReader()
    drifted = read(SHADOWED_CARRIER)
    assert _read_delta_bound(SHADOWED_CARRIER, read) == drifted
    assert _read_bound(SHADOWED_CARRIER, read) == drifted
    with pytest.raises(AssertionError):
        _run_reader(which, read)


# ---------------------------------------------------------------------------
# Codex Astra review of d0df326b6.  Every path a reader binds gets the
# same-directory import-form check (including the historical-gap files of the
# 2026-11-11 reader); case spellings are judged against the exact bound name;
# an extensionless regular file is not an importable form.

GAP_SHADOWS = (
    "tests/test_general_contact_c5_elastic_increment",
    "tests/test_general_contact_c5_numpy_coordinate_admission",
)


def test_successor_reader_refuses_a_package_beside_each_historical_gap_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _require_pin()
    successor = _strict_json(Q4_SOURCE_BINDING_SUCCESSOR.read_bytes())
    gap_paths = {g["live_path"] for g in successor["historical_gaps_of_current_historical_record"]}
    for shadow in GAP_SHADOWS:
        assert shadow + ".py" in gap_paths
    monkeypatch.setattr(sys.modules[__name__], "IMPORT_FORM_ROOT", tmp_path)
    for relative in gap_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    gaps = successor["historical_gaps_of_current_historical_record"]
    reviewed_raw = Q4_SOURCE_BINDING_AS_REVIEWED.read_bytes()
    current_raw = Q4_SOURCE_BINDING.read_bytes()
    for relative in sorted(gap_paths):
        _require_sole_import_form(relative)  # control
    for shadow in GAP_SHADOWS:
        (tmp_path / shadow).mkdir()
        (tmp_path / shadow / "__init__.py").write_bytes(b"")
        try:
            with pytest.raises(AssertionError, match="shadows"):
                _require_current_historical_record(
                    current_raw, reviewed_raw, successor,
                    _read_bound,
                )
        finally:
            os.remove(tmp_path / shadow / "__init__.py")
            (tmp_path / shadow).rmdir()
    assert len(gaps) == 5


def _listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(sys.modules[__name__], "IMPORT_FORM_ROOT", tmp_path)
    (tmp_path / "pkg").mkdir()
    for name in names:
        if name.endswith("/"):
            (tmp_path / "pkg" / name[:-1]).mkdir()
        else:
            (tmp_path / "pkg" / name).write_bytes(b"")


def test_import_form_check_rejects_a_case_variant_beside_the_bound_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _listing(tmp_path, monkeypatch, "elements.py")
    names = [("elements.py", False), ("ELEMENTS.py", False)]
    monkeypatch.setattr(sys.modules[__name__], "_import_form_entries", lambda directory: names)
    with pytest.raises(AssertionError, match="another"):
        _require_sole_import_form("pkg/elements.py")
    (tmp_path / "pkg" / "ELEMENTS.py").write_bytes(b"")
    names[:] = [("ELEMENTS.py", False), ("elements.py", False)]
    with pytest.raises(AssertionError, match="another"):
        _require_sole_import_form("pkg/ELEMENTS.py")


def test_import_form_check_ignores_an_extensionless_regular_file_but_not_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _listing(tmp_path, monkeypatch, "elements.py", "elements")
    _require_sole_import_form("pkg/elements.py")  # ordinary file named like the stem
    (tmp_path / "pkg" / "elements").unlink()
    (tmp_path / "pkg" / "elements").mkdir()  # namespace directory, no __init__.py
    with pytest.raises(AssertionError, match="shadows"):
        _require_sole_import_form("pkg/elements.py")
    (tmp_path / "pkg" / "elements" / "__init__.py").write_bytes(b"")
    with pytest.raises(AssertionError, match="shadows"):
        _require_sole_import_form("pkg/elements.py")
    (tmp_path / "pkg" / "elements" / "__init__.py").unlink()
    (tmp_path / "pkg" / "elements").rmdir()
    (tmp_path / "pkg" / "elements.pyc").write_bytes(b"\0")  # importable suffix
    with pytest.raises(AssertionError, match="importable form"):
        _require_sole_import_form("pkg/elements.py")


def test_import_form_check_ignores_a_directory_named_like_a_suffix_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Codex Astra P3: a directory ``elements.pyc/`` is no module form of ``elements``.
    _listing(tmp_path, monkeypatch, "elements.py", "elements.pyc/")
    _require_sole_import_form("pkg/elements.py")
    (tmp_path / "pkg" / "elements.pyc").rmdir()
    (tmp_path / "pkg" / "elements.pyc").write_bytes(b" ")  # the regular file still shadows
    with pytest.raises(AssertionError, match="importable form"):
        _require_sole_import_form("pkg/elements.py")
