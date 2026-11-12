"""Development old/new broad-phase parity and whole-route cost comparison."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from anyfesolver.general_contact import patch_mortar
from anyfesolver.general_contact.contact_search import ContactAABBIndex
from examples.general_contact_patch_sliding import main as sliding_example


BASE_COMMIT = "53871171"
BASE_PATH = "src/anyfesolver/general_contact/patch_broadphase.py"


class OwnerPatchAABBTree:
    """Disposable planar adapter used only for old/new route measurements."""

    def __init__(self, boxes):
        values = np.asarray(boxes, dtype=float)
        embedded = np.column_stack((values[:, :2], np.zeros(len(values)),
                                    values[:, 2:], np.zeros(len(values))))
        self._index = ContactAABBIndex(
            tuple(("patch_face", index) for index in range(len(values))),
            embedded,
        )

    def query(self, box, *, padding=0.0):
        row = np.asarray(box, dtype=float)
        embedded = np.array((row[0], row[1], 0.0,
                             row[2], row[3], 0.0))
        return tuple(key[1] for key in self._index.query(
            embedded, padding=padding))


def _historical_tree():
    source = subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:{BASE_PATH}"],
        check=True, capture_output=True, text=True,
    ).stdout
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == "PatchAABBTree")
    code = compile(ast.Module(body=[node], type_ignores=[]),
                   f"{BASE_COMMIT}:{BASE_PATH}", "exec")
    scope = {"np": np}
    exec(code, scope)
    return scope["PatchAABBTree"]


def _run_example(tree_type):
    previous = patch_mortar.PatchAABBTree
    patch_mortar.PatchAABBTree = tree_type
    started = time.perf_counter()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            sliding_example()
    finally:
        patch_mortar.PatchAABBTree = previous
    return time.perf_counter()-started, json.loads(output.getvalue())


def main(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    old = _historical_tree()
    observations = []
    for label, tree_type in (("old", old), ("owner", OwnerPatchAABBTree),
                             ("owner", OwnerPatchAABBTree), ("old", old)):
        elapsed, result = _run_example(tree_type)
        observations.append({"search": label, "elapsed_seconds": elapsed,
                             "result": result})
    reference = observations[0]["result"]
    record = {
        "candidate": "c4_owner_patch_search_dev",
        "historical_tree_commit": BASE_COMMIT,
        "paired_order": [row["search"] for row in observations],
        "all_outputs_match_exactly": all(row["result"] == reference
                                         for row in observations),
        "observations": observations,
    }
    output_path.write_text(json.dumps(record, indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({"all_outputs_match_exactly":
                      record["all_outputs_match_exactly"],
                      "seconds": [round(row["elapsed_seconds"], 3)
                                  for row in observations]}, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: search_owner_probe.py OUTPUT")
    main(Path(sys.argv[1]))
