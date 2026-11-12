"""Compare whole-face and projected certificates at one saved D1 pose."""

from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np

from general_contact_d1_coupled_refinement import build_case
from anyfesolver import validate_general_contact
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    model, definition, settings, _ = build_case(
        record["columns"], record["rows"], record["edge_control"],
    )
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    for node_id, values in record["final_nodal_displacements"].items():
        u[model.mesh.get_node(int(node_id)).dofs] = values
    cases = {}
    for name in ("whole_face", "projected_matching"):
        candidate_settings = replace(settings, q4_source_gap_certificate=name)
        capability = validate_general_contact(model, definition, candidate_settings)
        if not capability.valid:
            raise ValueError(f"{name} unsupported: {capability.issues}")
        manager = GeneralS3PatchManager(model, definition, candidate_settings)
        result = manager.evaluate(u, tangent=False)[2]
        cases[name] = {
            "profile": capability.profile,
            "minimum_gap_lower_m": result.minimum_gap,
            "sampled_overlap_gap_m": result.sampled_overlap_gap,
            "active_pressure_dofs": result.active_pressure_dofs,
            "normalized_weighted_gap_m": list(result.normalized_weighted_gap_m),
            "restart_certificate": manager.restart_contract()["settings"].get(
                "q4_source_gap_certificate", "whole_face",
            ),
        }
    output = {"input": str(args.input), "cases": cases}
    args.output.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in case.items()
                             if key != "normalized_weighted_gap_m"}
                      for name, case in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
