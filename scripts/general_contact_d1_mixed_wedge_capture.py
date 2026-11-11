"""Capture one exact Q4/S3 slide-branch value mismatch for local diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.s3_patch_augmented import (
    MovingS3PatchProjectedNormal,
)
from scripts.general_contact_d1_mixed_mesh_jet2_schur_press import main


def capture(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    original = MovingS3PatchProjectedNormal._jet2_s3_work
    captured = False

    def inspected(self, current_source, current_target, base, *, tangent,
                  tiny_wedge_pair=None):
        nonlocal captured
        try:
            return original(self, current_source, current_target, base,
                            tangent=tangent, tiny_wedge_pair=tiny_wedge_pair)
        except ValueError as exc:
            if captured or "differentiated S3 overlap disagrees" not in str(exc):
                raise
            captured = True
            pair = int(str(exc).split("pair=", 1)[1].split(",", 1)[0])
            target_count = len(self.target_face_indices)
            source_face, target_face = divmod(pair, target_count)
            source_nodes = self.source_face_indices[source_face]
            target_nodes = self.target_face_indices[target_face]
            np.savez(
                output,
                source=np.asarray(current_source),
                target=np.asarray(current_target),
                local_source=np.asarray(current_source)[source_nodes],
                local_target=np.asarray(current_target)[target_nodes],
                source_reference=(self.geometry.source_faces[source_face]
                                  .reference_positions),
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                pressure=np.asarray(base.pressure),
                reference_support=np.asarray(base.reference_support),
                reference_weighted_gap=np.asarray(base.reference_weighted_gap),
                pair_index=pair,
                tangent_requested=tangent,
                message=str(exc),
            )
            raise

    MovingS3PatchProjectedNormal._jet2_s3_work = inspected
    try:
        try:
            main(8, output.with_suffix(".unused.json"), 100.0, True)
        except ValueError as exc:
            if not captured:
                raise
            print(f"captured={output}; failure={str(exc).split(' (', 1)[0]}")
    finally:
        MovingS3PatchProjectedNormal._jet2_s3_work = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    capture(parser.parse_args().output)
