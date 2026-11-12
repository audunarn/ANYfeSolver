"""G0 F02 negative control for point-torque versus shell drill completion."""

from __future__ import annotations

import json

import numpy as np

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material


def specimen(formulation: str, coordinates: tuple[tuple[float, ...], ...]) -> dict:
    mesh = FEMesh()
    for i, x in enumerate(coordinates, 1):
        mesh.add_node(i, *x)
    element = create_shell_element(1, list(range(1, len(coordinates) + 1)), "steel",
                                   formulation=formulation, thickness=.07,
                                   reference_normal=(0., 0., 1.))
    material = Material("steel", 210e9, .3, density=7850.)
    components = element.compute_stiffness_components(mesh, material)
    point = np.zeros(6 * len(coordinates))
    point[5] = 1.
    rigid = np.zeros_like(point)
    for i, (x, y, _) in enumerate(coordinates):
        rigid[6 * i:6 * i + 3] = (-y, x, 0.)
        rigid[6 * i + 5] = 1.
    def energy(v: np.ndarray) -> dict:
        return {key: float(v @ np.asarray(components[key]) @ v)
                for key in ("physical", "numerical", "total")}
    return {"formulation": formulation, "point_normal_rotation": energy(point),
            "rigid_in_plane_rotation": energy(rigid)}


def main() -> int:
    q4 = specimen("e4-pl", ((0., 0., 0.), (1., 0., 0.),
                              (1., 1., 0.), (0., 1., 0.)))
    s3 = specimen("e4-pl-s3-v2d", ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    result = {"case": "G0_F02_DRILL_NEGATIVE_CONTROL_V1", "Q4": q4, "S3-V2D": s3}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
