"""Unregistered classical membrane operator comparison; no F03 acceptance.

This tiny independent plane-stress Q4/S3 implementation probes whether a
resolved, separately assembled membrane reference can share the F03 fixture's
in-plane idealization. It does not implement the mixed line attachment.
"""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material
from _b3_e4_f03_shell_trace import make_shell


def continuum_stiffness(points, topology, young=210e9, poisson=.3, thickness=.07):
    points = np.asarray(points, dtype=float)
    d = young / (1 - poisson**2) * np.array(
        [[1, poisson, 0], [poisson, 1, 0], [0, 0, (1-poisson)/2]])
    result = np.zeros((2*len(points), 2*len(points)))
    if topology == 'Q4':
        a = 1/np.sqrt(3)
        stations = ((r, s, 1.) for r in (-a, a) for s in (-a, a))
    else:
        stations = ((None, None, 1.),)
    for r, s, weight in stations:
        if topology == 'Q4':
            natural = .25*np.array(((-1+s, -1+r),
                                     (1-s, -1-r),
                                     (1+s, 1+r),
                                     (-1-s, 1-r)))
            jacobian = points.T @ natural
            gradient = natural @ np.linalg.inv(jacobian)
            measure = np.linalg.det(jacobian)
        else:
            affine = np.column_stack((np.ones(3), points))
            gradient = np.linalg.inv(affine)[1:].T
            measure = abs(np.linalg.det(affine))/2
        b = np.zeros((3, 2*len(points)))
        b[0, 0::2] = gradient[:, 0]
        b[1, 1::2] = gradient[:, 1]
        b[2, 0::2] = gradient[:, 1]
        b[2, 1::2] = gradient[:, 0]
        result += weight * measure * thickness * b.T @ d @ b
    return result


def compare(topology, points):
    mesh = FEMesh()
    for i, (x, y) in enumerate(points, 1):
        mesh.add_node(i, x, y, 0.)
    material = Material('steel', 210e9, .3, density=7850.)
    element = create_shell_element(
        1, list(range(1, len(points)+1)), 'steel',
        formulation='e4-pl' if topology == 'Q4' else 'e4-pl-s3-v2d',
        thickness=.07, reference_normal=(0., 0., 1.))
    physical = element.compute_stiffness_components(mesh, material)['physical']
    dofs = [6*i + axis for i in range(len(points)) for axis in (0, 1)]
    actual = physical[np.ix_(dofs, dofs)]
    independent = continuum_stiffness(points, topology)
    points = np.asarray(points, dtype=float)
    affine_energy_errors = []
    for strain in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)):
        ex, ey, gamma = strain
        displacement = np.column_stack((ex*points[:,0]+gamma*points[:,1],
                                        ey*points[:,1])).ravel()
        reference = displacement @ independent @ displacement
        affine_energy_errors.append(float(abs(displacement @ actual @ displacement
                                               - reference)/reference))
    # A single linear triangle interpolates any three nodal values as an
    # affine field; this diagnostic is meaningful only for the Q4 patch.
    nonaffine_ratio = None
    if topology == 'Q4':
        nonaffine = np.column_stack((points[:,0]*points[:,1],
                                     points[:,0]*points[:,1])).ravel()
        nonaffine_reference = nonaffine @ independent @ nonaffine
        nonaffine_ratio = float((nonaffine @ actual @ nonaffine)/nonaffine_reference)
    return dict(topology=topology, max_absolute_n_per_m=float(np.max(np.abs(actual-independent))),
                relative_frobenius=float(np.linalg.norm(actual-independent)/np.linalg.norm(independent)),
                max_affine_energy_relative=max(affine_energy_errors),
                nonaffine_energy_ratio=nonaffine_ratio,
                actual_norm=float(np.linalg.norm(actual)), independent_norm=float(np.linalg.norm(independent)))


def compare_panel(topology, n_s):
    """Separately assembled membrane panel and a fixed nodal-load diagnostic."""
    original = make_shell(n_s, topology)
    nx = 2*n_s
    x = np.linspace(-.5, 1.5, nx+1)
    y = (-.5, 0., .5)
    nodes = np.array([(xi, yi) for yi in y for xi in x])
    rows, columns, entries = [], [], []
    for j in range(2):
        for i in range(nx):
            a = j*(nx+1)+i
            b, d = a+1, a+nx+1
            c = d+1
            patches = ((a,b,c,d),) if topology == 'Q4' else ((a,b,c),(a,c,d))
            for patch in patches:
                local = continuum_stiffness(nodes[list(patch)], topology)
                dofs = [2*node+axis for node in patch for axis in (0,1)]
                for lr, gr in enumerate(dofs):
                    for lc, gc in enumerate(dofs):
                        rows.append(gr)
                        columns.append(gc)
                        entries.append(local[lr,lc])
    independent = sparse.coo_matrix((entries, (rows, columns)),
                                    shape=original.stiffness.shape).tocsr()
    free = original.free
    actual = original.stiffness[free, :][:, free]
    separate = independent[free, :][:, free]
    load = np.zeros(original.dof_count)
    load[2*((nx+1)+n_s)+1] = 1.
    exact_response = spsolve(actual.tocsc(), load[free])
    separate_response = spsolve(separate.tocsc(), load[free])
    return dict(topology=topology, n_s=n_s,
                relative_free_frobenius=float(sparse.linalg.norm(actual-separate)/sparse.linalg.norm(separate)),
                loaded_compliance_ratio=float(load[free]@exact_response/(load[free]@separate_response)))


if __name__ == '__main__':
    fixtures = (
        ('Q4', ((0., 0.), (1., 0.), (1., .5), (0., .5))),
        ('S3-V2D', ((0., 0.), (1., 0.), (1., .5))),
    )
    for topology, points in fixtures:
        print(json.dumps(compare(topology, points), sort_keys=True, allow_nan=False))
    for topology in ('Q4', 'S3-V2D'):
        for n_s in (1,2,4):
            print(json.dumps(compare_panel(topology, n_s), sort_keys=True, allow_nan=False))
