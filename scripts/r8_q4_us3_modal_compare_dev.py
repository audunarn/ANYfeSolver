"""Matched-grid physical Q4/US3 modal comparison; no convergence qualification.

Native massless spin remains in sparse equilibrium. Both nodal mode sets are
reconstructed through the same physical Q4 mass for MAC; that reconstruction
is not exact integration of the US3 piecewise-linear modal field.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, eigs

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def common_mass_mac(native, reference, mass):
    native, reference = np.asarray(native), np.asarray(reference)
    if (native.ndim != 2 or reference.ndim != 2 or native.shape[0] != reference.shape[0]
            or mass.shape != (native.shape[0], native.shape[0]) or
            not np.isfinite(native).all() or not np.isfinite(reference).all()):
        raise ValueError('finite mode columns and common physical mass required')
    def normalized(columns):
        norm = np.sum(columns*(mass@columns), axis=0)
        if not np.isfinite(norm).all() or np.any(norm <= 0.):
            raise ValueError('positive representable common-mass mode norms required')
        return columns/np.sqrt(norm)[None, :]
    left, right = normalized(native), normalized(reference)
    mac = (left.T@(mass@right))**2
    if not np.isfinite(mac).all() or np.any(mac > 1.+1e-12):
        raise ValueError('invalid common-mass correlation')
    return mac, left, right


def run(reference: Path, manifest: Path, output: Path, site: Path):
    import anyfesolver, numba, threadpoolctl
    from anyfesolver import stiffened_shell as api
    from anyfesolver._stiffened_shell_ge_experimental import _shell_reference, _assemble_shell
    from anyfesolver._stiffened_shell_b2_q4_panel_experimental import _modal_scaled_residual
    from anyfesolver._stiffened_shell_q4_algorithmic_pullback import POLICY as Q4_ALGORITHM
    from anyfesolver._stiffened_shell_physical_q4_mass import physical_mass
    from anyfesolver.linalg import MatrixClass, factorize_cached
    from anyfesolver.e4_pl_element import FORMULATION_ID, IMPLEMENTATION_ID

    if not Path(anyfesolver.__file__).resolve().is_relative_to(site.resolve()):
        raise ValueError('selected installed solver site required')
    if not numba.config.DISABLE_JIT or numba.get_num_threads() != 1:
        raise ValueError('declared one-thread JIT-off settings required')
    pools = threadpoolctl.threadpool_info()
    if not pools or any(pool['num_threads'] != 1 for pool in pools):
        raise ValueError('effective numerical pools must use one thread')
    module_manifest = site.parent.parent/'solver-source-manifest.json'
    module_hashes = json.loads(module_manifest.read_text())
    if any(sha(site/name.removeprefix('src/')) != digest for name,digest in module_hashes.items()):
        raise ValueError('selected installed solver modules changed')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output/'executed-runner.py')
    inventory = json.loads(manifest.read_text())['member_sha256']
    inputs = ('identity.json', 'result.json', 'plate-modes.inp', 'reference-modes.npz')
    for name in inputs:
        if sha(reference/name) != inventory[name]:
            raise ValueError('saved external input changed: '+name)
    identity = json.loads((reference/'identity.json').read_text())
    nx, ny = identity['nx'], identity['ny']
    if nx not in (16, 32, 64) or ny != nx//4:
        raise ValueError('declared fixed-aspect physical grid family required')
    node_count = (nx+1)*(ny+1)
    total = 6*node_count
    required = dict(nodes=node_count, elements=2*nx*ny, E_Pa=210.e9,
                    nu=.3, thickness_m=.035, density_kg_m3=7850.,
                    beam_count=0, attachment_count=0, dimensions_m=[1., .2])
    if any(identity[key] != value for key,value in required.items()):
        raise ValueError('exact declared common plate required')
    start = time.perf_counter()
    ids = tuple(range(1, node_count+1))
    xyz = np.array([(i/nx, .2*j/ny, 0.) for j in range(ny+1) for i in range(nx+1)])
    roots = [ids[i] for i,p in enumerate(xyz) if p[0] == 0.]
    if list(ids) != identity['node_ids'] or roots != identity['roots']:
        raise ValueError('external/native node and support numbering differ')
    deck = (reference/'plate-modes.inp').read_text().splitlines()
    begin, end = deck.index('*NODE,NSET=AllNodes')+1, deck.index('*ELEMENT,TYPE=US3,ELSET=Panel')
    decoded = [row.split(',') for row in deck[begin:end]]
    if ([int(row[0]) for row in decoded] != list(ids) or
            not np.array_equal(np.array([[float(v) for v in row[1:]] for row in decoded]),xyz) or
            deck[deck.index('*BOUNDARY')+1:deck.index('*STEP')] != [f'{n},1,6,0.' for n in roots]):
        raise ValueError('actual external deck geometry/supports differ')
    cells = []
    for j in range(ny):
        for i in range(nx):
            a = 1+j*(nx+1)+i
            cells.append((a, a+1, a+nx+2, a+nx+1))
    panel = api.ShellPanel(ids, tuple(map(tuple, xyz)), tuple(cells), identity['thickness_m'],
                           identity['E_Pa'], identity['nu'], identity['density_kg_m3'])
    model, compiled_ids, compiled_xyz, index, pieces = _shell_reference(panel)
    if compiled_ids != ids or not np.array_equal(compiled_xyz, xyz):
        raise ValueError('compiled physical coordinates changed')
    rows, cols, stiffness, mass = [], [], [], []
    symmetry = []
    for eid, element, slots, origin in pieces:
        local = element.compute_stiffness_matrix(model.mesh, model.get_material('shell'))
        vertices = xyz[[index[n] for n in element.node_ids]]
        inertia = physical_mass(vertices, density_kg_m3=identity['density_kg_m3'],
                                thickness_m=identity['thickness_m'], reference_normal=(0., 0., 1.))
        if local.shape != (24, 24) or not np.isfinite(local).all():
            raise ValueError('complete finite reference Q4 stiffness required')
        symmetry.append(float(np.linalg.norm(local-local.T)/np.linalg.norm(local)))
        rows.append(np.repeat(slots, 24)); cols.append(np.tile(slots, 24))
        stiffness.append(local.ravel()); mass.append(inertia.ravel())
    row, col = np.concatenate(rows), np.concatenate(cols)
    k = sparse.coo_matrix((np.concatenate(stiffness), (row,col)), shape=(total,total)).tocsc()
    m = sparse.coo_matrix((np.concatenate(mass), (row,col)), shape=(total,total)).tocsc()
    k.eliminate_zeros(); m.eliminate_zeros()
    np.savez_compressed(output/'layout.npz', ids=ids, xyz=xyz, cells=cells)
    sparse.save_npz(output/'stiffness.npz', k); sparse.save_npz(output/'mass.npz', m)

    # Bounded provenance controls: actual zero-state GE paths, same local cell.
    first = pieces[0]
    local = np.asarray(stiffness[0]).reshape(24,24)
    route_parity, reference_residuals, local_controls = {}, {}, {'direct':local}
    for policy in ('historical', 'algorithmic'):
        options = {} if policy == 'historical' else {'q4_pullback_policy': Q4_ALGORITHM}
        force, assembled, reference_error = _assemble_shell(model, [first], total, np.ones(total), **options)
        slots = first[2]
        pulled = assembled[slots][:,slots].toarray()
        route_parity[policy] = float(np.linalg.norm(pulled-local)/np.linalg.norm(local))
        reference_residuals[policy] = reference_error
        local_controls[policy] = pulled
        np.savez_compressed(output/'local-route-controls.npz', **local_controls)
        print(json.dumps(dict(stage='local virgin route control', policy=policy,
                              tangent_difference=route_parity[policy], reference_residual=reference_error)),flush=True)
        if reference_error > 1e-9 or route_parity[policy] > 1e-10:
            raise ValueError('virgin GE/direct Q4 provenance parity unresolved')
    vertices = xyz[[index[n] for n in first[1].node_ids]]
    centre = vertices.mean(axis=0)
    rigid = np.zeros((24,6))
    for i, p in enumerate(vertices):
        rigid[6*i:6*i+3,:3] = np.eye(3)
        for axis in range(3):
            rigid[6*i:6*i+3,3+axis] = np.cross(np.eye(3)[axis], p-centre)
            rigid[6*i+3+axis,3+axis] = 1.
    rigid_checks = [float(_modal_scaled_residual(local@rigid[:,j], abs(local)@abs(rigid[:,j]),
                                                 1./np.sqrt(local.diagonal()))) for j in range(6)]
    expected_translation = identity['density_kg_m3']*identity['thickness_m']*.2
    expected_rotation = identity['density_kg_m3']*identity['thickness_m']**3*.2/12
    kinetic = []
    for component in range(6):
        field = np.zeros(total); field[component::6] = 1.
        kinetic.append(float(field@m@field))
    wanted = np.array([expected_translation]*3+[expected_rotation]*2+[0.])
    kinetic_error = float(np.max(abs(np.asarray(kinetic[:5])-wanted[:5])/wanted[:5]))
    if max(symmetry) > 1e-9 or max(rigid_checks) > 1e-9 or kinetic_error > 1e-12 or kinetic[5] != 0.:
        raise ValueError('reference operator rigidity/symmetry/physical inertia control failed')
    spin = np.arange(5,total,6)
    if m[spin].nnz or m[:,spin].nnz:
        raise ValueError('normal spin must have zero physical inertia')
    free = np.array([6*i+d for i,node in enumerate(ids) if node not in roots for d in range(6)])
    reduced_k, reduced_m = k[free][:,free], m[free][:,free]
    sigma = -(2*np.pi*.5)**2
    shifted = reduced_k-sigma*reduced_m
    if np.any(shifted.diagonal() <= 0.) or not np.isfinite(shifted.data).all():
        raise ValueError('representable positive shifted diagonal required for congruence')
    scale = 1./np.sqrt(shifted.diagonal())
    d = sparse.diags(scale, format='csc')
    factor = factorize_cached((d@shifted@d).tocsc(), MatrixClass.GENERAL)
    if factor.status != 'ok':
        raise RuntimeError('native sparse factorization failed: '+str(factor.failure_reason))
    try:
        scaled_mass = (d@reduced_m@d).tocsc()
        operator = LinearOperator(reduced_k.shape, matvec=lambda x: factor.solve(scaled_mass@x), dtype=float)
        values, vectors = eigs(operator, k=8, which='LM', tol=1e-11, maxiter=1000,
                              v0=np.sin(np.arange(len(free))+.23))
        recovered = []
        for mu, y in zip(values, vectors.T):
            if (not np.isfinite(mu) or mu.real <= 0. or abs(mu.imag) > 1e-9*abs(mu.real) or
                    not np.isfinite(y).all() or np.linalg.norm(y.imag) > 1e-9*np.linalg.norm(y.real)):
                raise ValueError('unresolved nonphysical native modal pair')
            lam = sigma+1./mu.real
            v = scale*y.real
            norm = float(v@reduced_m@v)
            if not np.isfinite(lam) or lam <= 0. or not np.isfinite(norm) or norm <= 0.:
                raise ValueError('positive finite native eigenvalue and physical norm required')
            v /= np.sqrt(norm)
            residual = reduced_k@v-lam*(reduced_m@v)
            bound = abs(reduced_k)@abs(v)+abs(lam)*(abs(reduced_m)@abs(v))
            error = float(_modal_scaled_residual(residual, bound, scale))
            if error > 1e-8:
                raise ValueError('native equation residual fails unchanged threshold')
            physical = np.zeros(total); physical[free] = v
            recovered.append((float(np.sqrt(lam)/(2*np.pi)), physical, error))
        recovered.sort(key=lambda row: row[0])
        counts = {'factorizations':factor.factorization_count,'solves':factor.solve_count}
    finally:
        # The shared service has no close method. This uncached local handle
        # owns its factor; releasing the reference follows the normal service
        # lifetime, including native-backend finalization when applicable.
        del factor
    frequencies = np.array([row[0] for row in recovered])
    native = np.column_stack([row[1] for row in recovered])
    if np.max(abs(native.T@m@native-np.eye(8))) > 1e-7:
        raise ValueError('native physical-mass orthogonality fails existing control')
    with np.load(reference/'reference-modes.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['node_ids'], ids): raise ValueError('reference field ordering differs')
        reference_modes = saved['modes'].reshape(6,total).T.copy()
        reference_frequencies = saved['frequencies_hz'].copy()
    mac, normalized_native, normalized_reference = common_mass_mac(native, reference_modes, m)
    ni, ri = linear_sum_assignment(-mac)
    pairs = []
    for i,j in sorted(zip(ni,ri), key=lambda pair:pair[1]):
        sign = 1. if normalized_native[:,i]@(m@normalized_reference[:,j]) >= 0. else -1.
        pairs.append(dict(reference_mode=int(j+1), native_mode=int(i+1),
            reference_frequency_hz=float(reference_frequencies[j]), native_frequency_hz=float(frequencies[i]),
            signed_frequency_difference_percent=float(100*(frequencies[i]/reference_frequencies[j]-1)),
            common_q4_mass_mac=float(mac[i,j]),
            ignored_massless_spin_difference_norm=float(np.linalg.norm(normalized_native[spin,i]-sign*normalized_reference[spin,j]))))
    np.savez_compressed(output/'native-modes.npz', frequencies_hz=frequencies, modes=native,
                        reference_common_normalization=normalized_reference, common_mass_mac=mac, free=free)
    result = dict(status='COMPLETED_SINGLE_GRID_DEVELOPMENT_ONLY', qualification=False,
        source_formulation=FORMULATION_ID, source_implementation=IMPLEMENTATION_ID,
        mass_policy=api.Q4_DIRECTOR_MASS_POLICY, nx=nx, ny=ny, nodes=node_count, elements=nx*ny, free_rows=len(free),
        massless_spin_rows=len([i for i in free if i%6 == 5]), native_modes=8, reference_modes=6,
        route_parity=route_parity, reference_residuals=reference_residuals,
        rigid_residuals=rigid_checks, mass_uniform_fields=kinetic,
        mass_uniform_field_relative_error=kinetic_error, maximum_local_symmetry_error=max(symmetry),
        native_equation_residuals=[row[2] for row in recovered], pairs=pairs, counts=counts,
        reference_eigen_residual='NOT_VERIFIED', refinement_convergence='NOT_ASSESSED',
        metric='common physical Q4 nodal reconstruction; excludes massless rz; scalar MAC not clustered-subspace acceptance',
        runtime=dict(anyfesolver=anyfesolver.__file__, pools=pools, jit_disabled=bool(numba.config.DISABLE_JIT),
                     numba_threads=numba.get_num_threads(), installed_modules=len(module_hashes),
                     module_manifest_sha256=sha(module_manifest),
                     dependencies={name:metadata.version(name) for name in ('ANYfeSolver','ANYmaterial','numpy','scipy','numba')}),
        elapsed_seconds=time.perf_counter()-start, runner_sha256=sha(Path(__file__)),
        protected_reference_inputs={name:inventory[name] for name in inputs})
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','manifest','output','site'): p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    run(args.reference, args.manifest, args.output, args.site)
