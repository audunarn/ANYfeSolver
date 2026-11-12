"""One supervised development case, not a qualification campaign or canary."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time


from examples.stiffened_shell_ge_mixed_buckling import definition


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('compression','tension'), default='compression')
    parser.add_argument('--shell', choices=('q4','s3','mixed'), default='mixed')
    parser.add_argument('--beams', choices=('generalized','fibre','mixed'), default='mixed')
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='IN_PROGRESS', case=args.case, shell=args.shell, beams=args.beams, qualified=False,
        python=sys.executable, python_version=sys.version,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_DISABLE_JIT')})
    owner = None
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    start = time.monotonic()
    try:
        if any(value != '1' for value in status['environment'].values()):
            raise RuntimeError('one-thread/JIT-disabled supervised environment required')
        import numpy as np
        import scipy
        import numba
        import anyfesolver
        from threadpoolctl import threadpool_info
        from scipy import sparse
        from anyfesolver import stiffened_shell as public
        from anyfesolver._stiffened_shell_ge_point_buckling import capture
        status['imports'] = {module.__name__: module.__file__ for module in (np, scipy, numba, anyfesolver)}
        status['threadpools'] = threadpool_info()
        status['numba_disable_jit_effective'] = bool(numba.config.DISABLE_JIT)
        if not status['numba_disable_jit_effective'] or any(row['num_threads'] != 1 for row in status['threadpools']):
            raise RuntimeError('effective numerical runtime differs from required supervision')
        owner = public.create_analysis(definition=definition(args.shell, args.beams))
        sign = -1. if args.case == 'compression' else 1.
        accepted = owner.solve((public.NodalAction('beam',105,(sign*1000.,0.,100.,0.,0.,0.)),),
            (.5,1.), max_iterations=24, max_backtracks=12)
        if accepted.status != 'completed':
            raise RuntimeError('accepted equilibrium did not complete: '+accepted.status)
        raw = owner.checkpoint()
        (args.output/'accepted.checkpoint').write_bytes(raw)
        status.update(model_identity=owner._identity, generation=accepted.generation,
            checkpoint_sha256=sha256(raw).hexdigest(), accepted_equilibrium_ratio=accepted.free_equilibrium_ratio,
            accepted_constraint_max=accepted.constraint_max)
        packet, guard = capture(owner, accepted.generation, mixed=True)
        for name, matrix in (('material',packet.material),('geometric',packet.geometric),('hessian',packet.hessian),
            ('constraints',packet.constraints),('beam-geometric',packet.beam_geometric),
            ('shell-geometric',packet.shell_geometric),('multiplier-curvature',packet.multiplier_curvature)):
            sparse.save_npz(args.output/(name+'.npz'), matrix)
        save('snapshot.json', dict(q=packet.snapshot.q.tolist(), frames=packet.snapshot.frames.tolist(),
            cells=packet.snapshot.cells.tolist(), resultants=packet.snapshot.resultants.tolist(),
            multipliers=packet.snapshot.multipliers.tolist(), applied_dead_load=packet.snapshot.applied_dead_load.tolist(),
            free=packet.free, fixed=packet.fixed))
        result = owner.buckling(public.GEMixedBucklingRequest(accepted.generation, count=args.count))
        save('buckling.json', dict(status=result.status, policy=result.policy, prestress_multipliers=result.prestress_multipliers,
            equation_residuals=result.equation_residuals, constraint_residuals=result.constraint_residuals,
            full_coordinates=[mode.tolist() for mode in result.full_coordinates],
            shell_coordinate_modes=[mode.tolist() for mode in result.shell_coordinate_modes],
            shell_spatial_modes=[mode.tolist() for mode in result.shell_spatial_modes],
            resultant_modes=[mode.tolist() for mode in result.resultant_modes],
            multiplier_modes=[mode.tolist() for mode in result.multiplier_modes],
            coordinate_support_reactions=[mode.tolist() for mode in result.coordinate_support_reactions],
            physical_support_wrenches=[mode.tolist() for mode in result.physical_support_wrenches],
            failure_reason=result.failure_reason, selection=result.selection,
            load_factor_interpretation=result.load_factor_interpretation))
        guard()
        if owner.checkpoint() != raw:
            raise RuntimeError('buckling changed accepted checkpoint')
        status.update(status='COMPLETED_DEVELOPMENT' if result.status == 'completed' else 'NUMERICAL_NON_SUCCESS',
            result_status=result.status, failure_reason=result.failure_reason, decomposition_error=result.decomposition_error,
            symmetry_error=result.symmetry_error, factorization_count=result.factorization_count, solve_count=result.solve_count)
    except BaseException as exc:
        status.update(status='FAILED_EXECUTION', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic()-start
        save('status.json', status)
        if owner is not None:
            owner.close()


if __name__ == '__main__':
    main()

