"""Experimental matching-point conservative buckling with Q4/S3 native members."""
from dataclasses import replace
import argparse
import json


def definition(shell='mixed', beams='mixed'):
    """Small eccentric panel; fibre branches are intentionally elastic interior."""
    import numpy as np
    from anyfesolver import b3_ge, stiffened_shell as ss
    if shell not in ('q4', 's3', 'mixed') or beams not in ('generalized', 'fibre', 'mixed'):
        raise ValueError('q4/s3/mixed shells and generalized/fibre/mixed beams required')
    e, nu = 210e9, .3
    g = e/(2*(1+nu))
    generalized = b3_ge.EllipsoidalGeneralizedSection(
        np.diag((e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)), np.eye(6), 1e20, 1e9)
    arm = np.sqrt(8.333333333333333e-6/.01)
    fibres = tuple(b3_ge.Fibre(str(i), y*arm, z*arm, .0025, e,
                              b3_ge.FlowCurve((0.,), (1e20,), 1e9))
                   for i, (y, z) in enumerate(((-1,-1), (-1,1), (1,-1), (1,1))))
    physical = b3_ge.PhysicalFibreSection(fibres, np.diag(np.sqrt(
        (0., g*.008, g*.008, g*1.6666666666666667e-5, 0., 0.))))
    positions = np.array([(x, -.5, .1) for x in (0., .5, 1., 1.5, 2.)])
    members = tuple(b3_ge.define_beam('b3-ge', i+1, (101+2*i, 102+2*i, 103+2*i),
        positions[2*i:2*i+3], np.tile(np.eye(3), (3,1,1)),
        generalized if beams == 'generalized' or (beams == 'mixed' and i == 0) else physical,
        np.eye(6)) for i in range(2))
    panel = ss.ShellPanel((1,2,3,4,5,6),
        ((0.,-.5,0.), (1.,-.5,0.), (2.,-.5,0.),
         (0.,.5,0.), (1.,.5,0.), (2.,.5,0.)),
        ((1,2,5,4), (2,3,6,5)), .07, e, nu, 7850.)
    if shell != 'q4':
        panel = replace(panel, q4_elements=() if shell == 's3' else ((1,2,5,4),),
            s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5)) if shell == 's3'
                        else ((2,3,6),(2,6,5)))
    model = ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1, panel, members,
        tuple(ss.PointConnection(a,b) for a,b in ((1,101),(2,103),(3,105))),
        tuple(ss.FixedCoordinate('shell', node, axis) for node in (1,4) for axis in range(6)))
    return model if shell == 's3' else replace(model, q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)


def workflow(shell='mixed', beams='mixed', count=2):
    from anyfesolver import stiffened_shell as ss
    owner = ss.create_analysis(definition=definition(shell, beams))
    try:
        accepted = owner.solve((ss.NodalAction('beam',105,(-1000.,0.,100.,0.,0.,0.)),),
            (.5,1.), max_iterations=24, max_backtracks=12)
        if accepted.status != 'completed':
            raise RuntimeError('accepted equilibrium did not complete: '+accepted.status)
        before = owner.checkpoint()
        result = owner.buckling(ss.GEMixedBucklingRequest(accepted.generation, count=count))
        if owner.checkpoint() != before:
            raise RuntimeError('buckling changed accepted checkpoint')
        return dict(status=result.status, policy=result.policy, production_qualified=False,
            shell=shell, beams=beams, prestress_multipliers=result.prestress_multipliers,
            equation_residuals=result.equation_residuals, constraint_residuals=result.constraint_residuals,
            accepted_checkpoint_unchanged=True, selection=result.selection,
            load_factor_interpretation=result.load_factor_interpretation, failure_reason=result.failure_reason)
    finally:
        owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell', choices=('q4','s3','mixed'), default='mixed')
    parser.add_argument('--beams', choices=('generalized','fibre','mixed'), default='mixed')
    parser.add_argument('--count', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(workflow(args.shell, args.beams, args.count), indent=2, allow_nan=False))
