"""Headless experimental elastic Q4/native physical-fibre BL5 workflow."""
from dataclasses import replace
import argparse
import json
import numpy as np
from anyfesolver import b3_ge, stiffened_shell as ss


def definition(family='biaxial', beams='mixed', basis='p0', yield_stress=1e20, shell='q4'):
    if family not in ('axial', 'biaxial') or beams not in ('fibre', 'mixed'):
        raise ValueError('axial/biaxial fibres and fibre/mixed member families required')
    if shell not in ('q4','s3','mixed'): raise ValueError('q4/s3/mixed shells required')
    e, nu = 210e9, .3; g = e/(2*(1+nu))
    stiffness = (e*.01, g*.008, g*.008, g*1.6666666666666667e-5,
                 e*8.333333333333333e-6, e*8.333333333333333e-6)
    generalized = b3_ge.EllipsoidalGeneralizedSection(np.diag(stiffness), np.eye(6), 1e20, 1e9)
    arm = np.sqrt(8.333333333333333e-6/.01)
    fibres = tuple(b3_ge.Fibre(str(i), y*arm if family == 'biaxial' else 0.,
        z*arm if family == 'biaxial' else 0., .0025, e,
        b3_ge.FlowCurve((0.,), (yield_stress,), 1e9))
        for i, (y, z) in enumerate(((-1,-1), (-1,1), (1,-1), (1,1))))
    background = np.array(stiffness); background[0] = 0.
    if family == 'biaxial': background[4:] = 0.
    physical = b3_ge.PhysicalFibreSection(fibres, np.diag(np.sqrt(background)))
    positions = np.array([(x, 0., .0675) for x in (0., .25, .5, .75, 1.)])
    members = tuple(b3_ge.define_beam('b3-ge', i+1, (101+2*i,102+2*i,103+2*i),
        positions[2*i:2*i+3], np.tile(np.eye(3), (3,1,1)),
        generalized if beams == 'mixed' and i == 0 else physical, np.eye(6)) for i in range(2))
    xs = np.linspace(0., 1., 4)
    panel = ss.ShellPanel(tuple(range(1,9)),
        tuple((float(x),0.,0.) for x in xs)+tuple((float(x),.2,0.) for x in xs),
        ((1,2,6,5),(2,3,7,6),(3,4,8,7)), .035, e, nu, 7850.)
    line = ss.ExperimentalBL5Line((1,2,3,4), ((0,0),(0,1),(1,0),(1,1)), basis=basis)
    model = ss.LinearDefinition(ss.B3_GE_E4_PL_STIFFENED_SHELL_V1, panel, members, (),
        tuple(ss.FixedCoordinate('shell',node,axis) for node in (1,5) for axis in range(6)),
        (line,), q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1,
        bl5_material_policy=ss.BL5_ELASTIC_NATIVE_FIBRE_V1)
    if shell != 'q4':
        cells = panel.q4_elements if shell == 's3' else panel.q4_elements[1:]
        panel = replace(panel,q4_elements=() if shell == 's3' else panel.q4_elements[:1],
            s3_elements=tuple(triangle for a,b,c,d in cells for triangle in ((a,b,c),(a,c,d))))
        model = replace(model,shell=panel,
            bl5_material_policy=ss.BL5_ELASTIC_NATIVE_FIBRE_MIXED_SHELL_V1,
            q4_pullback_policy='historical_corotational' if shell == 's3' else ss.Q4_ALGORITHMIC_PULLBACK_V1)
    return model


def workflow(family='biaxial', beams='mixed', basis='p0', count=2, shell='q4'):
    model = definition(family, beams, basis, shell=shell)
    owner = ss.create_analysis(definition=model); resumed = ss.create_analysis(definition=model)
    try:
        preflight = owner.preflight()
        actions = (ss.NodalAction('beam',105,(-1000.,0.,100.,0.,0.,0.)),)
        prefix = owner.solve(actions, (.5,), max_iterations=24, max_backtracks=12)
        if prefix.status != 'completed': raise RuntimeError('force prefix failed: '+prefix.status)
        raw = owner.checkpoint(); restored = resumed.restore(raw)
        if resumed.checkpoint() != raw: raise RuntimeError('fresh restore changed checkpoint')
        accepted = owner.solve(actions, (1.,), max_iterations=24, max_backtracks=12)
        continued = resumed.solve(actions, (1.,), max_iterations=24, max_backtracks=12)
        if accepted.status != 'completed' or continued.status != 'completed':
            raise RuntimeError('force continuation failed')
        if resumed.checkpoint() != owner.checkpoint(): raise RuntimeError('fresh continuation differs')
        recovery = resumed.recover(); before = resumed.checkpoint()
        spectrum = resumed.buckling(ss.GENativeLineBucklingRequest(continued.generation, count=count))
        if resumed.checkpoint() != before: raise RuntimeError('buckling changed accepted state')
        return dict(status=spectrum.status, policy=spectrum.policy, model_identity=preflight['identity'],
            production_qualified=False, family=family, beams=beams, basis=basis, shell=shell,
            generation=continued.generation, free_equilibrium_ratio=continued.free_equilibrium_ratio,
            constraint_max=continued.constraint_max, fibre_stations=len(recovery.beam_fibre_material),
            line_work_records=len(recovery.beam_shell_line_work), restored_generation=restored.generation,
            fresh_restore_bytes_equal=True, continued_bytes_equal=True, accepted_checkpoint_unchanged=True,
            prestress_multipliers=spectrum.prestress_multipliers,
            equation_residuals=spectrum.equation_residuals, constraint_residuals=spectrum.constraint_residuals,
            failure_reason=spectrum.failure_reason, selection=spectrum.selection)
    finally:
        owner.close(); resumed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('axial','biaxial'), default='biaxial')
    parser.add_argument('--beams', choices=('fibre','mixed'), default='mixed')
    parser.add_argument('--basis', choices=('p0','continuous_p1_coarse'), default='p0')
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--shell', choices=('q4','s3','mixed'), default='q4')
    args = parser.parse_args()
    print(json.dumps(workflow(args.family,args.beams,args.basis,args.count,args.shell), indent=2, allow_nan=False))
