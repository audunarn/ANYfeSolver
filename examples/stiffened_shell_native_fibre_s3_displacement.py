"""Experimental mixed/all-S3 shells with native fibre plasticity and restart."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
from anyfesolver import stiffened_shell as ss
from stiffened_shell_ge_point_zero_unload import definition as reference_definition
from math import fsum


def definition(all_s3=False):
    base = reference_definition()
    q4 = () if all_s3 else ((1, 2, 5, 4),)
    s3 = ((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5)) if all_s3 else ((2, 3, 6), (2, 6, 5))
    shell = replace(base.shell, q4_elements=q4, s3_elements=s3)
    if all_s3:
        # No Q4 operator participates; retain the definition's default Q4 policy.
        return ss.LinearDefinition(base.profile, shell, base.beams, base.points, base.fixed)
    return replace(base, shell=shell)


def workflow(output, all_s3=False):
    output = Path(output)
    output.mkdir(exist_ok=False)
    model = definition(all_s3)
    analysis = ss.create_analysis(definition=model)
    resumed = ss.create_analysis(definition=model)
    try:
        actions = (ss.NodalAction('beam', 105, (0., 0., 1000., 0., 0., 0.)),)
        direction = (ss.NodalAction('beam', 105, (0., 0., 100., 0., 0., 0.)),)
        options = dict(max_iterations=20, max_backtracks=12)
        seed = analysis.solve(actions, (.5, 1.), **options)
        initial = float(seed.beam_positions[-1, 2] - model.beams[-1].instantiate()[0].operator.reference.coordinates[-1, 2])
        options['displacement_control'] = ss.DisplacementControl('beam', 105, 2)
        prefix = analysis.solve(direction, (initial*1.01,), **options)
        raw = analysis.checkpoint()
        (output/'prefix.checkpoint').write_bytes(raw)
        restored = resumed.restore(raw)
        assert resumed.checkpoint() == raw and restored.history == prefix.history
        final = analysis.solve(direction, (initial*1.02,), **options)
        continued = resumed.solve(direction, (initial*1.02,), **options)
        assert analysis.checkpoint() == resumed.checkpoint() and final.history == continued.history
        assert final.free_equilibrium_ratio <= 1e-8 and final.constraint_max <= 1e-9
        assert any(fsum(row.incremental_dissipation_j)>0. for row in final.beam_fibre_energy)
        result = dict(status=final.status, production_qualified=False,
                      fibre_last_increment_dissipation_j=[fsum(row.incremental_dissipation_j) for row in final.beam_fibre_energy],
                      mesh='all-S3' if all_s3 else 'mixed-Q4-S3', exact_restart=True,
                      equilibrium=final.free_equilibrium_ratio, gap=final.constraint_max,
                      history=final.history)
        (output/'completed.checkpoint').write_bytes(analysis.checkpoint())
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        return result
    finally:
        analysis.close()
        resumed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--all-s3', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(workflow(arguments.output, arguments.all_s3)))
