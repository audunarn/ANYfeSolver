"""Reopen a declared E04 material roof and continue an existing accepted path.

Uses installed ANYfeSolver/ANYfem public interfaces. Run serially under the task's
resource supervisor. Synthetic strengths do not establish physical capacity.
The source checkpoint must match the explicitly reconstructed definition.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

from r8_e04_native_roof import _assert_material_equal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--density',type=float,required=True)
    parser.add_argument('--seed-force',type=float,required=True)
    parser.add_argument('--radius',type=float,required=True)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--imperfection-m',type=float,default=0.,
        help='Downward crown amplitude of the new virgin reference; must match the checkpoint.')
    parser.add_argument('--shell-proportional-limit-pa',type=float,required=True)
    parser.add_argument('--beam-resultant-limits',required=True)
    parser.add_argument('--beam-hardening-n',type=float,required=True)
    args=parser.parse_args()
    if (args.steps<1 or not np.isfinite(args.seed_force) or args.seed_force<=0. or
            not np.isfinite(args.radius) or args.radius<=0. or
            not np.isfinite(args.shell_proportional_limit_pa) or args.shell_proportional_limit_pa<=0. or
            not np.isfinite(args.imperfection_m)):
        parser.error('positive finite load, radius and shell proportional stress; positive step count required')
    try:
        limits=tuple(float(v) for v in args.beam_resultant_limits.split(','))
    except ValueError:
        parser.error('six comma-separated beam resultant limits required')
    # Capture inputs before writing any output; the definition validates all strengths.
    checkpoint=args.checkpoint.read_bytes()
    from anymaterial import DNVC208MaterialCurve
    from anyfesolver import stiffened_shell as ss
    from anyfem.solve.stiffened_shell import ExperimentalGEPointPanel,save_case,load_case
    from anyfem.io.project_file import save_project,load_project
    from anyfem.model.project import Project
    from anyfem.model.records import AnalysisDefinition
    from docs.reference_cases.stiffened_shell_e04_roof import (
        build,material_definition,crown_reference_perturbations)
    import anyfesolver,anyfem,numba,threadpoolctl
    pools=threadpoolctl.threadpool_info()
    assert pools and all(p['num_threads']==1 for p in pools)
    assert numba.get_num_threads()==1 and numba.config.DISABLE_JIT
    case=build(args.size,args.size);p=args.shell_proportional_limit_pa
    curve=DNVC208MaterialCurve(p,1.05*p,1.1*p,.005,.01,4*p,.2)
    definition=material_definition(case,density_kg_m3=args.density,shell_curve=curve,
        beam_yield_limits=limits,beam_hardening_n=args.beam_hardening_n)
    reference_change=None
    if args.imperfection_m:
        reference_change=ss.create_stress_free_point_reference(definition,
            crown_reference_perturbations(case,args.imperfection_m))
        definition=reference_change.definition
    node=case.loaded_shell_node+1;row=case.loaded_shell_node
    action=(ss.NodalAction('shell',node,(0.,0.,-args.seed_force,0.,0.,0.)),)
    panel=ExperimentalGEPointPanel(definition.shell,definition.beams,definition.points,
        definition.fixed,action,q4_pullback_policy=definition.q4_pullback_policy,
        shell_reference_policy=definition.shell_reference_policy,
        retained_scale_policy=definition.retained_scale_policy)
    args.output.mkdir(parents=True,exist_ok=False)
    record=dict(status='IN_PROGRESS',input_checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        inputs={key:(str(value) if isinstance(value,Path) else value) for key,value in vars(args).items()},
        imported_solver=str(Path(anyfesolver.__file__).resolve()),
        imported_consumer=str(Path(anyfem.__file__).resolve()),threadpools=pools,
        reference_transport=(None if reference_change is None else dict(
            policy=reference_change.policy,source_identity=reference_change.source_identity,
            target_identity=reference_change.target_identity,
            stress_free_scaled_max=reference_change.stress_free_scaled_max)),
        interpretation='synthetic development laws; not calibrated roof strength',
        samples=[],timings={})
    def save():
        (args.output/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    save();owners=[]
    try:
        start=time.monotonic()
        payload=panel.to_dict()
        reopened=load_case(save_case(panel,args.output/'roof.anyfem-ge.json'))
        assert reopened.to_dict()==payload
        project=Project('Experimental E04 material roof',mesh_only=True)
        analysis=AnalysisDefinition.experimental_point_panel('material roof',reopened)
        project.add_analysis(analysis)
        stored=load_project(save_project(project,args.output/'roof.anyfem')).analyses[analysis.id]
        assert stored.beam_formulation_options['panel']==payload
        reference=ss.create_analysis(definition=definition);owners.append(reference)
        consumer=stored.create_point_panel_analysis();owners.append(consumer)
        assert reference.preflight()['identity']==consumer.preflight()['identity']
        assert consumer.preflight()['production_qualified'] is False
        record['identity']=reference.preflight()['identity'];record['schema']=payload['schema']
        before=reference.restore(checkpoint);recovered=consumer.restore(checkpoint)
        assert reference.checkpoint()==consumer.checkpoint()==checkpoint
        _assert_material_equal(before,recovered)
        np.testing.assert_array_equal(before.shell_displacement,recovered.shell_displacement)
        np.testing.assert_array_equal(before.cell_resultants,recovered.cell_resultants)
        assert before.history==recovered.history
        assert np.max(recovered.shell_layer_alpha)>0 and np.max(recovered.beam_station_accumulated)>0
        record['timings']['save_reopen_restore']=time.monotonic()-start;save()
        metric=ss.PhysicalChordMetric(case.metric_weights,case.length)
        control=ss.ArcLengthControl('shell',node,2,metric,args.radius,-1)
        expected=[]
        start=time.monotonic()
        def accepted_reference(result):
            expected.append(hashlib.sha256(reference.checkpoint()).hexdigest())
        final=reference.continue_arc_length(action,control,steps=args.steps,on_accepted=accepted_reference)
        record['timings']['reference_continuation']=time.monotonic()-start
        actual=[]
        start=time.monotonic()
        def accepted_consumer(result):
            blob=consumer.checkpoint();digest=hashlib.sha256(blob).hexdigest();actual.append(digest)
            (args.output/'accepted.bin').write_bytes(blob)
            record['samples'].append(dict(checkpoint_sha256=digest,
                downward_force_n=args.seed_force*(1+result.amplitude),
                central_translation_m=result.shell_displacement.reshape(-1,6)[row,:3].tolist(),
                equilibrium=result.free_equilibrium_ratio,constraint=result.constraint_max,
                shell_alpha_max=float(np.max(result.shell_layer_alpha)),
                beam_alpha_max=float(np.max(result.beam_station_accumulated))))
            save()
        repeated=consumer.continue_arc_length(action,control,steps=args.steps,on_accepted=accepted_consumer)
        record['timings']['consumer_continuation']=time.monotonic()-start
        assert actual==expected and len(actual)==args.steps
        assert reference.checkpoint()==consumer.checkpoint()
        assert final.history==repeated.history
        _assert_material_equal(final,repeated)
        np.testing.assert_array_equal(final.shell_displacement,repeated.shell_displacement)
        np.testing.assert_array_equal(final.cell_resultants,repeated.cell_resultants)
        record.update(status='COMPLETED_DEVELOPMENT_ONLY',each_checkpoint_equal=True,
            final_checkpoint_equal=True,detailed_material_equal=True,history_equal=True)
        imports={name:str(Path(module.__file__).resolve()) for name,module in __import__('sys').modules.items()
            if name.startswith(('anyfesolver','anyfem')) and getattr(module,'__file__',None)}
        record['imports']=imports;save()
    except Exception as error:
        record['status']='FAILED';record['failure']=dict(type=type(error).__name__,message=str(error))
        save();raise
    finally:
        for owner in reversed(owners):owner.close()


if __name__=='__main__':main()
