"""Bounded E04 development screen through the public curved-point owner.

Run under the task's existing serial resource supervisor. This is not capacity
adjudication. A numerical failure leaves outputs and the accepted checkpoint.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np


def _assert_material_equal(left,right):
    """Compare actual recovery, rather than reconstructing elastic stresses."""
    from dataclasses import fields
    np.testing.assert_array_equal(left.shell_layer_alpha,right.shell_layer_alpha)
    np.testing.assert_array_equal(left.beam_station_accumulated,right.beam_station_accumulated)
    assert len(left.shell_q4_material)==len(right.shell_q4_material)
    for a,b in zip(left.shell_q4_material,right.shell_q4_material):
        for name in ('layer_strain','layer_stress','plastic_strain','alpha'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert len(left.beam_generalized_material)==len(right.beam_generalized_material)
    for a,b in zip(left.beam_generalized_material,right.beam_generalized_material):
        assert a.element_id==b.element_id and len(a.stations)==len(b.stations)
        for x,y in zip(a.stations,b.stations):
            for field in fields(x):
                u,v=getattr(x,field.name),getattr(y,field.name)
                if isinstance(u,np.ndarray):np.testing.assert_array_equal(u,v)
                else:assert u==v


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=4)
    parser.add_argument('--density',type=float,required=True)
    parser.add_argument('--seed-force',type=float,default=10.)
    parser.add_argument('--radius',type=float,default=1e-5)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--imperfection-m',type=float,default=0.,
        help='Downward crown amplitude for a new virgin reference; metres.')
    parser.add_argument('--restart-after',type=int,
        help='Restore after this arc prefix and compare continued owners exactly.')
    parser.add_argument('--shell-proportional-limit-pa',type=float,
        help='Explicit synthetic shell J2 proportional stress; material mode needs all three strength options.')
    parser.add_argument('--beam-resultant-limits',
        help='Six positive declared associated-ellipsoid limits, N,N,N,Nm,Nm,Nm, comma separated.')
    parser.add_argument('--beam-hardening-n',type=float,
        help='Positive declared generalized hardening force, N; not fibre plasticity.')
    args=parser.parse_args()
    strengths=(args.shell_proportional_limit_pa,args.beam_resultant_limits,args.beam_hardening_n)
    material=any(value is not None for value in strengths)
    limits=None
    if material:
        try:
            limits=tuple(float(v) for v in args.beam_resultant_limits.split(','))
            if (any(v is None for v in strengths) or len(limits)!=6 or
                    not np.isfinite(limits).all() or min(limits)<=0. or
                    not np.isfinite(args.shell_proportional_limit_pa) or args.shell_proportional_limit_pa<=0. or
                    not np.isfinite(args.beam_hardening_n) or args.beam_hardening_n<=0.):
                raise ValueError('invalid strengths')
        except (ValueError,AttributeError):
            parser.error('material mode requires finite positive shell stress, six resultant limits and hardening force')
    if args.restart_after is not None and not 0<args.restart_after<args.steps:
        parser.error('--restart-after must be strictly between zero and --steps')
    args.output.mkdir(parents=True,exist_ok=False)
    from anyfesolver import stiffened_shell as ss
    from docs.reference_cases.stiffened_shell_e04_roof import (
        build,definition,material_definition,crown_reference_perturbations)
    import anyfesolver,threadpoolctl,numba
    pools=threadpoolctl.threadpool_info()
    assert pools and all(p['num_threads']==1 for p in pools)
    assert numba.get_num_threads()==1 and numba.config.DISABLE_JIT
    c=build(args.size,args.size);d=definition(c,density_kg_m3=args.density)
    if material:
        from anymaterial import DNVC208MaterialCurve
        p=args.shell_proportional_limit_pa
        curve=DNVC208MaterialCurve(p,1.05*p,1.1*p,.005,.01,4*p,.2)
        d=material_definition(c,density_kg_m3=args.density,shell_curve=curve,
            beam_yield_limits=limits,beam_hardening_n=args.beam_hardening_n)
    reference_change=None
    if args.imperfection_m:
        reference_change=ss.create_stress_free_point_reference(d,
            crown_reference_perturbations(c,args.imperfection_m))
        d=reference_change.definition
    node=c.loaded_shell_node+1;row=c.loaded_shell_node
    action=(ss.NodalAction('shell',node,(0.,0.,-args.seed_force,0.,0.,0.)),)
    record=dict(status='IN_PROGRESS',size=args.size,shells=len(c.quads),beams=len(c.beam_cells),
        density_kg_m3=args.density,density_use='static bookkeeping; spectra excluded',
        seed_force_n=args.seed_force,radius=args.radius,requested_arc_steps=args.steps,
        restart_after=args.restart_after,
        imperfection_m=args.imperfection_m,
        reference_transport=(None if reference_change is None else dict(
            policy=reference_change.policy,source_identity=reference_change.source_identity,
            target_identity=reference_change.target_identity,
            stress_free_scaled_max=reference_change.stress_free_scaled_max)),
        continuation_metric='fixed nominal reference area/length; equal domain weights',
        connection='six-channel nodal full-pose bonds',samples=[],timings={},
        imported_solver=str(Path(anyfesolver.__file__).resolve()),threadpools=pools)
    if material:
        record['material_definition']=dict(shell_curve=curve.as_dict(),
            native_resultant_limits=list(limits),native_resultant_units=['N']*3+['Nm']*3,
            beam_hardening_n=args.beam_hardening_n,
            retained_scale_policy=d.retained_scale_policy,
            strength_interpretation='declared synthetic development laws; not calibrated rectangle or roof strength')
    def save():
        (args.output/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    def sample(result,force,phase):
        import hashlib
        value=dict(phase=phase,downward_force_n=float(force),global_z_force_n=-float(force),
            central_translation_m=result.shell_displacement.reshape(-1,6)[row,:3].tolist(),
            equilibrium=float(result.free_equilibrium_ratio),constraint=float(result.constraint_max),
            checkpoint_sha256=hashlib.sha256(owner.checkpoint()).hexdigest())
        if phase=='arc':
            from dataclasses import asdict
            value['accepted_radius']=result.history[-1][0]
            value['amplitude']=result.amplitude
            value['attempts']=[asdict(a) for a in result.increment_attempts]
        if material:
            value['shell_alpha_max']=float(np.max(result.shell_layer_alpha,initial=0.))
            value['beam_alpha_max']=float(np.max(result.beam_station_accumulated,initial=0.))
        record['samples'].append(value);save();print(json.dumps(value),flush=True)
    owner=None
    try:
        t=time.monotonic();owner=ss.create_analysis(definition=d)
        record['timings']['compile']=time.monotonic()-t;record['preflight']=owner.preflight();save()
        t=time.monotonic();linear=owner.solve_linear(action)
        record['timings']['linear']=time.monotonic()-t
        record['linear']=dict(equilibrium=linear.free_equilibrium_ratio,constraint=linear.constraint_max,
            central_translation_m=linear.shell_displacement.reshape(-1,6)[row,:3].tolist())
        save()
        owner.close()
        t=time.monotonic();owner=ss.create_analysis(definition=d)
        record['timings']['finite_compile']=time.monotonic()-t
        controls=(dict(max_iterations=20,max_backtracks=16) if material else {})
        t=time.monotonic();seed=owner.solve(action,(1.,),**controls);record['timings']['seed']=time.monotonic()-t
        sample(seed,args.seed_force,'seed');(args.output/'seed.bin').write_bytes(owner.checkpoint())
        metric=ss.PhysicalChordMetric(c.metric_weights,c.length)
        expected=tuple(('shell',n) for n in d.shell.node_ids)+tuple(('beam',n) for n in range(100001,100001+len(c.beam_positions)))
        assert owner.preflight()['arc_metric_node_order']==expected
        control=ss.ArcLengthControl('shell',node,2,metric,args.radius,-1)
        def accepted(result):
            sample(result,args.seed_force*(1+result.amplitude),'arc')
            (args.output/'accepted.bin').write_bytes(owner.checkpoint())
        t=time.monotonic();arc=owner.continue_arc_length(action,control,
            steps=args.restart_after or args.steps,on_accepted=accepted)
        if args.restart_after is not None:
            import hashlib
            prefix=owner.checkpoint();(args.output/'restart-prefix.bin').write_bytes(prefix)
            continued=ss.create_analysis(definition=d)
            try:
                start=time.monotonic();at_prefix=continued.restore(prefix)
                assert continued.checkpoint()==prefix
                np.testing.assert_array_equal(at_prefix.cell_resultants,arc.cell_resultants)
                if material:_assert_material_equal(at_prefix,arc)
                record['timings']['prefix_restore']=time.monotonic()-start
                remaining=args.steps-args.restart_after
                start=time.monotonic();arc=owner.continue_arc_length(action,control,
                    steps=remaining,on_accepted=accepted)
                record['timings']['live_continuation']=time.monotonic()-start
                restarted_samples=[]
                def restarted_accepted(result):
                    restarted_samples.append(dict(amplitude=result.amplitude,
                        central_translation_m=result.shell_displacement.reshape(-1,6)[row,:3].tolist(),
                        checkpoint_sha256=hashlib.sha256(continued.checkpoint()).hexdigest()))
                start=time.monotonic();repeated=continued.continue_arc_length(action,control,
                    steps=remaining,on_accepted=restarted_accepted)
                record['timings']['restored_continuation']=time.monotonic()-start
                assert continued.checkpoint()==owner.checkpoint()
                np.testing.assert_array_equal(repeated.cell_resultants,arc.cell_resultants)
                np.testing.assert_array_equal(repeated.shell_displacement,arc.shell_displacement)
                assert repeated.history==arc.history
                if material:_assert_material_equal(repeated,arc)
                live_samples=record['samples'][-remaining:]
                assert [s['checkpoint_sha256'] for s in restarted_samples]==[
                    s['checkpoint_sha256'] for s in live_samples]
                record['turning_point_restart']=dict(prefix_steps=args.restart_after,
                    prefix_sha256=hashlib.sha256(prefix).hexdigest(),continued_steps=remaining,
                    final_bytes_equal=True,history_equal=True,each_checkpoint_equal=True,
                    samples=restarted_samples)
                (args.output/'restarted-final.bin').write_bytes(continued.checkpoint())
                save()
            finally:continued.close()
        record['timings']['arc']=time.monotonic()-t
        saved=owner.checkpoint()
        t=time.monotonic();restored=ss.create_analysis(definition=d)
        try:
            repeated=restored.restore(saved)
            assert restored.checkpoint()==saved
            np.testing.assert_array_equal(repeated.cell_resultants,arc.cell_resultants)
            if material:_assert_material_equal(repeated,arc)
        finally:restored.close()
        record['timings']['restore']=time.monotonic()-t
        record['restart_bytes_equal']=True;record['status']='COMPLETED_DEVELOPMENT_ONLY';save()
    except Exception as error:
        record['status']='FAILED';record['failure']=dict(type=type(error).__name__,message=str(error))
        if hasattr(error,'attempts'):
            from dataclasses import asdict
            record['failure']['attempts']=[asdict(a) for a in error.attempts]
        save();raise
    finally:
        if owner is not None:owner.close()


if __name__=='__main__':main()
