"""Experimental all-S3/two-stiffener J2/fibre signed displacement history.

Uses the existing coupled panel's unchanged low-strength laws and native point
bonds. Beam105 controls translation z; beam205 responds to the same load amplitude.
Reference forces define displacement targets, not prescribed endpoint forces.
This is a lifecycle demonstrator, not qualified engineering capacity or a repair
of failed fixed-step/force histories. Generalized member remains elastic.
"""
from dataclasses import replace
from pathlib import Path
import argparse,json,math
import numpy as np
from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken,SolveCancelled
from anyfesolver._stiffened_shell_q4_algorithmic_pullback import HISTORICAL_POLICY
if __package__:
    from .stiffened_shell_coupled_plastic_history import definition as base_definition,_shell_activity
else:
    from stiffened_shell_coupled_plastic_history import definition as base_definition,_shell_activity

OPTIONS=dict(max_iterations=20,max_backtracks=12)
LIMITS=ss.AdaptiveIncrementControl(6,24)
CONTROL=ss.DisplacementControl('beam',105,2)
DIRECTION=tuple(ss.NodalAction('beam',node,(0.,0.,150.,0.,0.,0.)) for node in (105,205))


def definition():
    base=base_definition()
    return replace(base,shell=replace(base.shell,q4_elements=(),
        s3_elements=((1,2,5),(1,5,4),(2,3,6),(2,6,5))),q4_pullback_policy=HISTORICAL_POLICY)


def activity(result,previous):
    _,alpha,families=_shell_activity(result,previous)
    increments=[math.fsum(f.plastic_increment) for r in result.beam_fibre_material for f in r.fibres]
    energies=[math.fsum(r.incremental_dissipation_j) for r in result.beam_fibre_energy]
    assert all(math.isfinite(x) and x>=0. for x in energies)
    fibre_active=any(abs(math.fsum(f.plastic_increment))>64.*np.finfo(float).eps*max(map(abs,f.plastic_increment))
                     for r in result.beam_fibre_material for f in r.fibres)
    return dict(target=result.target,amplitude=result.amplitude,shell_alpha_max=alpha,
        shell_active_by_family=families,fibre_active=fibre_active,
        simultaneous=all(families.values()) and fibre_active and math.fsum(energies)>0.,
        fibre_increment_min=min(increments),fibre_increment_max=max(increments),
        last_increment_fibre_dissipation_j=math.fsum(energies))


def workflow(output):
    output=Path(output);output.mkdir(exist_ok=False)
    model=definition();reference=ss.create_analysis(definition=model)
    analysis=ss.create_analysis(definition=model);resumed=ss.create_analysis(definition=model)
    try:
        actions=tuple(ss.NodalAction('beam',node,(0.,0.,1500.,0.,0.,0.)) for node in (105,205))
        linear=reference.solve_linear(actions)
        nominal=float(linear.beam_coordinates[6*reference._beam_index[105]+2])
        zero=tuple(ss.NodalAction('beam',node,(0.,0.,0.,0.,0.,0.)) for node in (105,205))
        seed=analysis.solve(zero,(1.,),**OPTIONS)
        kwargs=dict(OPTIONS,displacement_control=CONTROL,increment_control=LIMITS)
        loading=[];previous=seed
        def observe_loading(result):
            nonlocal previous
            assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
            loading.append(activity(result,previous));previous=result
        first=analysis.solve(DIRECTION,(nominal,),**kwargs,on_accepted=observe_loading)
        assert any(row['simultaneous'] for row in loading)
        (output/'loaded.checkpoint').write_bytes(analysis.checkpoint())
        targets=(nominal*.25,-nominal,nominal*1.1)
        token=CancellationToken();accepted=[];previous=first
        def observe(result):
            nonlocal previous
            assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
            accepted.append(activity(result,previous));previous=result
            token.cancel('accepted unload prefix')
        try:analysis.solve(DIRECTION,targets,**kwargs,cancellation_token=token,on_accepted=observe)
        except SolveCancelled:pass
        else:raise AssertionError('expected accepted-prefix cancellation')
        raw=analysis.checkpoint();(output/'unloaded.checkpoint').write_bytes(raw)
        restored=resumed.restore(raw);assert resumed.checkpoint()==raw
        assert restored.shell_layer_alpha==analysis.recover().shell_layer_alpha
        def remaining(result):
            nonlocal previous
            assert result.free_equilibrium_ratio<=1e-8 and result.constraint_max<=1e-9
            accepted.append(activity(result,previous));previous=result
        final=analysis.solve(DIRECTION,targets,**kwargs,on_accepted=remaining)
        repeated=resumed.solve(DIRECTION,targets,**kwargs)
        assert analysis.checkpoint()==resumed.checkpoint()
        assert final.shell_layer_alpha==repeated.shell_layer_alpha
        assert final.beam_fibre_energy==repeated.beam_fibre_energy
        assert final.status=='completed' and any(row['target']<0. and row['simultaneous'] for row in accepted)
        result=dict(status=final.status,production_qualified=False,mesh='all-S3',
            model_identity=analysis.preflight()['identity'],reference_nominal_target=nominal,
            exact_restart=True,loading_activity=loading,cyclic_activity=accepted,
            equilibrium=final.free_equilibrium_ratio,gap=final.constraint_max)
        (output/'completed.checkpoint').write_bytes(analysis.checkpoint())
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
    finally:reference.close();analysis.close();resumed.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(workflow(args.output)))
