"""Opt-in fine-Q1 Q4/Q4 supported-force bending through ordinary force control."""
from dataclasses import replace
from anyfesolver import solve_static_nonlinear
from anyfesolver.general_contact.automatic_domain import resolve_automatic_contact_domain
from examples.general_contact_q4_pair_bending import build as bending_build


def build(*, regularization=1000.):
    model,domain,settings,_resolution,load=bending_build()
    domain=replace(domain,defaults=replace(domain.defaults,normal_law='augmented',normal_stiffness_n_per_m3=regularization))
    # Inherited augmented admission/restart control; semismooth acceptance uses
    # the force residual C and the solve tolerance, not pressure_tolerance.
    settings=replace(settings,enforcement_solver='monolithic_active_set',complementarity_force_scale_n=1e-4,pressure_tolerance=1e-8)
    resolution=resolve_automatic_contact_domain(model,domain,settings,analysis_control='force')
    return model,domain,settings,resolution,load


def solve(model,settings,resolution,load,**kwargs):
    return solve_static_nonlinear(model,load,contact=resolution.definition,contact_settings=settings,
        tolerance=1e-12,num_steps=2,max_iterations=12,emit_restart_checkpoint=True,
        record_increment_snapshots=True,**kwargs)


if __name__=='__main__':
    import json
    model,domain,settings,resolution,load=build();result=solve(model,settings,resolution,load)
    print(json.dumps(dict(status=result.status,failure_reason=result.failure_reason,identity=resolution.identity,
        capability=result.info.get('contact_path'),contact=result.info.get('general_contact'),
        normal_forces=result.restart_checkpoint['path_state']['normal_forces'] if result.restart_checkpoint else None,
        displacements=result.displacements),default=lambda x:x.tolist() if hasattr(x,'tolist') else x))
    assert result.status=='completed',result.failure_reason
