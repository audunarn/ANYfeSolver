"""Exact binary axis-aligned full-line first touch; explicit V19 development profile."""
from examples.general_contact_beam_s3_transient_rotational import build as rotational_build
from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
from anyfesolver.dynamics import TransientConfig


def build():
    model,definition,settings,velocity=rotational_build()
    for node in (1,6,2):model.mesh.nodes[node].z=.02
    model.mesh.bump_revision('geometry')
    return model,definition,settings,velocity


def run(*,dt=1.e-5,end=2.e-5,restart=None):
    m,d,s,v=build()
    result=solve_transient_contact(m,d,TransientConfig(dt,end,initial_velocity=v),
        contact_settings=s,solver_controls=ContactTransientSolverControls(
            allow_directional_beam_s3_dev=True,save_constraint_reactions_dev=True),
        emit_restart_checkpoint=True,restart_checkpoint=restart)
    return m,result


if __name__=='__main__':
    _,result=run(end=.0008)
    print(dict(profile=result.profile,states=len(result.times),active=sum(x.active_line_length>0 for x in result.contact),final_gap=result.contact[-1].minimum_gap,version=result.restart_checkpoint['version']))
    assert any(x.active_line_length>0 for x in result.contact)
    assert result.contact[-1].active_line_length==0 and result.contact[-1].minimum_gap>0
