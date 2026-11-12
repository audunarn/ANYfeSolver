"""Bounded physical-side oriented automatic S3 pair static lifecycle."""
from dataclasses import replace
from examples.general_contact_automatic_s3_pair import build_s3_pair,solve_shell_pair
from anyfesolver.general_contact import resolve_automatic_contact_domain


def build_orientation_pair(*,director_z=1,orientation='physical_sides_dev',resolve_domain=True):
    model,domain,settings,_,path=build_s3_pair(director_z=director_z,resolve_domain=False)
    # Measured candidate orientation string; _dev suffixes mark unmeasured routes.
    settings=replace(settings,shell_pair_orientation=orientation)
    resolution=(resolve_automatic_contact_domain(model,domain,settings,analysis_control='contact_path')
                if resolve_domain else None)
    return model,domain,settings,resolution,path


if __name__=='__main__':
    import json
    result=solve_shell_pair(*build_orientation_pair(),emit_restart_checkpoint=True)
    if result.status!='completed':raise RuntimeError(result.failure_reason)
    print(json.dumps(dict(status=result.status,contact_path=result.info['contact_path'],
        contact=result.info['general_contact'],phases=result.info['force_displacement_history']),
        default=lambda x:x.item() if hasattr(x,'item') else x.tolist()))
