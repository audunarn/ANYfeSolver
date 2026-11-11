"""Bounded native/shell static transaction driver with complete cell updates.

Owners supply work, admission, state and checkpoint contracts. The original
single-beam owner keeps its equations, identities and public controls.
"""
import numpy as np
from anyfesolver.control import cancellation_safe_point, SolveCancelled
from anyfesolver._ge_beam3_p5_seeded.core import sha


def solve_native_shell(owner, schema, run_type, *, checkpoint=None,
        expected_sha256=None, stop_after=None, cancellation_token=None, progress=None,initial_guess=None):
    def admitted(contact):
        gate = getattr(owner, '_contact_admissible', None)
        return (gate(contact) if gate is not None else
                contact.contact.minimum_gap_m >= -owner.gap_tolerance)
    end = len(owner.program.targets) if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= len(owner.program.targets):
        raise ValueError("native/S3 bounded stop target")
    if initial_guess is not None and (checkpoint is not None or end==0):
        raise ValueError('first-target guess requires a fresh nonempty solve')
    if checkpoint is None:
        if expected_sha256 is not None:
            raise ValueError("native/S3 hash requires checkpoint")
        accepted, records = owner.initial, ()
        capsule = owner._checkpoint(records)
    else:
        accepted, records = owner._restore(checkpoint, expected_sha256)
        capsule = checkpoint
    if accepted.completed_targets > end:
        raise ValueError("native/S3 accepted target cannot rewind")
    try:
        begin=getattr(owner,'_begin_candidate_context',None)
        if begin is not None:begin(cancellation_token)
        for index in range(accepted.completed_targets, end):
            cancellation_safe_point(cancellation_token, 'native/S3 target')
            parameter = owner.program.targets[index]
            trial = accepted.mechanical
            u = accepted.shell_displacement.copy()
            origins = accepted.native_histories
            if initial_guess is not None and index==0:
                trial,u=owner._initial_guess_trial(initial_guess,cancellation_token)
            for iteration in range(owner.program.max_iterations+1):
                cancellation_safe_point(cancellation_token, 'native/S3 candidate')
                r, j, metrics, responses, contact = owner._assemble(
                    trial, u, parameter, origins,
                )
                step, correction = owner._step(r, j)
                if max(*metrics, correction) <= owner.equilibrium_tolerance:
                    if not admitted(contact):
                        raise RuntimeError("native/S3 endpoint gap tolerance not met")
                    previous = (sha(records[-1]) if records else sha(dict(
                        schema=schema, identity=owner.identity,
                    )))
                    proposed, record = owner._record(
                        index+1, trial, u, origins, previous,
                    )
                    cancellation_safe_point(cancellation_token, 'native/S3 commit')
                    next_records = (*records, record)
                    staged = owner._checkpoint(next_records)
                    cancellation_safe_point(cancellation_token, 'native/S3 atomic commit')
                    accepted, records, capsule = proposed, next_records, staged
                    if progress is not None:
                        progress(accepted.completed_targets, accepted)
                    break
                if iteration == owner.program.max_iterations:
                    raise RuntimeError("native/S3 Newton iteration limit")
                angular = [step[6*i+3:6*i+6]
                           for i in range(len(owner.context.node_ids))]
                angular.extend(step[owner.context.nodal_count+24*b+3*c:
                                    owner.context.nodal_count+24*b+3*c+3]
                               for b in range(len(owner.context.elements)) for c in (0, 1))
                fraction = min(1.0, 0.45*np.pi/max(
                    max(float(np.linalg.norm(value)) for value in angular),
                    np.finfo(float).tiny,
                ))
                activation = getattr(owner, '_activation_fraction', None)
                if activation is not None:
                    angular_bound = fraction
                    fraction = activation(trial, u, step, fraction, contact, cancellation_token)
                    if not np.isfinite(fraction) or not 0. < fraction <= angular_bound:
                        raise RuntimeError('native/S3 activation fraction unresolved')
                rejected = dict(geometry=0, gap=0, swept=0, merit=0)
                last_rejection = ""
                for cut in range(owner.program.max_backtracks+1):
                    increment = step*(fraction*0.5**cut)
                    candidate = owner.context.advance(
                        trial, increment[:owner.context.count],
                    )
                    candidate_u = u+increment[owner.context.count:]
                    try:
                        preflight=getattr(owner,'_candidate_geometry_admissible',None)
                        if preflight is not None and not preflight(candidate,candidate_u):
                            rejected['gap']+=1
                            last_rejection='whole contact geometry preflight'
                            continue
                        candidate_contact = owner._contact(candidate, candidate_u)
                    except (ValueError, MemoryError) as exc:
                        if owner._recoverable_trial_error(exc):
                            rejected["geometry"] += 1
                            last_rejection = str(exc)[:160]
                            continue
                        raise
                    if not admitted(candidate_contact):
                        rejected["gap"] += 1
                        last_rejection = ("gap "
                                          f"{candidate_contact.contact.minimum_gap_m} m")
                        continue
                    swept=getattr(owner,'_swept_cancellable',owner._swept_admissible)
                    swept_kwargs=({'cancellation_token':cancellation_token}
                                  if hasattr(owner,'_swept_cancellable') else {})
                    if not swept(
                            trial, u, increment, candidate, candidate_u,
                            contact.contact.minimum_gap_m,
                            candidate_contact.contact.minimum_gap_m,
                            contact.contact.minimum_rim_clearance_m,
                            candidate_contact.contact.minimum_rim_clearance_m,**swept_kwargs):
                        rejected["swept"] += 1
                        last_rejection = "swept path not certified"
                        continue
                    _, _, changed, _, _ = owner._assemble(
                        candidate, candidate_u, parameter, origins,
                    )
                    if max(changed) < max(metrics):
                        trial, u = candidate, candidate_u
                        break
                    rejected["merit"] += 1
                    last_rejection = f"merit {max(changed)} >= {max(metrics)}"
                else:
                    raise RuntimeError(
                        "native/S3 line-search/gap limit: "
                        f"iteration={iteration}, rejected={rejected}, "
                        f"last={last_rejection}"
                    )
        status = "completed" if end == len(owner.program.targets) else "paused"
        return run_type(status, accepted, None, capsule)
    except Exception as exc:
        return run_type(
            "cancelled" if cancellation_token is not None and isinstance(exc, SolveCancelled) else "failed", accepted, type(exc).__name__+": "+str(exc), capsule,
        )
    finally:
        end_context = getattr(owner, '_end_candidate_context', None)
        if end_context is not None:
            end_context()
