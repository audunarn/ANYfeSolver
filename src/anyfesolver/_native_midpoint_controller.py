"""Shared native midpoint trajectory transaction engine; no physics/codec."""
from .control import cancellation_safe_point, SolveCancelled

def advance(analysis, state, t_end, identity, contract, controller, native, token, published, adapter):
    NativeTransientError = adapter.error
    _RETRYABLE = adapter.retryable
    _result = adapter.result
    settings = contract['settings']
    while state.time < t_end:
        candidate = None
        commit_endpoint = None
        commit_attempted = False
        try:
            cancellation_safe_point(token, 'native_transient.before_step')
            if controller['accepted_steps'] >= settings['max_steps']:
                raise NativeTransientError('native trajectory accepted step budget exhausted')
            planned = controller['next_dt']
            remaining = float(t_end-state.time)
            dt = min(planned, remaining)
            truncated = remaining < planned
            if not dt > 0 or not state.time+dt > state.time or state.time+dt > t_end:
                raise NativeTransientError('requested native endpoint is not representable without overshoot')
            failures = []
            def receive_candidate(issued):
                nonlocal candidate
                candidate = issued
            for cut in range(settings['max_cutbacks']+1):
                candidate = None
                try:
                    analysis._prepare_trial(state, dt=dt, cancellation_token=token,
                                            prepare=receive_candidate)
                    if candidate is None:
                        raise NativeTransientError('native trial did not provide its issued receipt')
                    break
                except ValueError as error:
                    if candidate is not None:
                        analysis._discard_prepared_trial(candidate)
                        candidate = None
                    if str(error) not in _RETRYABLE:
                        raise
                    failures.append(dict(dt=dt, reason=str(error)))
                    if cut == settings['max_cutbacks'] or .5*dt < settings['minimum_dt']:
                        raise NativeTransientError('native trajectory cutback budget/floor exhausted') from error
                    dt *= .5
            # Prepare ALL disposable work, accounting, controller/result bytes
            # and native capsule before the one native commit. None of these
            # objects is published as an accepted step until commit succeeds.
            next_controller, next_native, prepared = adapter.prepare_step(
                analysis, state, t_end, identity, contract, controller, candidate,
                dt, failures, planned, truncated, settings, token)
            cancellation_safe_point(token, 'native_transient.before_commit')
            commit_endpoint = candidate.endpoint
            commit_attempted = True
            analysis.commit(candidate, cancellation_token=token)
            # The native owner's visible endpoint, rather than a return value,
            # is authoritative after its atomic commit has completed.
            if analysis.current is not commit_endpoint:
                raise NativeTransientError('native commit did not publish its issued endpoint')
            state = commit_endpoint
            controller, native, published = next_controller, next_native, prepared
            candidate = None
        except BaseException as error:
            if commit_attempted and analysis.current is commit_endpoint:
                # Commit may have completed before its caller is interrupted.
                # Publish only bytes fully prepared before that authentic commit;
                # never discard or roll back an already accepted native endpoint.
                state = commit_endpoint
                controller, native, published = next_controller, next_native, prepared
                candidate = None
            if candidate is not None:
                try:
                    analysis._discard_prepared_trial(candidate)
                except BaseException as discard_error:
                    if isinstance(error, Exception) and not isinstance(error, SolveCancelled):
                        raise NativeTransientError('native candidate cleanup failed: '+str(discard_error),
                            partial=published, native_checkpoint=native, accepted_state=state) from error
                    error.cleanup_error = discard_error
            if isinstance(error, SolveCancelled):
                error.partial = published
                error.native_checkpoint = native
                error.accepted_state = state
                raise
            if not isinstance(error, Exception):
                error.partial = published
                error.native_checkpoint = native
                error.accepted_state = state
                raise
            raise NativeTransientError(str(error), partial=published,
                native_checkpoint=native, accepted_state=state) from error
    return published
