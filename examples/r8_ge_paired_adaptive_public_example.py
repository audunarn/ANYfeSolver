"""Public paired signed load/unload/reload example; caller supplies limits."""
def continue_cycle(owner,loads,*,force_seed_sha256,increment_control,cancellation_token=None,on_accepted=None):
    from anyfesolver import stiffened_shell as ss
    if type(increment_control) is not ss.AdaptiveIncrementControl or increment_control!=ss.AdaptiveIncrementControl(4,96):
        raise ValueError('this prepared programme requires root-adopted4cutbacks/96total attempts')
    target=0.028578804694929322
    return owner.solve(loads,(target/2.,0.,target/2.,target),
        displacement_control=ss.DisplacementControl('shell',2,0,force_seed_sha256=force_seed_sha256),
        increment_control=increment_control,max_iterations=20,max_backtracks=8,
        cancellation_token=cancellation_token,on_accepted=on_accepted)
