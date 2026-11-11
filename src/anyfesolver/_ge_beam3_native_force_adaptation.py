"""Private deterministic step adaptation; no element or convergence tolerance changes."""
from dataclasses import dataclass

SCHEMA='GE_BEAM3_NATIVE_ADAPTIVE_FORCE_POLICY_V1'


@dataclass(frozen=True)
class AdaptiveForcePolicy:
    cutback_levels: int=2
    growth: bool=True
    fast_iterations: int=4
    slow_iterations: int=9

    def require(self):
        if type(self.cutback_levels) is not int or not 0<=self.cutback_levels<=6:
            raise ValueError('bounded integral cutback levels required')
        if type(self.growth) is not bool:
            raise ValueError('exact growth flag required')
        if (type(self.fast_iterations) is not int or type(self.slow_iterations) is not int
                or not 1<=self.fast_iterations<self.slow_iterations<=24):
            raise ValueError('ordered bounded iteration thresholds required')

    def descriptor(self):
        self.require()
        return dict(schema=SCHEMA,cutback_levels=self.cutback_levels,growth=self.growth,
                    fast_iterations=self.fast_iterations,slow_iterations=self.slow_iterations)


def describe(policy,steps,max_iterations,line_search):
    if policy is None:return None
    if type(policy) is not AdaptiveForcePolicy:
        raise ValueError('exact native adaptive force policy required')
    policy.require()
    if type(steps) is not int or steps not in (1,2,4,8,16):
        raise ValueError('adaptive nominal steps must be a bounded power of two')
    if type(max_iterations) is not int or not 1<=max_iterations<=24 or type(line_search) is not bool:
        raise ValueError('exact bounded adaptive Newton controls required')
    if steps*(1<<policy.cutback_levels)>64:
        raise ValueError('adaptive path exceeds complete-chain snapshot capacity')
    return dict(policy=policy.descriptor(),steps=steps,max_iterations=max_iterations,line_search=line_search)


def require_capacity(policy,steps,accepted_count):
    if policy is None:return
    policy.require()
    if type(accepted_count) is not int or accepted_count<0 or accepted_count+steps*(1<<policy.cutback_levels)>64:
        raise ValueError('adaptive continuation exceeds complete-chain snapshot capacity')


def settings(policy,steps,max_iterations,line_search):
    from .nonlinear_static import NonlinearConvergenceSettings
    describe(policy,steps,max_iterations,line_search)
    return NonlinearConvergenceSettings(profile='legacy',line_search='always' if line_search else 'never',
        fast_iterations=policy.fast_iterations,slow_iterations=policy.slow_iterations,
        growth_factor=2. if policy.growth else 1.,cutback_factor=.5,max_step_factor=2. if policy.growth else 1.,
        min_step_fraction=2.**(-policy.cutback_levels),max_line_search_cuts=8)
