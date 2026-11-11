"""Experimental generalized-cell event trial for explicit displacement selection."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import isfinite

POLICY="GENERALIZED_BRANCH_EVENT_TRIAL_V1_DEV"


@dataclass(frozen=True)
class BranchEvent:
    element_id: int
    station: int
    branch: str
    margin: tuple
    derivative: tuple
    estimated_fraction: float


def _scalar(pair):
    return Decimal.from_float(pair[0][0])+Decimal.from_float(pair[1][0])


def trial_fraction(driver,state,direction,predecessors,check,work):
    """Estimate the first crossing using the actual implicit cell sensitivity.

    A 1% extrapolation proposes a trial beyond the estimate. It does not prove
    a crossing. Caller reassembles and accepts only actual merit reduction.
    Immediate yield boundaries are explicitly unresolved in this prototype.
    """
    from ._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure
    owner=driver.owner;events=[]
    retained=owner._shell_count+driver._beam_nodal
    with localcontext() as context:
        context.prec=80
        for i,((element,_,_),origin) in enumerate(zip(owner._beam_parts,predecessors)):
            check()
            dp=direction[retained+24*i+6:retained+24*i+24]
            work["cell_evaluations"]+=1
            response=element.operator.cell.response(state.resultants[i].tolist(),origin,
                check=check,branch_direction=dp.tolist())
            for indicator in response['branch_direction']:
                margin=_scalar(indicator['margin'])
                derivative=indicator['margin_derivative']
                if derivative is None:continue # q=0 is strictly inside Y+H*p0.
                slope=_scalar(derivative)
                if margin==0 or indicator["branch"]=="YIELD_BOUNDARY":
                    raise DisplacementCorrectorFailure('immediate material boundary requires a reviewed directional selection')
                if margin*slope>=0:continue
                fraction=-margin/slope
                if not fraction.is_finite() or fraction<=0:continue
                value=float(fraction)
                if not isfinite(value) or value<=0:
                    raise DisplacementCorrectorFailure('material event fraction not representable')
                if value<=1.:
                    events.append(BranchEvent(element.element_id,indicator['station'],
                        indicator['branch'],tuple(v[0] for v in indicator['margin']),
                        tuple(v[0] for v in derivative),value))
    return (min(1.,1.01*min(e.estimated_fraction for e in events)) if events else 1.),tuple(events)
