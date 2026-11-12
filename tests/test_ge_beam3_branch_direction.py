"""Implicit native cell loading indicators, not a changed constitutive law."""
import numpy as np
import pytest
from anyfesolver._ge_beam3_generalized_cell import GeneralizedCellConjugate
from anyfesolver._ge_beam3_fibre_section import canonical
from test_ge_beam3_generalized_ellipsoid_section import section
from test_ge_beam3_fibre_cell import stations


def vector(parts):return np.asarray(parts[0])+np.asarray(parts[1])


@pytest.mark.parametrize("loaded", (False, True))
def test_implicit_station_direction_with_fixed_predecessor(loaded):
    law=section();cell=GeneralizedCellConjugate(law,stations(2))
    p=np.linspace(-.3,.5,18);d=.1*np.cos(np.arange(18))
    origin=cell.response((.7*p).tolist())["history"] if loaded else cell.virgin()
    captured=canonical(origin);plain=cell.response(p.tolist(),origin)
    response=cell.response(p.tolist(),origin,branch_direction=d.tolist())
    indicators=response.pop("branch_direction")
    assert canonical(response)==canonical(plain)
    epsilon=1e-5
    plus=cell.response((p+epsilon*d).tolist(),origin)
    minus=cell.response((p-epsilon*d).tolist(),origin)
    for j,row in enumerate(indicators):
        actual=(vector(plus["stations"][j]["resultants"])-vector(minus["stations"][j]["resultants"]))/ (2*epsilon)
        np.testing.assert_allclose(vector(row["resultant_direction"]),actual,rtol=1e-6,atol=1e-9)
        sp=vector(plus["stations"][j]["resultants"]);sm=vector(minus["stations"][j]["resultants"])
        derivative=(np.sqrt(sp@law.metric@sp)-np.sqrt(sm@law.metric@sm))/(2*epsilon)
        np.testing.assert_allclose(vector(row["margin_derivative"])[0],derivative,rtol=1e-6,atol=1e-9)
        s=vector(plain["stations"][j]["resultants"])
        margin=np.sqrt(s@law.metric@s)-law.yield_force-law.hardening*sum(origin.stations[j].accumulated)
        np.testing.assert_allclose(vector(row["margin"])[0],margin,rtol=1e-10,atol=1e-12)
    assert canonical(origin)==captured


def test_zero_norm_and_invalid_direction():
    cell=GeneralizedCellConjugate(section(),stations(2))
    response=cell.response([0.]*18,branch_direction=[1.]*18)
    assert all(r["margin_derivative"] is None for r in response["branch_direction"])
    for direction in ([0.]*17,[float("nan")]*18):
        with pytest.raises(ValueError,match="finite"):
            cell.response([0.]*18,branch_direction=direction)


def test_exact_boundary_and_failed_event_work_are_reported():
    from types import SimpleNamespace as NS
    from anyfesolver._stiffened_shell_branch_event_trial import trial_fraction
    from anyfesolver._stiffened_shell_displacement_corrector import DisplacementCorrectorFailure
    # Branch identity is authoritative even if a reconstructed margin rounds.
    cell=NS(response=lambda *args,**kwargs:dict(branch_direction=(dict(
        station=0,branch="YIELD_BOUNDARY",margin=([1e-80],[0.]),
        margin_derivative=([-1.],[0.])),)))
    element=NS(element_id=1,operator=NS(cell=cell))
    owner=NS(_shell_count=0,_beam_parts=((element,None,None),))
    driver=NS(owner=owner,_beam_nodal=0);state=NS(resultants=np.zeros((1,18)))
    work=dict(cell_evaluations=0)
    with pytest.raises(DisplacementCorrectorFailure,match="immediate"):
        trial_fraction(driver,state,np.ones(24),(None,),lambda:None,work)
    assert work["cell_evaluations"]==1
