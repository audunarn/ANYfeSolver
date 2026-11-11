"""Explicitly selected development screen, no continuum acceptance claim."""
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from anyfesolver.general_contact.native_surface_transaction import NativeS3MaterialTransaction
from test_general_contact_native_surface_patch import _patch


@pytest.mark.skipif('CONTACT_SURFACE_REFINEMENT_OUTPUT' not in os.environ,
                   reason='bounded development screen requires an explicit output destination')
def test_fixed_patch_coupled_integration_screen():
    path=Path(os.environ['CONTACT_SURFACE_REFINEMENT_OUTPUT'])
    path.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for order in (2,4,8):
        start=perf_counter()
        owner,bridge=_patch(activation_policy='transverse_events',axial_order=order,angular_order=order)
        transaction=NativeS3MaterialTransaction(bridge)
        try:
            accepted=owner.solve_material_increment(transaction,1.)
            response=transaction.propose(np.zeros(42),np.zeros(18),1.).assembly
            contact=response.material.contribution
            records.append(dict(order=order,points=order**2,status='converged',
                elapsed_seconds=perf_counter()-start,tip_z=float(accepted.mechanical.positions[2,2]+accepted.mechanical.position_low[2,2]),
                normal_force=float(-sum(contact.residual[[2,8,14]])),gap=contact.normal_gap_m,
                equilibrium=list(response.metrics),active=sum(p.normal_pressure_Pa>0 for p in contact.points),
                total_reference_area=sum(s.reference_weight_m2 for s in bridge.pair.rule.samples)))
        except Exception as error:
            records.append(dict(order=order,status='failed',error=repr(error),elapsed_seconds=perf_counter()-start))
            path.write_text(json.dumps(records,indent=2));raise
        path.write_text(json.dumps(records,indent=2))
    assert all(np.isfinite(r['normal_force']) and r['normal_force']>0 for r in records)
