"""Positive angular measures, precision rejection and historical identity guards."""
from decimal import Decimal, localcontext
from hashlib import sha256
from math import acos, pi
from pathlib import Path
import json
import pytest
import numpy as np
from anyfesolver.general_contact.native_normal_angles import PreciseNormalAngles


@pytest.mark.parametrize('slope',(0.,1e-20,-1e-20))
def test_precise_endpoint_roots_preserve_positive_partial_arcs(slope):
    field=PreciseNormalAngles(.0491,slope,.8,.05,0.)
    breaks=field.breakpoints(pi-.3,pi+.3)
    assert len(breaks)==(5 if slope==0. else 7)
    # Independent analytical acos locations, at ordinary precision.
    root=acos(-.0491/.05)
    assert abs(float(breaks[1])-root)<=1e-15
    assert abs(float(breaks[-2])-(2*pi-root))<=1e-15
    with localcontext() as ctx:
        ctx.prec=50
        widths=[b-a for a,b in zip(breaks[:-1],breaks[1:])]
        assert all(w>0 for w in widths)
        if slope:
            assert min(widths)<Decimal('1e-18') and float(min(widths))>0.
            # A float-only angular partition would silently merge this arc.
            assert any(float(a)==float(b) for a,b in zip(breaks[:-1],breaks[1:]))
        for angle in (breaks[1],breaks[-2]):
            g=field.value(angle)
            assert min(abs(g),abs(g+field.axial_change))<=Decimal('1e-45')


def test_unresolved_positive_axial_separation_fails_explicitly():
    field=PreciseNormalAngles(.0491,1e-100,.8,.05,0.)
    with pytest.raises(ValueError,match='angular axial separation unresolved'):
        field.breakpoints(pi-.3,pi+.3)


def test_constant_angular_gap_does_not_create_roots():
    field=PreciseNormalAngles(1.,0.,.8,0.,0.)
    assert field.breakpoints(0.,2*pi)==[Decimal(0),Decimal(2*pi)]
    assert field.endpoint_values(pi)==(1.,1.)


def test_positive_quadrature_measure_underflow_cannot_report_zero_work():
    from anyfesolver._ge_beam3_mixed_ad import Jet2
    from anyfesolver.general_contact.native_surface_normal import (
        StraightNormalIntegrationSettings, _integrate_material_normal_data)
    minimum=float(np.nextafter(0.,1.))
    with localcontext() as ctx:
        ctx.prec=50
        upper=2*Decimal(minimum)
        # Independent constant integral is positive and representable. This is
        # an arithmetic discriminator, not an admitted physical structural case.
        expected=float(upper*Decimal(1e307))
    assert expected>1e-17
    integrand=lambda angle:Jet2.constant(1e307,42)
    data=[(integrand,np.eye(42),[Decimal(0),upper])]
    settings=StraightNormalIntegrationSettings(1e-20,1e-20,1e-20,64,1.)
    with pytest.raises(ValueError,match='quadrature measure unresolved'):
        _integrate_material_normal_data(data,42,settings)


@pytest.mark.parametrize('kind',('s3','plane'))
def test_prior_material_angular_arithmetic_checkpoint_is_incompatible(kind):
    if kind=='s3':
        from test_general_contact_native_material_s3 import _analysis
    else:
        from test_general_contact_native_material_plane import _analysis
    owner=_analysis()
    path=Path(__file__).with_name('data')/f'general_contact_material_{kind}_ce3e5495.checkpoint'
    raw=path.read_bytes();record=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sha256(raw).hexdigest()==record['checkpoint_sha256']
    assert record['completed_targets']==0 and json.loads(raw)['completed_targets']==0
    with pytest.raises(ValueError,match='binding'):
        owner.solve(checkpoint=raw,expected_sha256=record['checkpoint_sha256'],stop_after=0)
