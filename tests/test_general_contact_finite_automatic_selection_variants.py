"""Persistence variants from existing automatic fixtures; no structural solve."""
import importlib.util
import json
import os

import pytest
from dataclasses import replace
from pathlib import Path

from anyfesolver.general_contact import (
    AutomaticContactDomain, AutomaticContactDomainError, AutomaticContactParticipant,
    AutomaticContactResolution, ContactInteraction, RigidContactBody,
    resolve_automatic_contact_domain, verify_automatic_contact_resolution,
)


@pytest.mark.skipif(not os.environ.get('AUTOMATIC_VARIANT_FIXTURE')
                    or not Path(os.environ['AUTOMATIC_VARIANT_FIXTURE']).is_file(),
                    reason='archived automatic fixture module under '
                           'AUTOMATIC_VARIANT_FIXTURE absent: selection-variant '
                           'inverse matrix requires the existing automatic fixtures')
def test_current_resolution_inverse_variant_matrix():
    fixture_path = Path(os.environ['AUTOMATIC_VARIANT_FIXTURE'])
    spec = importlib.util.spec_from_file_location('existing_automatic_fixtures', fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    rows = []

    def check(name, model, domain, settings, resolution):
        original = resolution.to_dict()
        payload = json.loads(json.dumps(original, allow_nan=False))
        restored = AutomaticContactResolution.from_dict(payload)
        assert restored.to_dict() == original
        assert restored.identity == resolution.identity
        assert verify_automatic_contact_resolution(model, domain, restored, settings) == resolution.definition
        rows.append(dict(case=name, status='PASS', identity=restored.identity,
                         interactions=len(restored.definition.interactions),
                         dispositions=[r.disposition for r in restored.selections],
                         outside_family_pair_count=restored.outside_family_pair_count))

    for name, sizes in [('q4_coarser', (2., 1.)), ('q4_stable_tie', (1., 1.))]:
        model, domain, settings = fixtures.shell_pair(size_a=sizes[0], size_b=sizes[1])
        result = resolve_automatic_contact_domain(model, domain, settings)
        assert result.definition.interactions[0].surface == 'a'
        check(name, model, domain, settings, result)
        explicit = ContactInteraction('reverse-role', 'b', 'a', 1e5, .001)
        result = resolve_automatic_contact_domain(model, domain, settings, explicit_pairs=(explicit,))
        assert result.definition.interactions[0].surface == 'b'
        check(name + '_explicit', model, domain, settings, result)

    model, _, domain, settings, baseline = fixtures.build()
    assert len(baseline.definition.interactions) == 2 and baseline.outside_family_pair_count == 1
    check('q4_two_rigid_complete', model, domain, settings, baseline)
    explicit = ContactInteraction('user-name', 'skin', 'floor', 2e6, 2e-6)
    overridden = replace(domain, pair_overrides=(explicit,))
    result = resolve_automatic_contact_domain(model, overridden, settings, explicit_pairs=(explicit,))
    pair = next(p for p in result.definition.interactions if p.target == 'floor')
    row = next(r for r in result.selections if r.target == 'floor')
    assert row.original_interaction_name == 'user-name'
    assert pair.name.startswith('automatic:') and result.identity in pair.name
    assert pair.normal_stiffness == 2e6 and pair.gap_tolerance == 2e-6
    check('rigid_explicit_precedence', model, overridden, settings, result)
    excluded = replace(domain, excluded_pairs=(('floor', 'skin'),),
                       pair_overrides=(explicit, replace(explicit, normal_stiffness=3e6)))
    result = resolve_automatic_contact_domain(model, excluded, settings)
    assert len(result.definition.interactions) == 1
    assert next(r for r in result.selections if r.participants == ('floor', 'skin')).disposition == 'excluded'
    check('exclusion_suppresses_conflicting_overrides', model, excluded, settings, result)

    shell = next(p for p in domain.participants if p.participant.name == 'skin')
    for kind in ('plane', 'cylinder'):
        body = RigidContactBody('oblique', kind, (0., 0., -.025), (1., 1., 0.),
                                radius=.2 if kind == 'cylinder' else None)
        selected = replace(domain, participants=(shell, AutomaticContactParticipant(body, 'oblique-body')))
        encoded = json.loads(json.dumps(selected.to_dict(), allow_nan=False))
        restored_domain = AutomaticContactDomain.from_dict(encoded)
        assert restored_domain.to_dict() == selected.to_dict()
        assert restored_domain == selected  # normalized vector bits must survive, not normalize twice
        try:
            result = resolve_automatic_contact_domain(model, restored_domain, settings)
        except AutomaticContactDomainError as error:
            assert kind == 'cylinder', str(error)
            rows.append(dict(case='oblique_cylinder', status='EXISTING_RESOLVER_REFUSAL',
                             reason=str(error), domain_bit_roundtrip=True,
                             resolution_inverse_measured=False))
        else:
            check('oblique_' + kind, model, restored_domain, settings, result)
    output_env = os.environ.get('AUTOMATIC_VARIANT_OUTPUT')
    if output_env:
        output = Path(output_env)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(status='PASS', structural_solves=0,
            cases=rows, scope='Inverse of actual admitted current records; refused cylinder is domain persistence only'), indent=2))
