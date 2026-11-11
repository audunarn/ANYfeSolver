"""Explicit public construction delegates to exact reviewed owners, not aliases."""
from hashlib import sha256
import json
import numpy as np
import pytest

from anyfesolver import ge_beam3_native as api
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_native_analysis import analysis
from test_ge_beam3_native_generalized import problem as generalized
from test_ge_beam3_native_fibre_static_element import problem as fibre
from test_ge_beam3_durable_coupled import durable


def make(family='generalized', curved=True):
    model, element = (generalized if family == 'generalized' else fibre)(curved, False)
    direct = analysis(model)
    reference = element.operator.reference
    definition = api.define_beam(api.SELECTOR, 1, (1, 2, 3), reference.coordinates,
        reference.nodal_triads, element.section, direct._inertias[1],
        quadrature=element.operator.order)
    owner = api.create_analysis(api.SELECTOR, (definition,), tuple(model.boundary_conditions))
    return definition, owner, direct


@pytest.mark.parametrize('family', ('generalized', 'fibre'))
@pytest.mark.parametrize('curved', (False, True))
def test_public_definition_owner_real_solve_and_restart_unchanged(family, curved, tmp_path):
    definition, owner, direct = make(family, curved)
    assert definition.raw == direct._definitions[0]
    assert type(owner) is type(direct) and owner.identity == direct.identity
    if family == 'generalized':
        args = (api.DistributedPattern(api.LinePattern(((1, .001, -.0002, .0001),)), ()),)
        method = 'solve_distributed'
    else:
        args = (((3, .001, -.0002, .0001),),)
        method = 'solve_nodal'
    result = getattr(owner, method)(*args, steps=1)
    expected = getattr(direct, method)(*args, steps=1)
    assert result.status == expected.status == 'completed'
    assert result.checkpoint == expected.checkpoint
    assert result.production_qualified is False
    fresh = api.create_analysis(api.SELECTOR, (api.BeamDefinition.from_bytes(
        definition.raw, expected_sha256=definition.sha256),), tuple(owner.model.boundary_conditions))
    recovery = fresh.recover(result.checkpoint, expected_sha256=result.checkpoint_sha256)
    assert canonical(recovery) == canonical(direct.recover(expected.checkpoint,
        expected_sha256=expected.checkpoint_sha256))
    provenance = json.loads(api.workflow_provenance(fresh))
    assert provenance['definition_sha256'] == owner.identity
    assert provenance['profile_sha256'] == sha256(api.PROFILE_BYTES).hexdigest()
    assert provenance['checkpoint_validated'] is False
    (tmp_path/'public.json').write_bytes(canonical(dict(checkpoint=result.checkpoint.decode('ascii'),
        recovery=recovery, provenance=provenance)))


@pytest.mark.parametrize('topology', ('Q4', 'S3-V2D'))
def test_public_coupled_owner_selects_exact_v2_with_real_replay(topology, tmp_path):
    prior, _ = durable(topology, True)
    owner = api.create_coupled_analysis(api.SELECTOR,
        tuple(api.BeamDefinition(raw) for raw in prior._definitions), prior._boundaries,
        prior._program, shell=prior._shell, shell_density=prior._density, **prior._controls)
    assert type(owner) is type(prior) and owner.identity == prior.identity
    actual = owner.solve(stop_after=1); expected = prior.solve(stop_after=1)
    assert actual.status == expected.status == 'paused', (actual.failure, expected.failure)
    assert actual.checkpoint == expected.checkpoint
    recovery = owner.recover(actual.checkpoint, expected_sha256=actual.checkpoint_sha256)
    assert canonical(recovery) == canonical(prior.recover(expected.checkpoint,
        expected_sha256=expected.checkpoint_sha256))
    assert json.loads(api.workflow_provenance(owner))['workflow'] == 'COUPLED_V2'
    (tmp_path/'public-coupled.json').write_bytes(canonical(dict(checkpoint=actual.checkpoint.decode('ascii'),
        recovery=recovery, provenance=json.loads(api.workflow_provenance(owner)))))


@pytest.mark.parametrize('selector', ('ge-beam3', 'beam', 'b3', '', None, True, 'GE-BEAM3-NATIVE'))
@pytest.mark.parametrize('factory', ('define_beam', 'create_analysis', 'create_coupled_analysis'))
def test_no_alias_or_legacy_fallback_before_construction(selector, factory, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('wrong selector entered implementation')
    for name in ('_Analysis', '_Coupled', 'ReferenceGeometry'): monkeypatch.setattr(api, name, forbidden)
    if factory == 'define_beam':
        args = (selector, None, None, None, None, None, None); kw = {}
    elif factory == 'create_analysis': args = (selector, None, None); kw = {}
    else:
        args = (selector, None, None, None)
        kw = dict(shell=None, targets=None, shell_fixed=None, nodal_forces=None)
    with pytest.raises(ValueError, match='explicit ge-beam3-native'): getattr(api, factory)(*args, **kw)


@pytest.mark.parametrize('quadrature', (True, 0, 3, 5, 4.0))
def test_unregistered_quadrature_rejected(quadrature):
    _, owner, _ = make()
    e = owner._elements[0]; r = e.operator.reference
    with pytest.raises(ValueError, match='quadrature'):
        api.define_beam(api.SELECTOR, 1, (1, 2, 3), r.coordinates, r.nodal_triads,
            e.section, owner._inertias[1], quadrature=quadrature)


def test_unknown_section_and_foreign_provenance_rejected():
    with pytest.raises(ValueError, match='section'):
        api.define_beam(api.SELECTOR, 1, (1, 2, 3), None, None, object(), None)
    with pytest.raises(ValueError, match='owner'): api.workflow_provenance(object())


def test_profile_does_not_self_declare_artifact_acceptance_or_relabel_history():
    value = json.loads(api.PROFILE_BYTES)
    assert value['acceptance'] == 'REQUIRES_SEPARATE_EXACT_ARTIFACT_ACCEPTANCE'
    assert value['selector'] == 'ge-beam3-native'
    assert 'production_qualified' not in value
    assert api.PROFILE_SHA256 == sha256(api.PROFILE_BYTES).hexdigest()
    assert set(api.__all__) <= set(vars(api))
    from anyfesolver.ge_beam3_element import GeometricallyExactBeam3D3NElement
    assert GeometricallyExactBeam3D3NElement.formulation_id == 'GE_BEAM3_DC_MIXED_K1_MACRO_V2'


def test_exported_programs_are_exact_existing_workflow_types():
    from anyfesolver._ge_beam3_retained_nodal_loading import Program as Nodal
    from anyfesolver._ge_beam3_retained_generalized_state import Program as Distributed
    from anyfesolver._ge_beam3_retained_translation_control import Program as Translation
    from anyfesolver._ge_beam3_retained_fibre_control import TranslationProgram as FibreTranslation
    assert api.NodalProgram is Nodal
    assert api.DistributedProgram is Distributed
    assert api.TranslationProgram is Translation
    assert api.FibreTranslationProgram is FibreTranslation
