"""Two-macro nonlinear correctness funnel, no full qualification campaign."""
import numpy as np
import pytest
import json
from decimal import Decimal as D
from anyfesolver._ge_beam3_seeded_load_program import ForceProgram
from anyfesolver._ge_beam3_p5_seeded import NativeP5BeamElement
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from docs.reference_cases.ge_beam3_retained_plastic_force_probe import solve
from test_ge_beam3_curved_contrast_probe import make_curved
from test_ge_beam3_complementary_section import section
from test_ge_beam3_station_resultant_cell import identities
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam


def section_data(element, origins):
    core = element.core; law = core.section
    source = CenteredStationaryBeam(core.reference, law, position_low=np.zeros((3, 3)), order=core.order)
    data = dict(elastic=law._elastic.tolist(), direction=law._direction.tolist(),
        yield_force=law._yield, hardening=law._hardening, stations=[])
    for cell in (0, 1):
        for index, t, _, weight, v, _, _ in source._stations[cell]:
            zh, zl, ph, pl = origins[cell*core.order+index]
            data['stations'].append(dict(cell=cell, t=float(t), weight=float(weight), v=v.tolist(),
                origin=[float(zh), float(ph)], origin_low=[float(zl), float(pl)]))
    return data


def model_with_plasticity(contrast):
    model, _ = make_curved(100.)
    for key, old in list(model.mesh.elements.items()):
        model.mesh.elements[key] = NativeP5BeamElement(key, old.node_ids, old.core.reference,
            section(contrast), line_force=np.zeros(3))
        current = model.mesh.elements[key]
        model.materials[current.material_name] = current.core.section
    return model


@pytest.mark.parametrize('contrast', [1., 1000000000000.])
def test_two_macro_plastic_load_reverse_final_state_replay(contrast, tmp_path):
    model = model_with_plasticity(contrast)
    program = ForceProgram((.25, .5, 1., .5, 0., -.5, 0.), ((5, .4, -.005, 0.),), max_iterations=24)
    result = solve(model, program)
    (tmp_path/'probe.json').write_bytes(canonical(result))
    assert result['status'] == 'completed', result['failure']
    assert result['completed_targets'] == len(program.targets)
    assert any(np.any(np.array(row['histories'])[:, :, 2] > 0.) for row in result['records'])
    assert max(max(row['metrics']) for row in result['records']) <= 1e-11
    # Plastic history is irreversible although the external load returns to zero.
    assert np.max(result['histories'][:, :, 2]) > 0.
    for a, b in zip(result['records'], result['records'][1:]):
        assert canonical(a['histories']) == canonical(b['origins'])
    checked = []
    for record in result['records']:
        for i, element in enumerate(model.mesh.elements.values()):
            data = section_data(element, record['origins'][i])
            errors = identities(data, record['state']['resultants'][i].tolist(), json.loads(record['material'][i]))
            checked.append(dict(target=record['target'], element=i+1, errors=errors))
    (tmp_path/'station-identities.json').write_bytes(canonical(checked))
    assert max(D(x) for row in checked for x in row['errors'].values()) <= D('1e-11')


@pytest.mark.parametrize('stage,target,accepted', [('before_commit', 1, 0), ('before_commit', 2, 1), ('committed', 2, 2)])
def test_interruption_preserves_last_published_state(stage, target, accepted, tmp_path):
    def interrupt(event):
        if event['stage'] == stage and event['target'] == target: raise RuntimeError('injected interruption')
    program = ForceProgram((.25, .5, 1.), ((5, .4, -.005, 0.),), max_iterations=24)
    result = solve(model_with_plasticity(1.), program, progress=interrupt)
    (tmp_path/'interrupted.json').write_bytes(canonical(result))
    assert result['status'] == 'failed' and 'injected interruption' in result['failure']
    assert result['completed_targets'] == accepted
    if accepted:
        record = result['records'][-1]
        assert canonical(result['histories']) == canonical(record['histories'])
        assert canonical(result['replay_origins']) == canonical(record['origins'])
        assert canonical(result['state']) == canonical(record['state'])
    else:
        assert np.all(result['histories'] == 0.)
        assert np.all(result['state']['resultants'] == 0.)
