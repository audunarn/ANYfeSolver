"""Named native fluid data serialization; no trajectories or source campaigns."""
import copy
from dataclasses import replace
import json
import numpy as np
import pytest
from test_marine_potential_flow import fixture, body, frame
from anyfesolver.marine_potential_flow import (PotentialFlowDevelopmentContribution,
    PotentialFlowDevelopmentData, MarinePotentialFlowError, data_to_dict, data_from_dict)
from anyfesolver.marine_mechanical import (MarineMechanicalModel, MarineMechanicalInputError,
    model_to_dict, model_from_dict, model_to_json, model_from_json)

pytestmark=pytest.mark.serde


def test_named_data_and_model_roundtrip_preserves_full_arrays_without_aliases():
    data=fixture((frame('a'),frame('b')))
    encoded=data_to_dict(data)
    restored=data_from_dict(json.loads(json.dumps(encoded)))
    assert data_to_dict(restored)==encoded
    for name in ('kernel','restoring','added_mass_infinite','baseline','motion_bounds'):
        np.testing.assert_array_equal(getattr(restored,name),getattr(data,name))
    model=MarineMechanicalModel((body('a'),body('b')),
        contributions=(PotentialFlowDevelopmentContribution(data),))
    restored_model=model_from_json(model_to_json(model))
    assert model_to_dict(restored_model)==model_to_dict(model)
    encoded['kernel'][0][0][0]+=1.
    assert restored.kernel[0,0,0]==data.kernel[0,0,0]
    with pytest.raises(ValueError):
        restored.kernel.setflags(write=True)


def test_unknown_versions_fields_physics_order_duplicates_and_subclasses_refused():
    data=fixture((frame('a'),frame('b')))
    encoded=data_to_dict(data)
    for delta in ({'api_version':'unknown'},{'provider':'arbitrary.module'},
                  {'source_kind':'bem'},{'gravity_included':True},{'frames':[]}):
        bad=copy.deepcopy(encoded)
        bad.update(delta)
        with pytest.raises(MarinePotentialFlowError):
            data_from_dict(bad)
    model=MarineMechanicalModel((body('a'),body('b')),
        contributions=(PotentialFlowDevelopmentContribution(data),))
    duplicated=model_to_dict(model)
    duplicated['contributions']*=2
    with pytest.raises(MarineMechanicalInputError,match='duplicate'):
        model_from_dict(duplicated)
    wrong_model=model_to_dict(model)
    wrong_model['bodies'].reverse()
    with pytest.raises(MarineMechanicalInputError,match='order'):
        model_from_dict(wrong_model)
    class UnknownProvider(PotentialFlowDevelopmentContribution):
        pass
    with pytest.raises(MarineMechanicalInputError,match='trusted named'):
        model_to_dict(replace(model,contributions=(UnknownProvider(data),)))
    class UnknownData(PotentialFlowDevelopmentData):
        pass
    unknown=UnknownData(**{name:getattr(data,name) for name in data.__dataclass_fields__})
    with pytest.raises(MarinePotentialFlowError,match='exact named'):
        data_to_dict(unknown)
