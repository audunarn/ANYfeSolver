"""Actual public Q4/V2D assembly and post-capture owned-cache refusal.

The injection seam is the supported element-activity provider. No protected
function/class/module is monkeypatched and no fake lease is substituted.
"""
import json
import numpy as np
import pytest

from anyfesolver import FEModel, create_shell_element
from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from anyfesolver.matrix_assembly import AssemblyError, assemble_stiffness_matrix


def model():
    made = FEModel("public_mixed_cache_lifecycle")
    made.add_material("elastic", 1e5, .3, density=1.)
    for node, xyz in enumerate(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
                                (2., 0., 0.), (3., 0., 0.), (2., 1., 0.)), 1):
        made.add_node(node, *xyz)
    made.add_element(1, create_shell_element(1, (1, 2, 3, 4), "elastic", thickness=.05,
                                            reference_normal=(0., 0., 1.)))
    made.add_element(2, create_shell_element(2, (5, 6, 7), "elastic", thickness=.05,
                                            reference_normal=(0., 0., 1.)))
    assert type(made.mesh.elements[2]) is NativeParityE4PLS3V2DShellElement
    return made


def prewarm(made, which="q4"):
    for key in ((1, 2) if which == "both" else (1,)):
        element = made.mesh.elements[key]
        element.compute_stiffness_components(made.mesh, made.materials["elastic"])


class ObserveActivity:
    sequence = 0

    def __init__(self, made, mutation=None):
        self.model = made
        self.mutation = mutation
        self.entries = []
        self.original_containers = []

    def scales(self, quantity, element_ids):
        mesh = self.model.mesh
        containers = (mesh._sparsity_cache, mesh._topology_signature_cache)
        assert all(type(x) is dict for x in containers)
        self.entries.append((quantity, tuple(element_ids)))
        self.original_containers.append(containers)
        if self.mutation is not None:
            self.mutation(self.model)
        return np.ones(len(element_ids))


def causes(error):
    out, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        out.append(type(error).__name__+": "+str(error))
        error = error.__cause__
    return out


def test_public_mixed_cold_and_prewarmed_assembly_are_identical(record_property):
    expected = None
    for which in ("cold", "q4", "both"):
        made = model()
        assert getattr(made.mesh, "_sparsity_cache", None) is None
        assert getattr(made.mesh, "_topology_signature_cache", None) is None
        if which != "cold":
            prewarm(made, which)
        first, _ = assemble_stiffness_matrix(made)
        caches = (made.mesh._sparsity_cache, made.mesh._topology_signature_cache)
        assert all(type(x) is dict and x for x in caches)
        second, _ = assemble_stiffness_matrix(made)
        assert made.mesh._sparsity_cache is caches[0]
        assert made.mesh._topology_signature_cache is caches[1]
        np.testing.assert_array_equal(first.toarray(), second.toarray())
        if expected is None:
            expected = first.toarray()
        else:
            np.testing.assert_array_equal(first.toarray(), expected)
    record_property("public_modes", "cold,q4-prewarm,both-prewarm;each repeated;byte-equal matrices")


def test_supported_provider_seam_without_mutation_preserves_matrix(record_property):
    baseline = assemble_stiffness_matrix(model())[0].toarray()
    made = model()
    prewarm(made)
    observer = ObserveActivity(made)
    made.mesh.element_activity = observer
    actual, _ = assemble_stiffness_matrix(made)
    assert observer.entries and all(ids == (1, 2) for _, ids in observer.entries)
    assert all(made.mesh._sparsity_cache is pair[0] and
               made.mesh._topology_signature_cache is pair[1] for pair in observer.original_containers)
    np.testing.assert_array_equal(actual.toarray(), baseline)
    record_property("actual_provider_entries", json.dumps(observer.entries))


@pytest.mark.parametrize("cache_name", ("_sparsity_cache", "_topology_signature_cache"),
                         ids=("sparsity", "topology"))
def test_later_owned_cache_replacement_is_refused(cache_name, record_property):
    made = model()
    prewarm(made)
    def replace_cache(current):
        # Normal supported caller mutation, not a numerical-authority patch.
        setattr(current.mesh, cache_name, {})
    observer = ObserveActivity(made, replace_cache)
    made.mesh.element_activity = observer
    with pytest.raises(AssemblyError) as caught:
        assemble_stiffness_matrix(made)
    assert observer.entries  # mutation was actually reached after capture
    index = 0 if cache_name == "_sparsity_cache" else 1
    assert getattr(made.mesh, cache_name) is not observer.original_containers[0][index]
    chain = causes(caught.value)
    assert any("qualified Q4 warm assembly raw inputs changed" in text for text in chain)
    record_property("refusal_chain", json.dumps(chain))
    record_property("actual_provider_entries", json.dumps(observer.entries))


def test_later_physical_input_mutation_is_refused(record_property):
    made = model()
    prewarm(made)
    before = made.mesh.nodes[1].z
    def move_node(current):
        current.mesh.nodes[1].z = before+.001
    observer = ObserveActivity(made, move_node)
    made.mesh.element_activity = observer
    with pytest.raises(AssemblyError) as caught:
        assemble_stiffness_matrix(made)
    assert observer.entries and made.mesh.nodes[1].z == before+.001
    chain = causes(caught.value)
    assert any("changed" in text for text in chain)
    record_property("refusal_chain", json.dumps(chain))
    record_property("actual_provider_entries", json.dumps(observer.entries))
