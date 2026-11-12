"""Run a selected supported-beam history and retain bounded envelopes.

Run from a source checkout with its src directory on PYTHONPATH.
"""
from contextlib import closing

from anyfesolver import (AnalysisSession, EnvelopeReducer, FixedSupport, LoadCase,
    RecoveryConfig, ResourceConfig, StaticCase, iter_linear_batches)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def build_model():
    model = FEModel("supported static history")
    model.add_material("steel", 210e9, 0.3, density=7850.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel",
        {"area": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}))
    model.add_boundary_condition(FixedSupport("clamp", [1]))
    return model


def history():
    # Independent force patterns at irregular times; no interpolation or
    # proportionality is used. Reusing this mutable builtin load is supported.
    load = LoadCase("borrowed history load")
    for index, (time, axial, transverse, moment) in enumerate([
            (0.0, 1000.0, 0.0, 10.0),
            (0.17, -250.0, 500.0, 0.0),
            (3.2, 400.0, -150.0, -20.0)]):
        load.nodal_loads.clear()
        load.add_nodal_load(2, [axial, 0.0, transverse, 0.0, moment, 0.0])
        yield StaticCase("supported-" + str(index), load, time)


def main():
    model = build_model()
    selected_dofs = [model.mesh.nodes[2].dofs[i] for i in (0, 2)]
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1],
        components=["axial_stress", "von_mises"], include_displacements=False,
        include_reactions=True)
    resources = ResourceConfig(solver_threads=1, assembly_threads=1,
        recovery_threads=1, memory_limit_bytes=64 * 1024 * 1024)
    envelope = EnvelopeReducer()
    with AnalysisSession(model) as session:
        # closing also handles an early break, cancellation, or consumer error.
        with closing(iter_linear_batches(model, history(), batch_size=2,
                session=session, output_dofs=selected_dofs,
                recovery_config=recovery, resource_config=resources)) as stream:
            for block in stream:
                envelope.update_batch(block)
                print("certified", block.case_ids, block.displacements.tolist())
            envelope.finish()  # Only after all intended cases were exhausted.
    tip = envelope.extrema[("displacement", selected_dofs[1])]
    print("complete", envelope.complete, "cases", envelope.case_count)
    print("tip uz absolute maximum", tip.absolute_maximum)


if __name__ == "__main__":
    main()
