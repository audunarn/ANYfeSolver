"""Run balanced free-free histories or demonstrate an incomplete envelope.

Use --include-failure to append an incompatible load, or
--disconnected-control to append a globally balanced load that is unbalanced
on each disconnected component. Neither control permits artificial balancing.
"""
import argparse
from contextlib import closing

from anyfesolver import (AnalysisSession, EnvelopeReducer, LoadCase, RecoveryConfig,
    ResourceConfig, StaticBatchError, StaticCase, iter_linear_batches)
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import FEModel


def build_model(disconnected=False):
    model = FEModel("free-free static history")
    model.add_material("steel", 210e9, 0.3, density=7850.0)
    pairs = [(1, 2, 0.0)] if not disconnected else [(1, 2, 0.0), (3, 4, 2.0)]
    for element, (left, right, y) in enumerate(pairs, 1):
        model.add_node(left, 0.0, y, 0.0)
        model.add_node(right, 1.0, y, 0.0)
        model.add_element(element, BeamElement(element, [left, right], "steel",
            {"area": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}))
    return model


def history(include_failure=False, disconnected=False):
    for index, (time, axial, torque) in enumerate([
            (0.0, 1000.0, 0.0), (0.17, -250.0, 20.0), (3.2, 400.0, -5.0)]):
        load = LoadCase("balanced-" + str(index))
        load.add_nodal_load(1, [-axial, 0.0, 0.0, -torque, 0.0, 0.0])
        load.add_nodal_load(2, [axial, 0.0, 0.0, torque, 0.0, 0.0])
        yield StaticCase(load.name, load, time)
    if include_failure or disconnected:
        load = LoadCase("incompatible component load")
        load.add_nodal_load(2, [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if disconnected:
            load.add_nodal_load(3, [-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        yield StaticCase("incompatible", load, 4.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-failure", action="store_true")
    parser.add_argument("--disconnected-control", action="store_true")
    args = parser.parse_args()
    model = build_model(args.disconnected_control)
    dofs = [model.mesh.nodes[2].dofs[i] for i in (0, 3)]
    recovery = RecoveryConfig(node_ids=[1, 2], element_ids=[1],
        components=["axial_stress", "von_mises"], include_displacements=False,
        include_reactions=True)
    resources = ResourceConfig(solver_threads=1, assembly_threads=1,
        recovery_threads=1, memory_limit_bytes=64 * 1024 * 1024)
    envelope = EnvelopeReducer()
    try:
        with AnalysisSession(model) as session:
            with closing(iter_linear_batches(model,
                    history(args.include_failure, args.disconnected_control),
                    batch_size=1, output_dofs=dofs, recovery_config=recovery,
                    resource_config=resources, session=session)) as stream:
                for block in stream:
                    envelope.update_batch(block)
                    info = block.case_results[0]["solver_info"]
                    print("certified", block.case_ids, "gauge residual", info["gauge_residual_norm"])
                envelope.finish()
    except StaticBatchError as error:
        # Earlier certified blocks remain useful; do not call finish here.
        print("stopped", error.diagnostics)
    print("complete", envelope.complete, "cases", envelope.case_count)
    print("tip ux extrema", envelope.extrema[("displacement", dofs[0])])


if __name__ == "__main__":
    main()
