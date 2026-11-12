"""Development cross-check of production float and Jet2 paths at a saved wedge."""

import argparse
import json
from pathlib import Path

import numpy as np

from anyfesolver.general_contact.q4_s3_overlap_derivatives import q4_s3_overlap_derivatives
from anyfesolver.general_contact.s3_patch_static import GeneralS3PatchManager
from scripts.general_contact_d1_coupled_refinement import build_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slide-offset", type=float, default=-3.0e-8)
    parser.add_argument("--difference-step", type=float, default=3.0e-9)
    parser.add_argument("--full-reduced", action="store_true")
    args = parser.parse_args()
    saved = json.loads(args.pose.read_text(encoding="utf-8"))
    model, definition, settings, _ = build_case(
        4, 2, True, gap_certificate="projected_matching",
        gap_tolerance=5.0e-5, normal_approach=0.01504,
    )
    owner = GeneralS3PatchManager(model, definition, settings)
    geometry = owner.law.geometry
    source = np.asarray(saved["source"], dtype=float)
    source[:, 0] += args.slide_offset
    target = np.asarray(saved["target"], dtype=float)
    total = geometry.evaluate(source, target)
    support_error = np.zeros_like(total.reference_support)
    gap_error = np.zeros_like(total.reference_weighted_gap)
    pair_rows = []
    for pair_index, law in enumerate(geometry.pair_laws):
        source_face = pair_index // len(geometry.target_faces)
        target_face = pair_index % len(geometry.target_faces)
        source_ids = geometry.source_face_indices[source_face]
        target_ids = geometry.target_face_indices[target_face]
        local_source = source[source_ids]
        local_target = target[target_ids]
        value = law.evaluate(local_source, local_target)
        if value.reference_overlap_area <= 0.0:
            continue
        derivative = q4_s3_overlap_derivatives(
            geometry.source_faces[source_face].reference_positions,
            local_source, local_target,
            area_tolerance=geometry.q4_source_area_tolerance,
            weighted_gap_tolerance=geometry.q4_source_weighted_gap_tolerance,
            maximum_triangles=geometry.maximum_q4_source_integration_triangles,
            allow_warped_source=True,
            **dict(zip(("support_gradient_tolerance",
                        "support_hessian_tolerance",
                        "weighted_gap_gradient_tolerance",
                        "weighted_gap_hessian_tolerance"),
                       geometry.q4_source_derivative_tolerances or (None,)*4)),
        )
        ds = np.abs(value.reference_support-derivative.reference_support)
        dw = np.abs(value.reference_weighted_gap-derivative.reference_weighted_gap)
        support_error[source_ids] += ds
        gap_error[source_ids] += dw
        if pair_index != 2:
            continue
        local_coordinate_count = 3*(len(source_ids)+len(target_ids))
        original_local = np.concatenate((local_source, local_target)).reshape(-1)

        def float_at(shift):
            trial = original_local+shift
            current = law.evaluate(
                trial[:3*len(source_ids)].reshape(-1, 3),
                trial[3*len(source_ids):].reshape(-1, 3),
            )
            return np.stack((current.reference_support,
                             current.reference_weighted_gap))

        directions = {}
        for name, axes in (("source_slide", (0,)), ("source_normal", (2,))):
            direction = np.zeros(local_coordinate_count)
            for index in range(len(source_ids)):
                for axis in axes:
                    direction[3*index+axis] = 1.0
            step = args.difference_step

            def float_values(scale):
                return float_at(scale*direction)

            plus = float_values(step)
            minus = float_values(-step)
            row = {}
            for label, reference, gradient, hessian, positive, negative in (
                ("support", value.reference_support,
                 derivative.support_gradient, derivative.support_hessian,
                 plus[0], minus[0]),
                ("weighted_gap", value.reference_weighted_gap,
                 derivative.weighted_gap_gradient,
                 derivative.weighted_gap_hessian,
                 plus[1], minus[1]),
            ):
                first_float = (positive-negative)/(2*step)
                first_jet2 = gradient @ direction
                second_float = (positive-2*reference+negative)/(step*step)
                second_jet2 = np.einsum("i,kij,j->k", direction,
                                        hessian, direction)
                row[label] = {
                    "float_first": first_float.tolist(),
                    "jet2_first": first_jet2.tolist(),
                    "float_second": second_float.tolist(),
                    "jet2_second": second_jet2.tolist(),
                    "first_absolute_error": float(np.max(np.abs(first_float-first_jet2))),
                    "second_absolute_error": float(np.max(np.abs(second_float-second_jet2))),
                }
            directions[name] = row
        pair_rows.append({
            "pair": pair_index,
            "overlap_area_m2": value.overlap_area,
            "absolute_support_error_m2": ds.tolist(),
            "absolute_weighted_gap_error_m3": dw.tolist(),
            "directions": directions,
        })
        if args.full_reduced:
            # Only source ux/uz coordinates are differential in this controlled
            # fixture; target motion and source uy are prescribed. Check every
            # Hessian entry in that actual reduced local subspace.
            selected = np.array([3*node+axis for node in range(len(source_ids))
                                 for axis in (0, 2)], dtype=np.intp)
            h = args.difference_step
            basis = np.eye(local_coordinate_count)[selected]
            center = float_at(np.zeros(local_coordinate_count))
            float_gradient = np.zeros((2, len(source_ids), len(selected)))
            float_hessian = np.zeros((2, len(source_ids), len(selected),
                                      len(selected)))
            positive = []
            negative = []
            for index, vector in enumerate(basis):
                plus = float_at(h*vector)
                minus = float_at(-h*vector)
                positive.append(plus)
                negative.append(minus)
                float_gradient[:, :, index] = (plus-minus)/(2*h)
                float_hessian[:, :, index, index] = (
                    plus-2*center+minus)/(h*h)
            for i in range(len(selected)):
                for k in range(i+1, len(selected)):
                    mixed = (float_at(h*(basis[i]+basis[k]))
                             - float_at(h*(basis[i]-basis[k]))
                             - float_at(h*(-basis[i]+basis[k]))
                             + float_at(-h*(basis[i]+basis[k])))/(4*h*h)
                    float_hessian[:, :, i, k] = mixed
                    float_hessian[:, :, k, i] = mixed
            jet_gradient = np.stack((derivative.support_gradient[:, selected],
                                     derivative.weighted_gap_gradient[:, selected]))
            jet_hessian = np.stack((
                derivative.support_hessian[:, selected][:, :, selected],
                derivative.weighted_gap_hessian[:, selected][:, :, selected],
            ))
            gradient_error = float_gradient-jet_gradient
            hessian_error = float_hessian-jet_hessian
            projected = owner.law.evaluate(source, target,
                                           np.asarray(saved["multipliers"]))
            pressure = projected.pressure[source_ids]
            normalized_gap = projected.normalized_gap[source_ids]
            force_error = np.einsum("i,ij->j", pressure,
                gradient_error[1]-normalized_gap[:, None]*gradient_error[0])
            hessian_work_error = np.einsum("i,ijk->jk", pressure,
                hessian_error[1]-normalized_gap[:, None, None]*hessian_error[0])
            pair_rows[-1]["full_reduced"] = {
                "local_selected_coordinates": selected.tolist(),
                "float_gradient": float_gradient.tolist(),
                "jet2_gradient": jet_gradient.tolist(),
                "float_hessian": float_hessian.tolist(),
                "jet2_hessian": jet_hessian.tolist(),
                "force_error_n": force_error.tolist(),
                "hessian_work_error_n_per_m": hessian_work_error.tolist(),
                "force_error_norm_n": float(np.linalg.norm(force_error)),
                "hessian_work_error_norm_n_per_m": float(
                    np.linalg.norm(hessian_work_error)),
                "maximum_support_gradient_error_m": float(np.max(
                    np.abs(gradient_error[0]))),
                "maximum_support_hessian_error": float(np.max(
                    np.abs(hessian_error[0]))),
            }
    record = {
        "pose": str(args.pose), "slide_offset_m": args.slide_offset,
        "difference_step_m": args.difference_step,
        "total_support_m2": total.reference_support.tolist(),
        "total_weighted_gap_m3": total.reference_weighted_gap.tolist(),
        "sum_absolute_pair_support_error_m2": support_error.tolist(),
        "sum_absolute_pair_weighted_gap_error_m3": gap_error.tolist(),
        "pair_rows": pair_rows,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("max_relative_node_support_error", float(np.max(
        support_error/np.maximum(total.reference_support, 1.0e-20))))
    print("max_relative_node_weighted_gap_error", float(np.max(
        gap_error/np.maximum(np.abs(total.reference_weighted_gap), 1.0e-20))))
    if args.full_reduced:
        print("force_error_norm_n", pair_rows[0]["full_reduced"]["force_error_norm_n"])
        print("hessian_work_error_norm_n_per_m",
              pair_rows[0]["full_reduced"]["hessian_work_error_norm_n_per_m"])
    else:
        print("pair2", pair_rows)


if __name__ == "__main__":
    main()
