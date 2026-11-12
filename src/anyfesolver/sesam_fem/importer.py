"""SESAM-to-ANYfeSolver adapter over ANYfileio's neutral semantic model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anyfileio import (
    FemDiagnostic,
    SesamFemDocument,
    SesamFemError,
    read_sesam_fem_document,
    read_sesam_semantics,
)
from anyfileio.diagnostics import raise_if_errors
from anyfileio.sesam import (
    beam_section as _beam_section,
    material_name as _material_name,
    shell_thickness as _shell_thickness,
)


@dataclass(frozen=True)
class SesamFemImportResult:
    document: SesamFemDocument
    model: object | None
    diagnostics: tuple[FemDiagnostic, ...]
    element_count_by_type: dict[int, int]


def import_sesam_fem(
    path: str | Path,
    *,
    strict: bool = True,
    build_model: bool = True,
) -> SesamFemImportResult:
    """Read with ANYfileio and optionally adapt neutral semantics to FEModel."""

    document = read_sesam_fem_document(path, strict=strict)
    diagnostics = list(document.diagnostics)
    model = None
    if build_model:
        model, model_diagnostics = build_fe_model_from_sesam_document(document)
        diagnostics.extend(model_diagnostics)
    if strict:
        raise_if_errors(diagnostics, "SESAM FEM import failed")

    element_count_by_type: dict[int, int] = {}
    for element in document.elements.values():
        element_count_by_type[element.type_code] = (
            element_count_by_type.get(element.type_code, 0) + 1
        )
    return SesamFemImportResult(
        document=document,
        model=model,
        diagnostics=tuple(diagnostics),
        element_count_by_type=element_count_by_type,
    )


def _adapter_diagnostics(
    document: SesamFemDocument,
    semantic_diagnostics: tuple[FemDiagnostic, ...],
) -> list[FemDiagnostic]:
    """Drop document diagnostics already returned by import_sesam_fem."""

    prefix = tuple(document.diagnostics)
    if semantic_diagnostics[: len(prefix)] == prefix:
        return list(semantic_diagnostics[len(prefix) :])
    remaining = list(semantic_diagnostics)
    for diagnostic in prefix:
        if diagnostic in remaining:
            remaining.remove(diagnostic)
    return remaining


def _transform_ids(document: SesamFemDocument, element_id: int) -> tuple[int, ...]:
    reference = document.element_references.get(element_id)
    if reference is None:
        return ()
    if reference.nodal_transform_ids:
        return tuple(int(value) for value in reference.nodal_transform_ids)
    if reference.transform_id is not None:
        return (int(reference.transform_id),)
    return ()


def _attach_sesam_metadata(
    solver_element: object,
    document: SesamFemDocument,
    element_id: int,
    local_axes: dict[str, tuple[float, float, float]] | None,
) -> None:
    reference = document.element_references.get(element_id)
    if reference is not None:
        setattr(solver_element, "sesam_reference", reference)
    transform_ids = _transform_ids(document, element_id)
    if transform_ids:
        setattr(solver_element, "sesam_transform_ids", transform_ids)
    if local_axes is not None:
        setattr(solver_element, "sesam_local_axes", local_axes)


def build_fe_model_from_sesam_document(
    document: SesamFemDocument,
) -> tuple[object, tuple[FemDiagnostic, ...]]:
    """Build native solver objects from ANYfileio's neutral SESAM semantics."""

    from ..boundary import BoundaryCondition, LoadCase
    from ..elements import BeamElement, QuadraticBeamElement, create_shell_element
    from ..fe_core import FEModel

    semantics = read_sesam_semantics(document, strict=False)
    diagnostics = _adapter_diagnostics(document, semantics.diagnostics)
    name = f"sesam:{document.source_path.name}" if document.source_path else "sesam:fem"
    model = FEModel(name=name)

    material_names: dict[int, str] = {0: "default"}
    for material_id, specification in semantics.materials.items():
        material = specification.build()
        model.register_material(material)
        material_names[int(material_id)] = material.name

    for node_id, coordinates in semantics.mesh.nodes.items():
        model.add_node(int(node_id), *(float(value) for value in coordinates))

    for source_element in document.elements.values():
        element_id = int(source_element.element_id)
        material_id = semantics.material_of_element.get(element_id, 0)
        material_name = material_names.get(int(material_id), "default")
        try:
            if element_id in semantics.mesh.quads:
                node_ids = semantics.mesh.quads[element_id]
                solver_element = create_shell_element(
                    element_id,
                    list(node_ids),
                    material_name=material_name,
                    thickness=semantics.thickness_of_element[element_id],
                )
            elif element_id in semantics.mesh.tris:
                node_ids = semantics.mesh.tris[element_id]
                solver_element = create_shell_element(
                    element_id,
                    list(node_ids),
                    material_name=material_name,
                    thickness=semantics.thickness_of_element[element_id],
                    # Neutral legacy records carry connectivity but no
                    # physical-owner normal or formulation identity.
                    formulation="legacy-s3",
                )
                diagnostics.append(
                    FemDiagnostic(
                        "FEM131",
                        "historical TRI3 record loaded as legacy-s3; qualified S3 was not inferred",
                        severity="warning",
                        context={"element_id": element_id},
                    )
                )
            elif element_id in semantics.mesh.beams:
                node_ids = semantics.mesh.beams[element_id]
                element_type = QuadraticBeamElement if len(node_ids) == 3 else BeamElement
                solver_element = element_type(
                    element_id,
                    list(node_ids),
                    material_name=material_name,
                    cross_section=dict(semantics.section_of_element.get(element_id, {})),
                )
                # Beam offsets come from newer ANYfileio semantics (absent before).
                offsets=(getattr(semantics, "offsets_of_element", None) or {}).get(element_id)
                if offsets is not None:
                    solver_element.set_linear_nodal_offsets(offsets)
            else:
                continue
        except Exception as exc:
            diagnostics.append(
                FemDiagnostic(
                    "FEM130",
                    f"could not construct element {element_id}: {exc}",
                    context={"element_id": element_id},
                )
            )
            continue

        _attach_sesam_metadata(
            solver_element,
            document,
            element_id,
            semantics.local_axes_of_element.get(element_id),
        )
        model.add_element(element_id, solver_element)

    for index, support in enumerate(semantics.supports, start=1):
        model.add_boundary_condition(
            BoundaryCondition(
                f"sesam_BNBCD_{index}",
                [int(support.node_id)],
                {dof: 0.0 for dof in support.dofs},
            )
        )

    # Per-case nodal loads and pressures come from newer ANYfileio semantics;
    # with ANYfileio releases that lack them, nodal loads are simply absent.
    nodal_loads_by_case = getattr(semantics, "nodal_loads_by_case", None) or {}
    pressure_by_case = getattr(semantics, "pressure_by_case", None) or {}
    if nodal_loads_by_case:
        cases=set(nodal_loads_by_case)|set(pressure_by_case)
        cases|={int(r.raw_values[0]) for r in document.load_records if r.record_name=='BGRAV'}
        if len(cases)!=1:raise ValueError('nodal SESAM import requires exactly one shared source load case')
    if semantics.pressure_of_element or semantics.gravity is not None or nodal_loads_by_case:
        load_case = LoadCase("sesam_imported_load")
        for loads in nodal_loads_by_case.values():
            for node,force in loads.items():load_case.add_nodal_load(node,force)
        for element_id, pressure in semantics.pressure_of_element.items():
            load_case.add_pressure_load(int(element_id), float(pressure))
        if semantics.gravity is not None:
            load_case.set_gravity(*(float(value) for value in semantics.gravity))
        model.add_load_case(load_case)

    setattr(model, "sesam_document", document)
    setattr(model, "sesam_import_diagnostics", tuple(diagnostics))
    return model, tuple(diagnostics)


__all__ = [
    "SesamFemImportResult",
    "build_fe_model_from_sesam_document",
    "import_sesam_fem",
]
