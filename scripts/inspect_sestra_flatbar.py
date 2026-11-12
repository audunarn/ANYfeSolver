"""Audit and display the actual ANYfem SESAM import; neither solver is run.

Run from any directory with the ANYfeSolver development Python environment.
By default this inspects the retained FEM used in the comparison. Pass --fem
to inspect another file, including the live GeniE export. Outputs are an
interactive, self-contained 3D HTML model and a JSON runtime audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import webbrowser


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fem', type=Path, default=repo / 'reports/sestra_development/correction_parent/reference_snapshot/20261008_140714_T1.FEM')
    parser.add_argument('--output', type=Path, default=repo / 'reports/sestra_development/flatbar_inspection')
    parser.add_argument('--open', action='store_true', help='Open the interactive model in your browser')
    args = parser.parse_args()
    # Use the checked-out ecosystem source rather than older installed packages.
    for name in ('ANYfem', 'ANYfeSolver', 'ANYfileIO', 'ANYgeometry', 'ANYmesh', 'ANYmaterial'):
        source = repo.parent / name / 'src'
        if source.is_dir():
            sys.path.insert(0, str(source))

    import numpy as np
    import plotly.graph_objects as go
    from anyfem.io.sesam import import_sesam
    from anyfesolver.elements import BeamElement

    fem = args.fem.resolve(strict=True)
    imported = import_sesam(fem, strict=True)
    model = imported.fe_model
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shells = [e for e in model.mesh.elements.values() if len(e.node_ids) == 4]
    beams = [e for e in model.mesh.elements.values() if isinstance(e, BeamElement)]
    if not shells or not beams:
        raise ValueError('Expected a mixed Q4 shell / two-node beam model')
    # Plate normals come from imported shell geometry, not a hard-coded Z axis.
    shell_xyz = [e.get_node_coordinates(model.mesh) for e in shells]
    normals = []
    for xyz in shell_xyz:
        n = np.cross(xyz[1]-xyz[0], xyz[3]-xyz[0])
        normals.append(n / np.linalg.norm(n))
    shell_node_sets = [set(e.node_ids) for e in shells]

    fig = go.Figure()
    # Shell midsurfaces: explicit FEM triangles, no inferred BRep geometry.
    vertices = np.concatenate(shell_xyz)
    faces = [(4*j,4*j+1,4*j+2) for j in range(len(shells))] + [(4*j,4*j+2,4*j+3) for j in range(len(shells))]
    i,j,k = np.array(faces).T
    fig.add_trace(go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=i, j=j, k=k, color='#789fba', opacity=.55, name='Plate midsurface', showlegend=True, hoverinfo='name'))
    edges = []
    for xyz in shell_xyz:
        edges.extend([*xyz, xyz[0], [np.nan]*3])
    edges = np.array(edges)
    fig.add_trace(go.Scatter3d(x=edges[:,0], y=edges[:,1], z=edges[:,2], mode='lines', line=dict(color='#567184',width=1), name='Actual shell mesh', hoverinfo='skip'))

    audit = []
    box_triangles = np.array([(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)])
    centers, arms = [], []
    for index, e in enumerate(sorted(beams, key=lambda e:e.element_id)):
        section = e.cross_section
        if section.get('section_type') != 'FB':
            raise ValueError(f'Element {e.element_id}: this script draws FB sections only')
        height, width = float(section['web_height']), float(section['web_thickness'])
        # These are the effective centroid coordinates and transform used in
        # linear stiffness/recovery. The transform includes the rigid-arm map;
        # its leading translation block remains the global-to-local rotation.
        xyz = e.get_node_coordinates(model.mesh)
        length, transform = e._beam_frame_and_transform(xyz)
        rotation = transform[:3,:3].T
        system_xyz = np.array([model.mesh.get_node(n).coords() for n in e.node_ids])
        offsets = xyz-system_xyz
        adjacent = [n for n,node_set in zip(normals,shell_node_sets) if set(e.node_ids)&node_set]
        # Several adjoining plate planes make one "perpendicular" judgement
        # ambiguous. Still display and report the model; never guess a plane.
        plane_resolved = bool(adjacent) and all(abs(float(np.dot(n,adjacent[0]))) > 1-1e-8 for n in adjacent)
        perpendicular = plane_resolved and abs(float(np.dot(rotation[:,2],adjacent[0]))) > 1-1e-8
        expected_area = width*height
        expected_iy, expected_iz = width*height**3/12, height*width**3/12
        inertia_ok = bool(np.allclose([section['area'],section['Iy'],section['Iz']], [expected_area,expected_iy,expected_iz], rtol=1e-6, atol=0))
        audit.append(dict(element_id=e.element_id, element_class=type(e).__module__+'.'+type(e).__name__, section=dict(section), length_m=length, local_x_global=rotation[:,0].tolist(), local_y_global=rotation[:,1].tolist(), local_z_web_global=rotation[:,2].tolist(), plate_normal_global=adjacent[0].tolist() if plane_resolved else None, adjoining_plate_plane_resolved=plane_resolved, adjoining_shell_normals_global=[n.tolist() for n in adjacent], centroid_endpoints_m=xyz.tolist(), system_endpoints_m=system_xyz.tolist(), global_offsets_m=offsets.tolist(), web_perpendicular_to_plate=bool(perpendicular), rectangular_flatbar_properties_match=inertia_ok))
        # Solid rectangular section swept using the actual local-y/local-z axes.
        corners = np.array([p + sy*width/2*rotation[:,1] + sz*height/2*rotation[:,2] for p in xyz for sy,sz in [(-1,-1),(1,-1),(1,1),(-1,1)]])
        i,j,k = box_triangles.T
        fig.add_trace(go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2], i=i, j=j, k=k, color='#df8c32', name='Flatbars from actual solver axes', legendgroup='beams', showlegend=index==0, hovertemplate=f'Beam {e.element_id}<br>{height*1000:.1f} x {width*1000:.1f} mm<extra></extra>'))
        centers.extend([*xyz,[np.nan]*3])
        for p,q in zip(system_xyz,xyz):
            arms.extend([p,q,[np.nan]*3])
    centers, arms = np.array(centers), np.array(arms)
    fig.add_trace(go.Scatter3d(x=centers[:,0],y=centers[:,1],z=centers[:,2],mode='lines',line=dict(color='#713d12',width=3),name='Solver beam centroid lines'))
    fig.add_trace(go.Scatter3d(x=arms[:,0],y=arms[:,1],z=arms[:,2],mode='lines',line=dict(color='#555555',width=2,dash='dash'),name='System-node to centroid offsets'))
    modules = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in ('anyfem.io.sesam','anyfesolver.sesam_fem.importer','anyfesolver.elements','anyfileio.sesam.document')}
    checks_pass = all(a['web_perpendicular_to_plate'] and a['rectangular_flatbar_properties_match'] for a in audit)
    record = dict(source_fem=str(fem), source_sha256=hashlib.sha256(imported.source_contents).hexdigest(), route='anyfem.io.sesam.import_sesam(strict=True) -> imported.fe_model', loaded_modules=modules, shell_count=len(shells), beam_count=len(beams), diagnostics=[str(d) for d in imported.diagnostics], upright_flatbar_checks_pass=checks_pass, beams=audit, limitation='Import/orientation audit only; no numerical solve or result acceptance. Solid flatbars are a display reconstruction from effective beam axes and section dimensions. Shells displayed at their midsurfaces.')
    (output/'beam_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    fig.update_layout(title=f'Actual ANYfem imported model: {fem.name}', scene=dict(xaxis_title='Global X [m]',yaxis_title='Global Y [m]',zaxis_title='Global Z [m]',aspectmode='data',camera=dict(eye=dict(x=1.2,y=-1.6,z=.45))), margin=dict(l=0,r=0,b=0,t=55))
    html = output/'imported_model.html'
    fig.write_html(html,include_plotlyjs=True)
    print(json.dumps(dict(fem=str(fem),shells=len(shells),beams=len(beams),upright_flatbar_checks_pass=checks_pass,flagged_beams=[a['element_id'] for a in audit if not (a['web_perpendicular_to_plate'] and a['rectangular_flatbar_properties_match'])],first_beam_axes={k:audit[0][k] for k in ('element_id','local_x_global','local_y_global','local_z_web_global','global_offsets_m')},model_html=str(html),audit_json=str(output/'beam_audit.json')),indent=2))
    if args.open:
        webbrowser.open(html.as_uri())
    return 0 if checks_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
