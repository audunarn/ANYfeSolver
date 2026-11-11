"""Reproducible accuracy evidence PDF; reads reports and never runs solvers."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path
from collections import Counter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
DATE = '8 October 2026'
NAVY = colors.HexColor('#17364a')
TEAL = colors.HexColor('#177c80')
PALE = colors.HexColor('#eff5f6')
INK = colors.HexColor('#263d49')

def ident(path):
    p=Path(path)
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}

def load(relative, evidence):
    p=ROOT / relative
    if not p.exists():return None
    evidence.append(ident(p))
    return json.loads(p.read_text(encoding='utf-8'))

def fmt(value, digits=6):
    return '-' if value is None else f'{value:.{digits}g}'

def build(structure_report=None,visual_directory=None):
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=[]
    historical=load('reports/verification/beam_shell_verification_report.json',evidence)
    external=load('reports/verification/external_reference_report.json',evidence)
    handoff=load('reports/external_references/external_reference_report.json',evidence)
    application=load('reports/verification/verification.json',evidence)
    modal=load('reports/verification/mass_modal_validity_report.json',evidence)
    buckling=load('reports/verification/buckling_validity_report.json',evidence)
    plasticity=load('reports/verification/plasticity_qualification_report.json',evidence)
    deep=load('reports/sestra_development/deep_parent/final_comparison/diagnostics_report.json',evidence)
    original=load('reports/sestra_development/three_case_parent/comparison2/suite_report.json',evidence)
    strip=load('reports/sestra_development/deep_worker/strip_refinement.json',evidence)
    cuts=load('reports/sestra_development/deep_worker/common_station_shear.json',evidence)
    structure=load(structure_report,evidence) if structure_report else None
    for relative in ['docs/QUALITY_CONTROL.md','docs/GE_BEAM3_DELIVERY_STATUS.md','docs/E4_PL_Q1AA_LOCAL_QUALIFICATION.md','docs/S4_IMPROVED_QUALIFICATION_FINAL_STATUS.md','docs/STIFFENED_SHELL_EXPERIMENTAL.md','docs/agent_plans/B3_GE_E4_PL_G0_F03_VARIATIONAL_REFINEMENT_V2_RESULTS.md','docs/agent_plans/B3_GE_E4_PL_G0_F03_SHELL_SPATIAL_LINEAR_PORT_RESULT_20260924.md']:
        p=ROOT/relative
        if p.exists():evidence.append(ident(p))
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyAccuracy',fontName='Helvetica',fontSize=9.3,leading=13.2,textColor=INK,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallAccuracy',fontName='Helvetica',fontSize=7.6,leading=10.3,textColor=INK,spaceAfter=5))
    styles.add(ParagraphStyle(name='TitleAccuracy',fontName='Helvetica-Bold',fontSize=31,leading=35,textColor=NAVY,spaceAfter=18))
    styles.add(ParagraphStyle(name='HeadingAccuracy',fontName='Helvetica-Bold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=13))
    styles.add(ParagraphStyle(name='SubAccuracy',fontName='Helvetica-Bold',fontSize=11.5,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=6))
    styles.add(ParagraphStyle(name='CellAccuracy',fontName='Helvetica',fontSize=7.5,leading=10,textColor=INK))
    story=[]
    def p(text,small=False):story.append(Paragraph(text,styles['SmallAccuracy' if small else 'BodyAccuracy']))
    def heading(title):story.append(Paragraph(title,styles['HeadingAccuracy']))
    def sub(title):story.append(Paragraph(title,styles['SubAccuracy']))
    def page(title):story.append(PageBreak());heading(title)
    def table(headers,rows,widths=None):
        headerstyle=ParagraphStyle(name='HeaderCell',parent=styles['CellAccuracy'],textColor=colors.white,fontName='Helvetica-Bold')
        content=[[Paragraph(html.escape(str(v)),headerstyle) for v in headers]]
        content += [[Paragraph(html.escape(str(v)),styles['CellAccuracy']) for v in row] for row in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,TEAL)]))
        story.append(t);story.append(Spacer(1,9))
    story.append(Spacer(1,22*mm))
    story.append(Paragraph('ANYfeSolver<br/>Accuracy evidence',styles['TitleAccuracy']))
    p('Linear-static commercial reference comparisons and a synthesis of earlier verification work.')
    p(f'<b>{DATE} | Evidence report, revision 1</b>')
    story.append(Spacer(1,12*mm))
    p('<b>What this report establishes</b><br/>Measured agreement for explicitly defined models, observables and software routes. It records independent analytical controls, reference-solver comparisons, mesh sensitivity and unresolved limitations.')
    p('<b>What a general accuracy statement requires</b><br/>Accuracy is not a single percentage for a solver. It depends on element formulation, loading, constraints, mesh, recovered quantity and execution route. Passing implementation checks does not establish physical convergence or certify every application.')
    p('The current study uses SESTRA as the analysis reference and GeniE-authored files as initial provenance. New geometry is generated automatically; no manual GeniE modelling is required. The commercial reference is checked against analytical solutions and is not treated as exact truth.')
    p(f'Current repository HEAD: {commit}. Current comparison source includes uncommitted development work; per-file hashes are retained. Earlier reports apply to their own recorded versions.',small=True)

    page('1 | Conclusions and evidence strength')
    p('The existing evidence supports a useful but bounded conclusion: elementary beam response and selected shell controls are accurate within their declared criteria. Matched shell formulations remove most of the earlier apparent ANYfeSolver/SESTRA discrepancy. Combined models still require quantity-specific convergence and some comparisons retain failed components.')
    table(['Finding','Evidence and practical meaning'],[
        ['Standalone upright flatbar','Translations and all beam-force/moment stations pass the existing SESTRA criteria; analytical Timoshenko tip displacement agrees.'],
        ['Plate reference formulation','SESTRA originally used FQUS thin shells. THCK=1 replaces them with THQS thick shells; this substantially reduces displacement and beam-force discrepancies.'],
        ['Independent plate-strip control','Thick-reference translations pass. Native tip agrees with the continuum hand solution, while interior discretisation error falls approximately fourfold after refinement.'],
        ['Combined structure','Stable-looking displacement does not establish local shear convergence. Beam shear changes substantially in both solvers when refining the same physical model.'],
        ['Other solver work','Historical analytical/invariant evidence and scoped native delivery are included separately from current external comparisons. Research and experimental routes retain their restrictions.']
    ],[42*mm,132*mm])
    if structure:
        p('Additional automatically generated structures have been executed and compared in this revision. Their measured results and dispositions appear in section 5; failed, unsupported and not-executed cases remain visible.')
    else:p('Additional frame, crossing-stiffener and folded-plate comparisons are not yet in this build. Their absence is a coverage gap, not a pass.')
    p('No default change, installed-package release acceptance or universal accuracy certificate is implied. A numerical failure is reported even when input generation, process execution and test infrastructure all succeed.')

    page('2 | Method and reproducibility')
    sub('One physical input, two execution routes')
    p('A parameterised definition sets geometry, materials, sections, local axes, eccentricities, supports and loads. The canonical ANYfileIO writer produces FEM files. SESTRA executes those files directly; ANYfeSolver imports the same files through its SESAM adapter. This reduces model differences but does not remove the possibility of a shared modelling mistake.')
    p('Independent audits check topology, section orientation, centroids, load resultants, clamped DOFs, physical coordinate systems and source/result identity. Analytical controls and refinement are therefore retained alongside cross-solver agreement. The original reference files remain immutable; modified geometry receives a distinct case identity.')
    sub('Fixed comparison criteria')
    table(['Observable','Development criterion'],[
        ['Global nodal translation','|u_ANY - u_ref| <= 1e-8 m + 0.001 |u_ref|, component by component'],
        ['Physical beam force','|F_ANY - F_ref| <= 0.001 N + 0.001 |F_ref|'],
        ['Physical beam moment','|M_ANY - M_ref| <= 0.001 Nm + 0.001 |M_ref|'],
        ['Global force/moment balance','Absolute 1e-6 in the respective SI unit plus relative 1e-8'],
        ['Rotations and strain energy','Separate quantities with explicit units; diagnostic criteria must be stated separately, never combined in a mixed-unit norm.']
    ],[47*mm,127*mm])
    p('Matching uses physical locations and result-case identities, not element numbers across different meshes. Metres, radians, newtons and newton-metres remain explicit. Complex or transformed result layouts are refused unless the canonical reader establishes their meaning.')
    sub('Recorded execution')
    p('SESTRA 10.20 is the tested executable. Inputs, controls, executable and outputs are hashed; runs use isolated folders with retained logs and process-wall/resource accounting. Native development runs use numerical pools of one and JIT off. Historical campaigns were inspected rather than rerun, preserving frozen authority and failed evidence.')

    page('3 | Original plate, flatbar and combined model')
    p('The retained GeniE-origin model has a 2 m x 2 m XY plate, 20 mm thickness, 400 quadrilateral shells and an upright 100 mm x 10 mm flatbar along its centreline. The y=0 edge is fully clamped and pressure is -1000 Pa. The section strong-axis inertia is about 8.333333e-7 m4. Its original centroid offset is -50 mm relative to the plate midsurface.')
    table(['Original model','ANYfeSolver peak mm','SESTRA thin mm','SESTRA thick mm'],[
        ['Plate','13.424466','13.440851','13.423233'],['Combined, -50 mm','4.428891','4.452274','4.429457'],['Isolated beam, 1000 N tip','15.275238','15.275238','Not applicable']
    ],[56*mm,39*mm,39*mm,40*mm])
    p('Changing only the SESTRA shell option reduces peak displacement difference from 0.122% to 0.0092% for the plate, and from 0.525% to 0.0128% for the combined model. Original root beam shear changes from SESTRA thin -1404.721 N to thick -665.326 N; ANYfeSolver gives -669.054 N. Small peak errors do not mean the complete field passes.')
    p('The official SESTRA manual places ELOP WCOR in columns 21-25 and THCK in columns 26-30. The original second field is zero. THCK=1 replaces FQUS with THQS; the actual log override is checked, since the input inventory can still display FQUS names.')
    p('The isolated beam hand check is w_tip = -P L^3/(3 E Iy) - P L/(kz G A), using the actual admitted section shear factor. The result is approximately -15.275238 mm. The flatbar orientation was audited from its actual transformation and inertia, rather than inferred from a schematic drawing.')
    p('For a bar directly below the underside of a 20 mm plate, the correct centroid is -(plate thickness + bar height)/2, approximately -60 mm. A separate under-plate model was generated; the original -50 mm model was not silently changed.')

    page('4 | Analytical controls and mesh sensitivity')
    p('For a cylindrical nu=0 cantilever plate strip under uniform pressure of positive magnitude p acting in negative Z, the independent continuum displacement is:')
    p('<b>w(y) = -p y^2 (6L^2 - 4Ly + y^2)/(24D) - p y(2L-y)/(2kGh)</b><br/>D = E h^3/12; G = E/2; k = 5/6.')
    if strip:
        rows=[]
        for v in strip['common_points'][:3]:rows.append([fmt(v['y_m']),fmt(v['error20_m']*1e6),fmt(v['error40_m']*1e6),fmt(v['error40_over20'],4)])
        table(['Y from clamp m','20-element error micrometres','40-element error micrometres','Refined/coarse'],rows,[33*mm,49*mm,49*mm,43*mm])
    p('The tip agrees almost exactly, but interior errors remain. Their approximately fourfold reduction is consistent with second-order discretisation. This is why the report separates a tip oracle, whole-field comparisons and mesh convergence.')
    if cuts:
        table(['Common physical right cut Y m','SESTRA shear, 20 N','SESTRA shear, 40 N'],[[fmt(c['global_y_m']),fmt(c['reference20_Fz_N']),fmt(c['reference40_Fz_N'])] for c in cuts],[62*mm,56*mm,56*mm])
    p('Local shear is not mesh-converged, including locations away from the clamp. The next physical question concerns shared-node rotational/shear compatibility and the converged coupled response. No core mechanics defect or repair is established by these results.')
    p('Applying native beam recovery to SESTRA six-DOF displacement fields removes most of the force discrepancy. A few small same-field differences remain. A propagated float32 and nine-significant-digit output bound explains some but not all of them; stiffness serialization and arithmetic are excluded from that bound. No tolerance was relaxed.')

    figure=ROOT/'reports/sestra_development/deep_parent/deep_findings.png'
    if figure.exists():
        page('Visual summary | Four discriminating checks')
        evidence.append(ident(figure));story.append(Image(str(figure),width=174*mm,height=107*mm))
        p('Top left: the original peak comparison changes when the reference shell option changes. Top right: the upright flatbar and distinct -50/-60 mm placements. Bottom left: beam shear depends on mesh in both solvers. Bottom right: interior strip error decreases fourfold although the tip already agrees.',small=True)
        p('These findings support matched formulations and quantity-specific convergence checks; they do not justify an unconditional solver-wide percentage.')

    page('5 | Automatically generated combined structures')
    table(['Structure','Mechanical purpose and definition'],[
        ['Inclined two-member cantilever frame','Two rigidly joined 2 m members with different local axes. Six-component terminal wrench exercises axial force, biaxial bending, torsion and joint transfer. Analytical tip compliance follows Castigliano strain energy.'],
        ['Crossing-stiffener panel','Two upright 100 x 10 mm under-plate stiffeners cross at a shared system node. A 2 x 2 m plate carries -1000 Pa, with its y=0 edge clamped. Checks intersecting load paths and eccentric beam-shell attachment.'],
        ['Right-angle folded plate','Two orthogonal 2 x 1 m faces share a continuous fold line. Both root edges are clamped. Two exact -500 N global-Z outer tip-corner forces preserve load points and resultants across refinements.']
    ],[48*mm,126*mm])
    p('Three mesh levels are defined per family. Folded-tip point forces create local stress singularities; displacement, energy and physically selected recovery locations are primary. Stress convergence at the loaded corner is not assumed.')
    if structure:
        cases=structure.get('cases',{})
        if isinstance(cases,list):cases={str(i):c for i,c in enumerate(cases)}
        rows=[];details={}
        for name,c in cases.items():
            detail=load(c['report_path'],evidence);details[name]=detail
            fields=detail.get('global_nodal_sixdof',{});peak=max((abs(v[2]) for v in fields.values()),default=0)
            error=max(detail.get('maximum_translation_absolute_errors_xyz_m',[0]))
            rows.append([name,str(c.get('status','unknown')),str(c.get('translation_failures','-'))+'/'+str(c.get('translation_components','-')),str(c.get('beam_failures','-'))+'/'+str(c.get('beam_components','-')),fmt(1000*peak,5),fmt(1e6*error,5)])
        table(['Case','Disposition','Translation failures / N','Beam failures / N','Peak |Uz| mm','Max |du| microm'],rows,[47*mm,22*mm,30*mm,29*mm,22*mm,24*mm])
        p('The case-level report is authoritative for observable definitions and units. A process exit of zero is execution success; numerical acceptance additionally requires the fixed comparison predicates and the reported controls. Unsupported layouts remain unsupported, not fabricated results.')
    else:p('Not executed in this build. The final PDF must be regenerated with the measured structure-suite report before claiming these cases were compared.')
    sub('Independent frame reference')
    p('For a terminal global wrench, determine each member internal force and moment solely from static equilibrium and its physical lever arm. Integrate N^2/(EA), shear terms V^2/(GAs), torsion T^2/(GJ) and biaxial M^2/(EI) over the members. The half-integral is strain energy; its derivative gives terminal displacements and rotations. This reference uses no native element stiffness or recovery code. It assumes a linear Timoshenko/Saint-Venant beam idealisation without restrained warping.')
    if structure:
        p('The frame load is (0.2, -0.3, -1) N plus My=0.1 Nm. The initial 1000-times larger load is retained as a linear algebra check; its metre-scale displacements are unsuitable as a small-displacement physical example. The final reduced-load cases are independently rerun in both solvers.')
        for name,title,label in [('inclined_frame_8','Model view | Inclined rigid frame','Two members and an explicit full terminal wrench. The displayed mesh has eight segments per member.'),('crossing_stiffeners_8','Model view | Crossing stiffeners','Upright bars below the plate underside. Red lines are physical beam centroids; dashed connectors are rigid eccentric arms, not additional structural members.'),('folded_plate_8','Model view | Folded plate','Two orthogonal faces share the fold. The global corner load positions and clamped edge remain identical across all meshes.')]:
            page(title)
            figure=Path(visual_directory)/(name+'.png') if visual_directory else Path(cases[name]['visual'])
            if figure.exists():
                evidence.append(ident(figure));story.append(Image(str(figure),width=174*mm,height=124.2857*mm));p(label)
            audit=details[name]['audit']
            table(['Displayed model inventory','Count'],[['Nodes',audit['nodes']],['Shells',audit['shells']],['Beams',audit['beams']]],[125*mm,49*mm])
            if name.startswith('inclined'):
                p('The independent frame oracle checks each translation using absolute 1e-10 m plus relative 1e-6, and each rotation using absolute 1e-10 rad plus relative 1e-6. Its focused energy check uses relative 1e-6. These analytical predicates are distinct from the external-comparison criteria.')
            if name.startswith('crossing'):
                p('Section height is 100 mm normal to the plate and thickness is 10 mm. The centroid lies 60 mm below the plate midsurface: half the 20 mm plate plus half the 100 mm bar. Strong and weak inertias differ by a factor of 100; the bar orientation is explicitly audited.')
        page('Refinement | Common locations and strain energy')
        p('The following values use the same physical probe at each mesh: terminal node for the frame, (1,2,0) for crossing stiffeners, and (1,2,0) for the folded plate. Energy is half the external load work for these linear, zero-prescribed-displacement models. Percent differences below are diagnostics, not substituted acceptance criteria.')
        rows=[]
        for name,d in details.items():
            n=d['native_probes'][1 if name.startswith('inclined') or name.startswith('crossing') else 0]['translations_m'][2]
            ref=d['reference_probes'][1 if name.startswith('inclined') or name.startswith('crossing') else 0]['translations_m'][2]
            en=d['native']['nodal_load_energy_j'];er=d['reference_load_energy_j']
            rows.append([name,fmt(n*1000),fmt(ref*1000),fmt(en),fmt(er),fmt(100*abs(en-er)/abs(er),4)])
        table(['Case','Uz ANY mm','Uz ref mm','U ANY J','U ref J','Energy difference %'],rows,[45*mm,26*mm,26*mm,24*mm,24*mm,29*mm])
        p('Folded-plate maximum translation discrepancy decreases with refinement, while failed-component counts can increase because the sampled inventory grows and small components have stricter absolute limits. Crossing-stiffener discrepancies are not monotonic. Neither observation establishes convergence of local beam forces or shell stresses. Concentrated-load corner stress singularities are excluded from a stress-convergence claim.')
        for relative in ['reports/sestra_development/structures_parent/external_ledger.json','reports/sestra_development/structures_parent/effective_reference_audit.json','reports/sestra_development/structures_parent/integration1/execution.json','reports/sestra_development/structures_parent/integration3/execution.json','reports/sestra_development/structures_parent/final_replay4/execution.json']:
            load(relative,evidence)
        if visual_directory:
            load(str(Path(visual_directory)/'provenance.json'),evidence)
        p('Independent integration checks pass 32 focused tests, followed by 13 focused checks covering the final reduced frame loads and serialization audit. The nine-case numerical replay retains every failed component and all native load balances pass. SESTRA MLG confirms the thick-shell override. Commercial agreement and force balance remain separate checks; neither supplies an independent shell exact solution.')
        p('Result-file load payloads can be rounded by SESTRA, for example 0.2 N to 0.200000003 N. A structure-only identity audit records each difference and its float32-epsilon relative bound with zero absolute allowance; record metadata, IDs, cases, zero entries and signs remain exact. Native source loads and all numerical comparison tolerances remain unchanged.')

    page('6 | Earlier analytical and invariant evidence')
    if historical:
        p(f'The retained beam-shell coverage report is dated {historical.get("generated_at")} and binds commit {historical.get("commit")}. It reports 128 PASS and 2 XFAIL entries. It is historical evidence for that version, not a rerun of this dirty checkout.')
        lookup={r['case_id']:r for r in historical['results']}
        selected=['BEAM-001','BEAM-002','BEAM-003','BEAM-005','BEAM-010','SHELL-004','SHELL-009']
        rows=[]
        for key in selected:
            c=lookup[key];result=c['result'];error=result.get('relative_error',result.get('max_relative_error'))
            if error is None and 'ratio_to_reference' in result:error=abs(result['ratio_to_reference']-1)
            rows.append([key,c['title'],fmt(100*error,5)+'%' if error is not None else 'Not quantified',c['status']])
        table(['Case','Analytical check','Recorded error','Status'],rows,[27*mm,92*mm,30*mm,25*mm])
        p('The Navier plate entry includes different element families and meshes: its displayed error is the maximum recorded family error, not a universal shell error. Beam extension, torsion and bending controls test elementary laws. Rigid-body, assembly symmetry, work, mass, patch and reaction tests support implementation correctness but are different evidence from physical accuracy.')
        sub('Literature names are not independent targets')
        p('The historical twisted-cantilever, Scordelis-Lo roof, pinched-cylinder and hemispherical-shell entries explicitly lack their bundled external literature targets. Their checks are internal refinement or symmetry. They must not be presented as completed literature-accuracy validations, even where the stored disposition is PASS. The hemispherical refinement response ratio is about 1.324, so a passed bookkeeping check does not imply mesh convergence.')
    if application:
        p(f'The separate ANYfem application report is dated {application.get("generated")} and records 21 passing checks. It includes analytical static, modal, buckling and dynamic controls plus meshing/load/project invariants. It belongs to the recorded application version and is supporting ecosystem evidence, not an interchangeable ANYfeSolver certification.')

    page('7 | Earlier external and wider-domain evidence')
    if external:
        p('A retained CalculiX report records actual execution of pressure-plate, beam-column buckling and cylindrical-shell cases. Its observable comparisons are primarily against analytical targets. Those are external-solver versus analytical checks, not automatically direct current ANYfeSolver versus CalculiX comparisons.')
        rows=[]
        for c in external['cases']:
            validation=c.get('validation',{})
            for v in validation.get('comparisons',[]):
                rows.append([c['name'],v.get('name',''),fmt(v.get('relative_error',0)*100,4)+'%' if v.get('relative_error') is not None else 'Absolute metric',str(v.get('status'))])
        table(['External case','Observable','Recorded error','Status'],rows,[49*mm,76*mm,28*mm,21*mm])
        p('The retained CalculiX version probe attempted --version as an input filename and returned an error; the report also identifies version 2.22 through its execution evidence. This provenance caveat is preserved. A separate canonical external-handoff report is not_executed and contributes no numerical accuracy evidence.')
    sub('Modal, buckling, material and workflow evidence')
    p('Earlier reports cover physical mass and inertia, free-free rigid modes, axial-bar frequency, Euler buckling, material-point response and element-tangent finite differences. They support those specific mechanics and consistency predicates. Plasticity convergence, nonlinear equilibrium, contact, dynamics and restart invariants require their own intended-use evidence; passing static SESTRA cases does not establish them.')
    if modal and buckling and plasticity:
        table(['Historical observable','Recorded value','Meaning'],[
            ['Axial-bar frequency',fmt(modal['modal_axial_bar']['frequency_hz'][0])+' Hz; relative error '+fmt(modal['modal_axial_bar']['relative_error']),'Analytical frequency at this configuration'],
            ['Free-free rigid modes',str(modal['free_free_modal']['num_rigid_body_modes'])+' modes; residual '+fmt(modal['free_free_modal']['diagnostics']['max_residual_norm']),'Nullspace/eigenpair invariant'],
            ['Euler critical load',fmt(buckling['euler_column']['critical_load_factor'])+' N; error '+fmt(100*buckling['euler_column']['relative_error'])+'%','Compared with Euler analytical load'],
            ['Material-point yield residual',fmt(plasticity['material_point']['max_abs_yield_residual']),'Constitutive consistency, not structural validation'],
            ['Maximum discrete tangent FD error',fmt(plasticity['element_tangents']['max_algorithmic_tangent_error']),'Derivative consistency for recorded material/element states']
        ],[53*mm,58*mm,63*mm])
    sub('Current source and historical binaries')
    p('Historical report commits, the current dirty source and an installed wheel are distinct identities. The report does not silently transfer a historical installed-artifact acceptance to modified source. Archived numerical results are included without replaying consumed qualification authority.')

    page('8 | Recent native and research work')
    table(['Route or programme','Retained conclusion and boundary'],[
        ['GE-B3 native delivery','Scoped opt-in delivery accepted for admitted straight/regular-curved workflows at its exact unpublished wheel and source identity. Covers declared beam section laws, retained inertia, recovery/restart and one owned Q4/S3 connection. Not a universal physical qualification or default replacement.'],
        ['E4-PL local synthesis','Q1AA transport, local algebra, support/KKT and recovery synthesis covers seven geometries, 56 numbered cases and 224 stations. Its terminal permits a later reviewed plan; assembled stability, locking and production activation are not established by this local synthesis.'],
        ['Improved-S4 research candidates','Registered candidates closed without a production selection. Numerical/test closeout passes preserve the NO_GO physical/release terminals; they are not successful candidate qualification.'],
        ['F03 variational line coupling','96 exploratory cases passed finite-model algebraic checks, while the physical response screen failed. Law and mortar sensitivity remained; F03/G0B stayed open. This distinction is directly relevant to local-force convergence.'],
        ['F03 shell spatial port','Bounded virgin linear spatial-map diagnostics passed. They do not establish nonlinear pose mechanics, a coupled stability result or production integration.'],
        ['Experimental stiffened-shell owners','Implemented GE/B2/BL5 and native material/restart workflows retain experimental, route-specific admission. Recovery/invariant coverage is not a substitute for physical torque, drilling, interface or general assembled qualification.']
    ],[51*mm,123*mm])
    p('These records document valuable work and expose unresolved scientific questions. Their original exact authorities and restrictions remain intact. No historical failure is removed to improve an overall pass percentage.')

    page('9 | Accuracy claims and remaining programme')
    sub('Claims supported by this evidence')
    p('For the measured standalone beam and analytical strip controls, the specified linear response agrees closely with independent formulas and the matched reference. For the tested coupled models, most of the earlier discrepancy is attributable to reference formulation and nodal response; remaining component failures and local-force mesh sensitivity are explicit.')
    sub('What still prevents a general statement')
    p('The evidence does not establish a solver-wide error bound. Remaining obligations include converged local beam/shear transfer, pointwise shell bending and membrane resultants in matched frames, broader boundary/load/section orientations, distorted and curved geometry, benchmark targets independent of ANYfeSolver, and current installed-artifact evidence. New nonlinear/dynamic claims need their own completed studies.')
    table(['Next evidence slice','Acceptance question'],[
        ['Local force convergence','Do common physical beam cuts converge with independently controlled beam and shell meshes, under matched reference formulations?'],
        ['Shell recovery','Do top/bottom bending and membrane resultants agree at matching stations and layers, instead of averaging bending away?'],
        ['Independent benchmarks','Are literature targets, analytical series or a second executed solver available with their assumptions and uncertainty?'],
        ['Reproducible release identity','Do the installed artifact, dependencies, source hashes and effective runtime reproduce the accepted cases?']
    ],[49*mm,125*mm])
    p('The practical regime is plate alone, isolated beam, then combined structures; analytical controls and whole-field/local-force refinements accompany every stage. Neither a commercial solver name nor a large test count removes the need for these checks.')

    page('Appendix A | Inputs, software and evidence identities')
    p('Machine-readable companion: ANYfeSolver_Accuracy_Report.evidence.json. It contains full SHA-256 values, file sizes, source commit and the input inventory used to generate this PDF. The PDF table uses hash prefixes for readability. Full raw values, commands, failures and logs remain in the referenced evidence directories.')
    unique={item['path']:item for item in evidence}
    rows=[]
    for item in unique.values():
        path=Path(item['path'])
        try:label=path.relative_to(ROOT).as_posix()
        except ValueError:label=path.as_posix()
        rows.append([label,item['sha256'][:16],str(item['bytes'])])
    table(['Evidence path relative to repository','SHA-256 prefix','Bytes'],rows,[117*mm,37*mm,20*mm])
    p('SESTRA runs: C:/DNV/Workspaces/ANYfeSolver_SESTRA_deep_20261008 and C:/DNV/Workspaces/ANYfeSolver_SESTRA_structures_20261008. Each executed case has its own hashed input/control/output and execution record. Failed comparisons remain evidence; unexecuted decks remain handoff artifacts.')
    p('References: DNV SESTRA user manual, ELOP control and THQS description; DNV SESAM Input and Results Interface Format for IDs, eccentricities, result cases, stations, frames and physical force conventions.',small=True)
    for label,url in [('SESTRA user manual','https://sesam.dnv.com/download/userdocumentation/sestra-um.pdf'),('SESAM input interface','https://sesam.dnv.com/download/windows64/sesam_input_interface_format.pdf'),('SESAM results interface','https://sesam.dnv.com/download/windows64/sesam_results_interface_format.pdf')]:
        p(f'<link href="{url}" color="#177c80">{label}</link>',small=True)
    manifest={'schema_version':1,'report_date_local':DATE,'report_scope':'linear-static external comparisons plus historical and route-specific evidence synthesis','git_head':commit,'dirty_source':True,'inputs':list(unique.values()),'structure_report':structure_report,'visual_directory':visual_directory,'comparison_loaded_sources':structure.get('loaded_sources') if structure else None,'comparison_runtime':structure.get('runtime') if structure else None,'comparison_harness':structure.get('harness') if structure else None,'report_builder':ident(__file__),'qualification_claim':False}
    (OUT/'ANYfeSolver_Accuracy_Report.evidence.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    pdf=OUT/'ANYfeSolver_Accuracy_Report.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(TEAL);canvas.setLineWidth(.5);canvas.line(18*mm,17*mm,192*mm,17*mm)
        canvas.setFont('Helvetica',7);canvas.setFillColor(INK);canvas.drawString(18*mm,12*mm,'ANYfeSolver | Accuracy evidence | '+DATE);canvas.drawRightString(192*mm,12*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf),pagesize=(210*mm,297*mm),rightMargin=18*mm,leftMargin=18*mm,topMargin=20*mm,bottomMargin=23*mm,title='ANYfeSolver Accuracy Evidence',author='ANYfeSolver verification',subject='Measured scope, reference comparisons and limitations')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'pdf':str(pdf),'inputs':len(unique),'structure_report':structure_report}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--structure-report');parser.add_argument('--visual-directory');args=parser.parse_args();build(args.structure_report,args.visual_directory)
