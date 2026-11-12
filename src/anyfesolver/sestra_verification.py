"""Development-only same-FEM linear membrane comparisons; interchange stays in ANYfileio."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

from .sesam_fem import import_sesam_fem, read_sesam_fem_document
from .sesam_fem.sif_importer import read_sesam_sif_stress

COMPONENTS = ("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX")
METRIC = "global_shell_membrane_arithmetic_station_mean"
RESULT_METADATA = frozenset({"HIERARCH", "RDRESREF", "RBLODCMB", "RDFORCES", "RDIELCOR",
    "RDNODBOC", "RDNODREA", "RDNODRES", "RDPOINTS", "RDSTRESS", "RSUMLOAD", "RSUMMASS",
    "RSUMREAC", "RVFORCES", "RVNODDIS", "RVNODREA", "RVSTRESS", "TDRESREF"})


class Unsupported(ValueError):
    """The model/result lies outside this explicitly admitted development slice."""


def generate_hanging_shell(path, *, overwrite=False):
    """Write the unchanged unit-height gravity fixture with complete SESAM records.

    Layouts follow the user-supplied successful GeniE FEM, with independent
    geometry/load values. ANYfileio owns numeric formatting and ASCII writing.
    No GeniE or SESTRA process is launched and no external acceptance is implied.
    """
    from anyfileio.sesam.records import FemRawRecord, canonical_record_lines
    from anyfileio.sesam.document import parse_sesam_fem_records
    from anyfileio.sesam.exporter import write_sesam_fem_document
    specs = [("IDENT",(1,1,3,0),()),("UNITS",(5,1,1,1,1),()),
             ("TDMATER",(4,1,104,0),("Steel",)),
             ("MISOSEL",(1,210000000000,.3,7850,0,.000012,1,355),()),
             ("GELTH",(1,.02),())]
    for nid,xyz in ((1,(0,0,0)),(2,(1,0,0)),(3,(1,1,0)),(4,(0,1,0))):
        specs.append(("GNODE",(nid,nid,6,123456),()))
        specs.append(("GCOORD",(nid,*xyz),()))
    specs.extend([("GELMNT1",(1,1,24,0,1,2,3,4),()),
                  ("GELREF1",(1,1,0,0,0,0,0,0,1,0,0,0),()),
                  ("BNBCD",(1,6,0,0,1,1,1,1),()),
                  ("BNBCD",(2,6,0,0,1,1,1,1),()),
                  ("BNBCD",(3,6,0,1,1,1,1,1),()),
                  ("BNBCD",(4,6,1,1,1,1,1,1),()),
                  ("TDLOAD",(4,1,103,0),("LC1",)),
                  ("BGRAV",(1,0,0,0,0,-10,0),()),("IEND",(0,0,0,0),())])
    records=[]
    for name,values,text in specs:
        numeric = FemRawRecord(name,tuple(map(float,values)),(),0,0,())
        lines=canonical_record_lines(numeric)+tuple(" "*8+x for x in text)
        records.append(FemRawRecord(name,numeric.numeric_fields,text,0,0,lines))
    document=parse_sesam_fem_records(records,strict=True)
    report=write_sesam_fem_document(document,path,mode="raw",overwrite=overwrite,newline="\r\n")
    validate_execution_fem(path)
    return report


def validate_execution_fem(path):
    """Refuse incomplete or newline-corrupted external inputs before launch."""
    from anyfileio.sesam.records import read_raw_records
    data=Path(path).read_bytes()
    if b"\r\r\n" in data or b"\n" in data.replace(b"\r\n",b"") or not data.endswith(b"\r\n"):
        raise Unsupported("external FEM must use single CRLF, without doubled CR or bare LF")
    try: data.decode("ascii")
    except UnicodeDecodeError as exc: raise Unsupported("external FEM must be ASCII") from exc
    records=read_raw_records(path,strict=True)
    by_name={}
    for r in records:
        _finite(r.numeric_fields,f"external {r.name}")
        by_name.setdefault(r.name,[]).append(r)
    lengths={"IDENT":4,"UNITS":5,"MISOSEL":8,"GNODE":4,"GCOORD":4,"GELMNT1":8,
             "GELREF1":12,"BNBCD":8,"TDLOAD":4,"BGRAV":7,"IEND":4}
    for name,length in lengths.items():
        if name not in by_name or any(len(r.numeric_fields)!=length for r in by_name[name]):
            raise Unsupported(f"external FEM requires complete {name} {length}-field records")
    if len(by_name["IDENT"])!=1 or by_name["IDENT"][0].numeric_fields!=(1.,1.,3.,0.):
        raise Unsupported("external IDENT layout must be 1 1 3 0")
    if by_name["UNITS"][0].numeric_fields!=(5.,1.,1.,1.,1.):
        raise Unsupported("external UNITS layout must be 5 1 1 1 1")
    if any(r.numeric_fields!=(0.,0.,0.,0.) for r in by_name["IEND"]):
        raise Unsupported("external IEND must contain four zeros")
    if any(not r.text_fields for r in by_name["TDLOAD"]):
        raise Unsupported("external TDLOAD requires explicit case name")
    d=read_sesam_fem_document(path,strict=True)
    node_ids=set(d.nodes)
    if {int(r.numeric_fields[0]) for r in by_name["GNODE"]} != node_ids:
        raise Unsupported("external GNODE coverage must match GCOORD")
    for r in by_name["GNODE"]:
        v=r.numeric_fields
        if v!=(v[0],v[0],6.,123456.):
            raise Unsupported("external GNODE requires identical IDs and six global DOFs")
    if {int(r.numeric_fields[0]) for r in by_name["GELREF1"]}!=set(d.elements):
        raise Unsupported("external GELREF1 must cover exactly every element")
    return d


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def _finite(values, context):
    if not all(math.isfinite(float(x)) for x in values):
        raise Unsupported(f"nonfinite {context}")


def _unadmitted_diagnostics(diagnostics):
    # ANYfileio document validation still emits this legacy warning even though
    # neutral semantics now implement BGRAV. Admission verifies its mapping below.
    return [x for x in diagnostics if not (x.code == "FEM120" and x.record_name == "BGRAV"
            and x.message == "BGRAV is preserved but not translated into solver semantics yet")]


def load_manifest(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get('profile')=='mixed_q4_beas15_constant_pressure':
        from .sestra_mixed_verification import validate_manifest
        return validate_manifest(m,path)
    required = {"schema_version", "case_id", "fem", "load_case", "units", "frame", "observable", "element_ids", "tolerances"}
    if not required <= m.keys() or m["schema_version"] != 1:
        raise Unsupported("manifest schema_version=1 and all contract fields required")
    if m["units"] != "SI" or m["frame"] != "GLOBAL" or m["observable"] != METRIC:
        raise Unsupported("only SI / GLOBAL / averaged shell membrane observable admitted")
    if type(m["load_case"]) is not int or m["load_case"] < 0:
        raise Unsupported("explicit nonnegative integer load_case required")
    ids = m["element_ids"]
    if not ids or any(type(i) is not int or i <= 0 for i in ids) or len(set(ids)) != len(ids):
        raise Unsupported("nonempty unique positive external element IDs required")
    t = m["tolerances"]
    if set(t) != {"absolute_pa", "relative"}:
        raise Unsupported("absolute_pa and relative tolerances required")
    _finite(t.values(), "tolerances")
    if any(type(v) not in (int, float) or v < 0 for v in t.values()):
        raise Unsupported("tolerances must be nonnegative finite numbers")
    for key in ("fem", "reference_sif"):
        if key in m:
            m[key] = str((path.parent / m[key]).resolve())
    if "execution" in m and "control_template" in m["execution"]:
        m["execution"]["control_template"] = str((path.parent / m["execution"]["control_template"]).resolve())
    m["manifest_identity"] = identity(path)
    return m


def admit(m):
    imported = import_sesam_fem(m["fem"], strict=True)
    d = imported.document
    if _unadmitted_diagnostics(imported.diagnostics):
        raise Unsupported("FEM diagnostics: " + "; ".join(str(x) for x in imported.diagnostics))
    allowed = {"IDENT", "DATE", "UNITS", "MISOSEL", "TDMATER", "GELTH", "GCOORD", "GNODE", "GELMNT1", "GELREF1", "BNBCD", "BGRAV", "TDLOAD", "IEND", "COMM"}
    unexpected = set(d.record_counts) - allowed
    if unexpected:
        raise Unsupported("unadmitted FEM records: " + ", ".join(sorted(unexpected)))
    if tuple(d.header.unit_values) not in {(1., 1., 1.), (5., 1., 1., 1., 1.)}:
        raise Unsupported("FEM UNITS must explicitly be 1 1 1 (manifest declares SI)")
    if d.dependencies or d.coordinate_transforms or d.unit_vectors:
        raise Unsupported("dependencies and coordinate transforms not admitted")
    if set(d.elements) != set(m["element_ids"]):
        raise Unsupported("manifest must request exactly all FEM element IDs")
    loads = [r for r in d.load_records if r.record_name != "TDLOAD"]
    for r in d.load_records:
        if r.record_name == "TDLOAD" and (len(r.raw_values) < 2 or r.raw_values[:2] != (4.,float(m["load_case"]))):
            raise Unsupported("TDLOAD metadata must use established field-count4 / selected case layout")
    if len(loads) != 1 or loads[0].record_name != "BGRAV":
        raise Unsupported("exactly one BGRAV load record required; all other loads refused")
    load = loads[0]
    if load.load_case_id != m["load_case"] or len(load.raw_values) != 7:
        raise Unsupported("BGRAV must have seven fields and match selected source case")
    _finite(load.raw_values, "BGRAV")
    if any(load.raw_values[1:4]):
        raise Unsupported("BGRAV selector fields must be zero in admitted layout")
    gravity = np.asarray(load.raw_values[-3:])
    if gravity[2] != 0 or not np.any(gravity[:2]):
        raise Unsupported("nonzero XY in-plane gravity required")
    if len(imported.model.load_cases) != 1 or not np.array_equal(imported.model.load_cases[0].gravity,gravity):
        raise Unsupported("native gravity mapping differs from admitted source record")
    for node in d.nodes.values():
        _finite(node.coordinates, "coordinates")
        if node.coordinate_system_id != 0:
            raise Unsupported("global nodes required")
    planes = set()
    for e in d.elements.values():
        ref = d.element_references.get(e.element_id)
        if e.type_code != 24 or ref is None or ref.material_id not in d.materials or ref.section_id not in d.sections:
            raise Unsupported("Q4 with explicit material and thickness required")
        if ref.transform_id or ref.nodal_transform_ids:
            raise Unsupported("element transforms not admitted")
        mat, sec = d.materials[ref.material_id], d.sections[ref.section_id]
        if len(ref.raw_values) > 3 and any(ref.raw_values[2:8]):
            raise Unsupported("GELREF1 offsets/release/other physical options not admitted")
        if mat.elastic_modulus is None or mat.poisson_ratio is None or mat.density is None or sec.thickness is None:
            raise Unsupported("complete isotropic material and thickness required")
        if len(mat.raw_values)<4 or mat.raw_values[1:4]!=(mat.elastic_modulus,mat.poisson_ratio,mat.density):
            raise Unsupported("MISOSEL declared E/nu/rho slots do not match parsed material; no heuristic defaults admitted")
        _finite((mat.elastic_modulus, mat.poisson_ratio, mat.density, sec.thickness), "material/thickness")
        if mat.elastic_modulus <= 0 or not -1 < mat.poisson_ratio < .5 or mat.density <= 0 or sec.thickness <= 0:
            raise Unsupported("invalid material or thickness")
        element = imported.model.mesh.elements[e.element_id]
        effective = imported.model.get_material(element.material_name)
        if (effective.elastic_modulus,effective.poisson_ratio,effective.density,element.thickness) != (mat.elastic_modulus,mat.poisson_ratio,mat.density,sec.thickness):
            raise Unsupported("effective native material/thickness differs from source")
        c = np.array([d.nodes[n].coordinates for n in e.node_ids])
        planes.update(c[:, 2])
        a, b = c[1]-c[0], c[3]-c[0]
        if not np.array_equal(c[2], c[0]+a+b) or a[1] != 0 or b[0] != 0 or a[0] <= 0 or b[1] <= 0:
            raise Unsupported("positive counterclockwise axis-aligned XY rectangles required")
    if len(planes) != 1 or len({e.material_id for e in d.elements.values()}) > 1:
        raise Unsupported("one plane and homogeneous material required")
    supports = {}
    for bc in d.boundaries:
        if len(bc.dof_flags) != 6 or any(x not in (0, 1) for x in bc.dof_flags) or any(bc.prescribed_values):
            raise Unsupported("binary zero-valued fixed supports required")
        supports.setdefault(bc.node_id, np.zeros(6, dtype=int))[:] |= np.asarray(bc.dof_flags, dtype=int)
    if any(n not in supports or not np.all(supports[n][2:]) for n in d.nodes):
        raise Unsupported("every node must restrain uz/rx/ry/rz for the membrane-only slice")
    if not any(np.any(v[:2]) for v in supports.values()):
        raise Unsupported("in-plane supports required")
    return imported


def native_stresses(imported):
    from .assembly import solve_linear
    model = imported.model
    if len(model.load_cases) != 1:
        raise Unsupported("adapter did not produce exactly one native load case")
    u, info = solve_linear(model, model.load_cases[0], constraint_mode="transformation")
    if info.get("convergence_info", {}).get("status") != "converged":
        raise Unsupported("native linear solve did not converge")
    _finite(u, "native displacements")
    tensors = {}
    for eid, element in model.mesh.elements.items():
        stress = element.compute_stresses(model.mesh, u, model.get_material(element.material_name), return_global=True)
        vals = []
        for component in ("xx", "yy", "zz", "xy", "yz", "xz"):
            top = np.asarray(stress[f"global_{component}_top"])
            bot = np.asarray(stress[f"global_{component}_bot"])
            if top.shape != bot.shape or top.size == 0:
                raise Unsupported("incomplete native stress stations")
            _finite(top, "native stresses"); _finite(bot, "native stresses")
            vals.append(float(np.mean(.5*(top+bot))))
        tensors[eid] = vals
    return tensors, {"displacement_norm": float(np.linalg.norm(u)), "solver_type": info["solver_type"],
                     "constraint_mode": info["constraint_mode"], "thread_policy": info.get("thread_policy")}


def reference_stresses(path, m, fem_document):
    d = read_sesam_fem_document(path, strict=True)
    diagnostics = [x for x in _unadmitted_diagnostics(d.diagnostics) if not
                   (x.code == "FEM110" and x.record_name in RESULT_METADATA and x.message == f"unknown FEM record preserved as raw data: {x.record_name}")]
    if diagnostics or any(r.name not in RESULT_METADATA for r in d.unknown_records):
        raise Unsupported("reference SIF has diagnostics or unknown records")
    if d.header.unit_values != fem_document.header.unit_values:
        raise Unsupported("reference units differ")
    if d.coordinate_transforms or d.unit_vectors or d.dependencies:
        raise Unsupported("reference coordinate/dependency semantics not admitted")
    precision=float(np.finfo(np.float32).eps)
    normalization=[]
    def metadata_equal(source,result,label):
        if source is None or result is None:
            raise Unsupported(f"missing reference/source {label}")
        _finite((source,result),label)
        equal = source==result if source==0 else abs(source-result)<=precision*abs(source)
        normalization.append({"field":label,"source":source,"reference":result,"rtol":precision,"atol":0.,"matched":bool(equal)})
        if not equal:raise Unsupported(f"reference {label} differs beyond float32 serialization precision")
    if set(d.materials)!=set(fem_document.materials) or set(d.sections)!=set(fem_document.sections):
        raise Unsupported("reference material/section IDs differ from FEM")
    for mid,source in fem_document.materials.items():
        reference=d.materials[mid]
        for field in ('elastic_modulus','poisson_ratio','density'):
            metadata_equal(getattr(source,field),getattr(reference,field),f'material:{mid}:{field}')
    for sid,source in fem_document.sections.items():
        reference=d.sections[sid]
        if source.kind!=reference.kind:raise Unsupported("reference section kind differs")
        metadata_equal(source.thickness,reference.thickness,f'section:{sid}:thickness')
    def refs(document):
        return {k:(v.material_id,v.section_id,v.transform_id,v.nodal_transform_ids,v.raw_values) for k,v in document.element_references.items()}
    if refs(d)!=refs(fem_document):raise Unsupported("reference element-reference identity differs from FEM")
    def without_zero_tail(values):
        values=list(values)
        while values and values[-1]==0:values.pop()
        return tuple(values)
    def supports(document):
        return sorted((x.node_id,x.dof_flags,without_zero_tail(x.prescribed_values)) for x in document.boundaries)
    if d.boundaries and supports(d)!=supports(fem_document):
        raise Unsupported("reference supports differ from FEM")
    def loads(document):
        result=[]
        for r in document.load_records:
            values=r.raw_values
            if r.record_name=='BGRAV' and len(values)==8 and values[-1]==0:values=values[:7]
            if r.record_name=='TDLOAD':
                if len(values)!=4 or values[:2]!=(4.,float(m['load_case'])):
                    raise Unsupported("reference TDLOAD source-case identity differs")
                text=()  # Case name is result metadata; numeric case binding stays exact.
            else:text=r.text
            result.append((r.record_name,values,text))
        return sorted(result)
    if d.load_records and loads(d)!=loads(fem_document):
        raise Unsupported("reference load contract differs from FEM")
    if d.nodes != fem_document.nodes:
        # GNODE raw values can differ in a result file; compare physical IDs/coordinates only.
        if {k: v.coordinates for k,v in d.nodes.items()} != {k: v.coordinates for k,v in fem_document.nodes.items()}:
            raise Unsupported("reference node IDs/coordinates differ from FEM")
    if {k:(v.type_code,v.node_ids) for k,v in d.elements.items()} != {k:(v.type_code,v.node_ids) for k,v in fem_document.elements.items()}:
        raise Unsupported("reference element IDs/types/connectivity differ from FEM")
    internal = {}
    for record in d.raw_records:
        _finite(record.numeric_fields, f"reference record {record.name}")
        if record.name == "GELMNT1":
            ext, ins = (int(v) for v in record.numeric_fields[:2])
            if ins in internal and internal[ins] != ext:
                raise Unsupported("ambiguous internal-to-external element identity")
            internal[ins] = ext
    seen = set()
    reported = bool(set(d.record_counts)&{'RDPOINTS','RDSTRESS'})
    for r in d.raw_records:
        if r.name != "RVSTRESS":
            continue
        v = r.numeric_fields
        if len(v) == 4 and v[0] == -4 and all(x == int(x) for x in v):
            continue  # Established RVSTRESS section inventory, not an element result.
        if len(v) < (8 if reported else 11) or any(v[i] != int(v[i]) for i in (1,2,4)):
            raise Unsupported("malformed RVSTRESS identity/payload")
        if int(v[1]) != m["load_case"]:
            continue
        eid = internal.get(int(v[2]), int(v[2]))
        if eid in seen or (not reported and (v[4] != 24 or len(v[5:]) % 6)):
            raise Unsupported("duplicate or unadmitted Q4 RVSTRESS layout")
        seen.add(eid)
    if seen != set(m["element_ids"]):
        raise Unsupported("reference result case missing or has wrong element IDs")
    result = read_sesam_sif_stress(path, load_case=m["load_case"])
    frames=getattr(result,'frame_provenance',None)
    if frames is not None and any(p['external_result_case']!=m['load_case'] for p in frames.values()):
        raise Unsupported("reported external result case differs from sole source case")
    if result.components != COMPONENTS or set(result.element_stress) != seen:
        raise Unsupported("incomplete reference observable coverage")
    m["reference_metadata"] = {"preserved_diagnostics":[x.as_dict() for x in d.diagnostics],
                               "record_counts":dict(d.record_counts),
                               "frame_provenance":frames,
                               "serialization_normalization":{"precision":"float32 epsilon","rtol":precision,"atol":0.,"property_pairs":normalization},
                               "uncompared_result_metadata":sorted((set(d.record_counts)&RESULT_METADATA)-{"RVSTRESS"})}
    return result.element_stress


def compare(candidate, reference, m):
    expected = set(m["element_ids"])
    if set(candidate) != expected or set(reference) != expected or not expected:
        raise Unsupported("exact and nonempty result IDs required")
    rows = []
    for eid in sorted(expected):
        a, b = np.asarray(candidate[eid]), np.asarray(reference[eid])
        if a.shape != (6,) or b.shape != (6,):
            raise Unsupported("all six tensor components required")
        _finite(a, "candidate tensor"); _finite(b, "reference tensor")
        limit = m["tolerances"]["absolute_pa"] + m["tolerances"]["relative"]*np.abs(b)
        for name, av, bv, tol in zip(COMPONENTS,a,b,limit):
            error = abs(float(av-bv))
            rows.append({"element_id": eid, "component": name, "native_pa": float(av), "reference_pa": float(bv),
                         "error_pa": error, "limit_pa": float(tol), "passed": bool(error <= tol)})
    return {"status": "passed" if all(r["passed"] for r in rows) else "failed", "compared_components": len(rows),
            "failed_components": sum(not r["passed"] for r in rows), "comparisons": rows}


def _kill_tree(process):
    import psutil
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            try: child.kill()
            except psutil.NoSuchProcess: pass
        try: root.kill()
        except psutil.NoSuchProcess: pass
        psutil.wait_procs(children+[root], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=5)


def execute_reference(m, case_dir):
    spec = m.get("execution", {})
    validate_execution_fem(m["fem"])
    argv = spec.get("argv")
    timeout = spec.get("timeout_seconds")
    prefix = spec.get("prefix")
    if not isinstance(argv, list) or not argv or any(not isinstance(s,str) for s in argv):
        raise Unsupported("execution requires explicit argv list, never a shell command")
    if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout <= 0:
        raise Unsupported("positive finite timeout_seconds required")
    if not isinstance(prefix,str) or not prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in prefix):
        raise Unsupported("safe explicit execution prefix required")
    executable = Path(argv[0]).resolve()
    if not executable.is_file():
        raise Unsupported("explicit executable does not exist")
    work = case_dir / "external"
    work.mkdir()  # must be new; no stale output can enter this execution
    fem = work / f"{prefix}T1.FEM"
    shutil.copyfile(m["fem"], fem)
    template = Path(spec["control_template"])
    token = spec.get("prefix_token", "{prefix}")
    content = template.read_text(encoding="ascii")
    if not token or token not in content:
        raise Unsupported("control template prefix token missing")
    control = work / "sestra.inp"
    control.write_text(content.replace(token,prefix), encoding="ascii")
    output = work / f"{prefix}R1.SIF"
    mapping = {"prefix":prefix,"fem":str(fem),"workdir":str(work)}
    try: command = [s.format_map(mapping) for s in argv]
    except (KeyError,ValueError) as exc: raise Unsupported(f"invalid argv placeholder: {exc}") from exc
    command[0] = str(executable)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_DISABLE_JIT="1")
    evidence = {"argv":command,"executable":identity(executable),"input":identity(fem),
                "control_template":identity(template),"control":identity(control), "cwd":str(work),
                "timeout_seconds":timeout, "runtime_environment":{k:env[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_DISABLE_JIT")}}
    start = time.perf_counter()
    with (work/"stdout.log").open("wb") as out, (work/"stderr.log").open("wb") as err:
        process = subprocess.Popen(command,cwd=work,env=env,stdout=out,stderr=err,shell=False)
        try:
            evidence["returncode"] = process.wait(timeout=timeout)
            evidence["status"] = "completed" if process.returncode == 0 else "process_failed"
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            evidence.update(status="timeout", returncode=process.returncode)
    evidence["process_wall_seconds"] = time.perf_counter()-start
    evidence["stdout"] = str(work/"stdout.log"); evidence["stderr"] = str(work/"stderr.log")
    if evidence["status"] == "completed" and (not output.is_file() or output.stat().st_size == 0):
        evidence["status"] = "missing_output"
    if output.is_file(): evidence["output"] = identity(output)
    evidence["retained_files"] = [identity(p) for p in sorted(work.iterdir()) if p.is_file()]
    evidence["budget_used_seconds"] = evidence["process_wall_seconds"]
    evidence["budget_remaining_seconds"] = max(0.,timeout-evidence["process_wall_seconds"])
    (work/"execution.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    return output, evidence


def run_verification(manifest, output_root, *, mode="prepare"):
    """Prepare, compare an existing SIF, or explicitly execute then compare.

    Every invocation writes a new immutable-by-convention development case directory.
    A passed report is only the declared averaged-membrane development comparison.
    """
    if mode not in {"prepare","compare","execute"}:
        raise ValueError("mode must be prepare, compare, or execute")
    root = Path(output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    case = root / uuid.uuid4().hex; case.mkdir()
    report = {"schema_version":1,"status":"not_executed","mode":mode,"qualification":False,
              "case_directory":str(case), "python":sys.version,"platform":platform.platform(),
              "source":identity(__file__), "dependencies":{},
              "sampling_limitation":"arithmetic station mean; no pointwise or bending acceptance"}
    for name in ("ANYfeSolver","ANYfileio","numpy","scipy"):
        try: report["dependencies"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report["dependencies"][name]="uninstalled source"
    from . import assembly, elements
    from .sesam_fem import importer
    from anyfileio.sesam import sif, document, semantics
    from threadpoolctl import threadpool_info
    report["loaded_sources"] = {module.__name__:identity(module.__file__) for module in (assembly,elements,importer,sif,document,semantics)}
    from . import e4_pl_element
    report['loaded_sources'][e4_pl_element.__name__]=identity(e4_pl_element.__file__)
    report["runtime"] = {"numba_disable_jit":os.environ.get("NUMBA_DISABLE_JIT"), "threadpools":threadpool_info()}
    start = time.perf_counter()
    try:
        m = load_manifest(manifest)
        report.update(case_id=m["case_id"],manifest=m["manifest_identity"],fem=identity(m["fem"]), contract=m)
        mixed=m.get('profile')=='mixed_q4_beas15_constant_pressure'
        if mixed:
            from .sestra_mixed_verification import admit_mixed, compare_mixed
            from . import sestra_mixed_verification
            report['loaded_sources'][sestra_mixed_verification.__name__]=identity(sestra_mixed_verification.__file__)
            imported=admit_mixed(m)
        else:imported = admit(m)
        report["admission"] = "admitted"
        report["preserved_diagnostics"] = [x.as_dict() for x in imported.diagnostics]
        if mode != "prepare":
            if mixed:
                if mode!='compare':raise Unsupported('mixed profile currently supports cached comparison only')
                report.update(compare_mixed(imported,m))
                return _save(report,case,start)
            candidate, native = native_stresses(imported)
            report["native"] = native
            report["native_tensors_pa"] = candidate
            if mode == "execute":
                reference_path, execution = execute_reference(m,case)
                report["execution"] = execution
                if execution["status"] != "completed":
                    report.update(status="failed",reason=execution["status"])
                    return _save(report,case,start)
            else:
                if "reference_sif" not in m: raise Unsupported("compare requires explicit reference_sif")
                if m.get("reference_fem_sha256") != report["fem"]["sha256"]:
                    raise Unsupported("imported reference requires explicit reference_fem_sha256 source binding")
                reference_path = Path(m["reference_sif"])
            report["reference"] = identity(reference_path)
            report["reference_origin"] = "fresh_execution" if mode == "execute" else "imported_existing_file"
            reference = reference_stresses(reference_path,m,imported.document)
            report["reference_metadata"] = m.pop("reference_metadata")
            report.update(compare(candidate,reference,m))
    except Unsupported as exc:
        report.update(status="unsupported",reason=str(exc))
    except Exception as exc:
        report.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
    return _save(report,case,start)


def _save(report, case, start):
    report["wall_seconds"] = time.perf_counter()-start
    report["report_path"] = str(case/"report.json")
    (case/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report
