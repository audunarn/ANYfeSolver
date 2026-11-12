"""Bounded, isolated G3b confirmation. No review means rehearsal only, never GO."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from hashlib import sha256
import importlib.util
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

# Isolated -I -S workers do not inherit the script directory on sys.path.
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ge_beam3_g3b_confirmation_authority as authority
import ge_beam3_g3b_environment as environment

ROOT=Path(__file__).resolve().parents[1]
THREADS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
         "BLIS_NUM_THREADS","NUMBA_NUM_THREADS","TBB_NUM_THREADS")
PACKETS={
    "mb2":("packet.json",4465,"f8b5caa5fa48931ca66688c7bdf36a362886bb5e0ed657af071ca35d8357623b"),
    "mb3":("mb3-packet.json",4675,"7bc0525083fb4be09744de3c1e12d29a48662149d22c6c6de550e6e7760c3796"),
    "mq4":("mq4-packet.json",13315,"88c30dbf8911cafd7c6d1ed0f0c38d34593b8a487236a4580ffd78e29a817397"),
    "ms3":("ms3-packet.json",9759,"da4e100fbd9bd15e1411dc732df1529f0d4b7902c7a30aefd1d7a7947192f895"),
    "weighted":("weighted-packet.json",13494,"d2e7f3694e1abe20a41a60aed428be64a3a42ece392ffecff90ea3c0d236c71d"),
    "transport":("transport-packet.json",198387,"8be4188c9febfe0be3f26d100f01870cbc077951505c769d78e115a6cc70765f"),
    "owner":("owner-packet.json",135368,"acee4a3a6350a99fa830e2007ce664e6386afe86036d00d1e436a6f4ba6ca49f")}


def write(path,value): environment.write(path,authority.canonical(value))


def isolated_startup():
    return bool(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and
                sys.pycache_prefix is not None and not Path(sys.pycache_prefix).exists())


def inputs(candidate):
    authority.hash_id(candidate,40)
    def git(*args): return environment.git(ROOT,*args).decode().strip()
    if git("rev-parse","HEAD")!=candidate or git("status","--porcelain","--untracked-files=all"):
        raise ValueError("wrong/dirty candidate")
    graft=Path(git("rev-parse","--git-path","info/grafts"))
    if not graft.is_absolute(): graft=ROOT/graft
    if graft.exists() and graft.stat().st_size: raise ValueError("Git grafts forbidden")
    files={}
    for name in sorted(git("ls-files").splitlines()):
        path=environment.regular(ROOT/name)
        raw=path.read_bytes()
        if path.suffix.lower() in (".py",".json",".md",".toml",".txt",".yml",".yaml"):
            raw=raw.replace(b"\r\n",b"\n")
        files[name]=dict(bytes=len(raw),sha256=sha256(raw).hexdigest())
    return dict(candidate=candidate,tree=git("rev-parse","HEAD^{tree}"),files=files,
                git=environment.git_identity())


def validate_lease(path,expected_hash):
    raw=environment.regular(path).read_bytes()
    if sha256(raw).hexdigest()!=expected_hash: raise ValueError("worker lease hash mismatch")
    lease=authority.strict(raw)
    if (type(lease) is not dict or set(lease)!={"schema","mode","cycle","lane","candidate","inputs",
            "environment","environment_sha256","review","review_sha256","inventory_sha256"} or
            lease["schema"]!="G3B_WORKER_LEASE_V1" or lease["mode"] not in ("REHEARSAL","FORMAL") or
            type(lease["cycle"]) is not int or lease["cycle"] not in (1,2) or lease["lane"] not in authority.LANES):
        raise ValueError("worker lease schema")
    inv_raw=authority.INVENTORY.read_bytes().replace(b"\r\n",b"\n")
    registered=authority.inventory(inv_raw)
    if lease["inventory_sha256"]!=sha256(inv_raw).hexdigest() or inputs(lease["candidate"])!=lease["inputs"]:
        raise ValueError("worker frozen inputs changed")
    frozen=environment.verify(Path(lease["environment"]),lease["environment_sha256"])
    if lease["mode"]=="FORMAL":
        authority.review(authority.canonical(lease["review"]),lease["review_sha256"],
            candidate=lease["candidate"],tree=lease["inputs"]["tree"],
            inputs_hash=sha256(authority.canonical(lease["inputs"])).hexdigest(),
            environment_hash=lease["environment_sha256"],inventory_hash=lease["inventory_sha256"])
    elif lease["review"] is not None or lease["review_sha256"] is not None:
        raise ValueError("rehearsal cannot borrow formal review authority")
    return lease,frozen,next(r for r in registered["lanes"] if r["lane"]==lease["lane"])


class Recorder:
    def __init__(self): self.collected=[]; self.reports=[]
    def pytest_collection_finish(self,session): self.collected=[item.nodeid for item in session.items]
    def pytest_runtest_logreport(self,report):
        self.reports.append(dict(node=report.nodeid,phase=report.when,outcome=report.outcome))


class ProgressStream:
    def __init__(self,target): self.target=target; self.pending=""; self.phases=set()
    def write(self,text):
        result=self.target.write(text); self.pending+=text
        while "\n" in self.pending:
            line,self.pending=self.pending.split("\n",1)
            prefix="G3B CHECKPOINT "
            if line.startswith(prefix): self.phases.add(line[len(prefix):].strip())
        return result
    def flush(self): return self.target.flush()
    def __getattr__(self,name): return getattr(self.target,name)


def import_origins(frozen,environment_path):
    site=environment_path.parent/"site"
    known={str((site/p).resolve()) for p in frozen["files"]}
    known.update(frozen["runtime"])
    known.update(str((ROOT/p).resolve()) for p in environment.git(ROOT,"ls-files").decode().splitlines())
    result={}
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,"__file__",None)
        if not filename or filename.startswith("<"): continue
        path=Path(filename).resolve()
        if str(path) not in known: raise ValueError("unbound import origin: "+name+" "+str(path))
        result[name]=dict(path=str(path),**environment.digest(path))
    return result


def install_import_guard(lease,frozen):
    site=Path(lease["environment"]).parent/"site"
    known={str((site/p).resolve()):(binding,False) for p,binding in frozen["files"].items()}
    known.update({str(Path(p).resolve()):(binding,False) for p,binding in frozen["runtime"].items()})
    known.update({str((ROOT/p).resolve()):(binding,True) for p,binding in lease["inputs"]["files"].items()})
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            found=importlib.machinery.PathFinder.find_spec(fullname,path,target)
            if found is not None and found.origin not in (None,"built-in","frozen"):
                origin=environment.regular(found.origin).resolve()
                item=known.get(str(origin))
                if item is None: raise ImportError("unbound import: "+fullname+" "+str(origin))
                expected,normalize=item
                raw=origin.read_bytes()
                if normalize and origin.suffix.lower()==".py": raw=raw.replace(b"\r\n",b"\n")
                if dict(bytes=len(raw),sha256=sha256(raw).hexdigest())!=expected:
                    raise ImportError("changed import: "+fullname)
            return found
    guard=Guard(); sys.meta_path.insert(0,guard)
    return guard


def worker(lease_path,lease_hash):
    if not isolated_startup(): raise ValueError("isolated -I -S worker required")
    lease,frozen,registered=validate_lease(lease_path,lease_hash)
    output=lease_path.parent
    write(output/"attempt.json",dict(lease_sha256=lease_hash))  # exclusive, never reusable
    sys.dont_write_bytecode=True; sys.pycache_prefix=str(output/"disabled-pycache")
    site=Path(lease["environment"]).parent/"site"
    sys.path[:0]=[str(ROOT/"src"),str(ROOT),str(site)]
    install_import_guard(lease,frozen)
    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTEST_ADDOPTS="",G3_PROGRESS="1",
        G3B_DIAGNOSTIC_PAYLOAD=str(output/"packet.json"),
        G3B_CORRECTION_RECORDS=str(output/"corrections"))
    for lane in ("mb3","mq4","ms3","weighted","transport","owner"):
        os.environ["G3B_"+lane.upper()+"_DIAGNOSTIC_PAYLOAD"]=str(output/PACKETS[lane][0])
    (output/"corrections").mkdir(exist_ok=False)
    import pytest
    from unittest.mock import patch
    from anyfesolver._ge_beam3_g3b_owner import MixedReferenceOwner
    # Install observation before owners capture dispatch; never change equations,
    # return values, guard calls or their identities during an accepted history.
    solve=MixedReferenceOwner.solve
    def observed(*args,**kwargs):
        result=solve(*args,**kwargs)
        print("G3B CHECKPOINT acceptance",flush=True)
        return result
    before=sys.stdout; stream=ProgressStream(before); sys.stdout=stream
    recorder=Recorder()
    try:
        with patch.object(MixedReferenceOwner,"solve",observed):
            code=pytest.main(["-vv","-s","-p","no:cacheprovider","--basetemp",str(output/"pytest"),
                              registered["test_file"]],plugins=[recorder])
        write(output/"tests.json",dict(lane=lease["lane"],collected=recorder.collected,
                                      reports=recorder.reports,exitcode=int(code)))
        write(output/"checkpoints.json",sorted(stream.phases))
        write(output/"imports.json",import_origins(frozen,Path(lease["environment"])))
        # Revalidate before returning terminal success; parent validates again.
        validate_lease(lease_path,lease_hash)
        return int(code)
    finally: sys.stdout=before


def process_module():
    spec=importlib.util.spec_from_file_location("g3b_process",ROOT/"docs/reference_cases/e4_pl_s3_v2_bounded_process.py")
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def run_child(directory,lease,deadline):
    write(directory/"lease.json",lease); lease_hash=environment.digest(directory/"lease.json")["sha256"]
    env={k:v for k,v in os.environ.items() if not k.startswith(("PYTHON","G3B_"))}
    for key in THREADS: env[key]="1"
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTEST_ADDOPTS="")
    job=process_module()._ProcessJob(24*1024**3)
    started=time.monotonic(); last=started; previous=None; status="FAILED"
    try:
        with (directory/"stdout.log").open("xb") as out,(directory/"stderr.log").open("xb") as err:
            process=job.launch([sys.executable,"-I","-S","-u","-B","-X",
                "pycache_prefix="+str(directory/"disabled-pycache"),str(Path(__file__).resolve()),
                "--worker",str(directory/"lease.json"),"--lease-sha256",lease_hash],
                cwd=ROOT,env=env,stdout=out,stderr=err)
            while True:
                cpu,active,peak=job.accounting(); now=time.monotonic()
                progress=(cpu,(directory/"stdout.log").stat().st_size,(directory/"stderr.log").stat().st_size)
                if progress!=previous: previous=progress; last=now
                if now>=deadline or now-started>=600 or now-last>=120 or peak>24*1024**3:
                    status="RESOURCE_BLOCKED"; job.terminate(); break
                if process.poll() is not None and active==0:
                    status="PASSED" if process.returncode==0 else "FAILED"; break
                time.sleep(.1)
            process.wait(timeout=15)
        record=dict(status=status,returncode=process.returncode,active_processes=job.accounting()[1],
            elapsed_seconds=time.monotonic()-started,peak_tree_bytes=peak,
            stdout_sha256=environment.digest(directory/"stdout.log")["sha256"],
            stderr_sha256=environment.digest(directory/"stderr.log")["sha256"])
        write(directory/"process.json",record)
        authority.process_record(authority.canonical(record))
        return record
    finally:
        if job.accounting()[1]: job.terminate()
        job.close()


def scientific(directory,registered):
    lane=registered["lane"]
    expected_files={"lease.json","attempt.json","stdout.log","stderr.log","process.json",
                    "tests.json","checkpoints.json","imports.json","corrections"}
    if lane in PACKETS: expected_files.add(PACKETS[lane][0])
    actual={p.name for p in directory.iterdir()}
    # pytest creates --basetemp lazily, only if a fixture needs it. It is an
    # optional diagnostic directory, never evidence of scientific coverage.
    if actual not in (expected_files,expected_files|{"pytest"}):
        raise ValueError("unexpected/missing lane output inventory")
    if "pytest" in actual and not (directory/"pytest").is_dir():
        raise ValueError("pytest diagnostic must be a directory")
    tests=(directory/"tests.json").read_bytes(); authority.lane_result(tests,registered)
    phases=authority.strict((directory/"checkpoints.json").read_bytes())
    needed={"capture","local solve","assembly","recovery","output"}
    if lane in ("owner","owner_corrections"):
        needed|={"native_prepared","other_prepared","acceptance","restart"}
    if type(phases) is not list or phases!=sorted(set(phases)) or not needed<=set(phases):
        raise ValueError("missing actual execution checkpoints")
    if lane=="owner_corrections":
        names=[sha256(n.encode()).hexdigest()+".json" for n in registered["nodes"]]
        if sorted(p.name for p in (directory/"corrections").iterdir())!=sorted(names):
            raise ValueError("correction case inventory mismatch")
        records=[]
        for name,node in zip(names,registered["nodes"]):
            raw=environment.regular(directory/"corrections"/name).read_bytes(); value=authority.strict(raw)
            if (set(value)!={"schema","node","qualification","observed"} or
                    value["schema"]!="G3B_OWNER_CORRECTION_CASE_V1" or value["node"]!=node or
                    value["qualification"] is not False or type(value["observed"]) is not dict):
                raise ValueError("correction case schema mismatch")
            observed=value["observed"]
            if "test_recovery_closure_mutation" in node or "test_entry_bookkeeping_rejection" in node:
                if (set(observed)!={"lane","boundary","target","accepted_before","accepted_after",
                                   "replay_after","factor_reused"} or observed["factor_reused"] is not True or
                        authority.canonical(observed["accepted_after"])!=authority.canonical(observed["replay_after"]) or
                        authority.canonical(observed["accepted_after"]["history"][:-1])!=authority.canonical(observed["accepted_before"]["history"]) or
                        observed["accepted_after"]["generation"]!=observed["accepted_before"]["generation"]+1):
                    raise ValueError("incomplete correction recovery observation")
            elif (set(observed)!={"lane","mutation","accepted","rejected_envelope","rejection_before_owner_construction"} or
                    observed["rejection_before_owner_construction"] is not True or
                    authority.canonical(observed["accepted"])==authority.canonical(observed["rejected_envelope"])):
                raise ValueError("incomplete correction preflight observation")
            records.append(dict(node=node,bytes=len(raw),sha256=sha256(raw).hexdigest()))
        packet=authority.canonical(records)
    else:
        if any((directory/"corrections").iterdir()): raise ValueError("unregistered correction record")
        name,size,expected=PACKETS[lane]
        packet=environment.regular(directory/name).read_bytes(); authority.strict(packet)
        if len(packet)!=size or sha256(packet).hexdigest()!=expected:
            raise ValueError("historical scientific packet changed")
    return authority.canonical(dict(lane=lane,tests=authority.strict(tests),checkpoints=phases,
        packet=dict(bytes=len(packet),sha256=sha256(packet).hexdigest())))


def publish(path,value):
    pending=path.with_name(path.name+".pending"); write(pending,value)
    if authority.strict(pending.read_bytes())!=value: raise ValueError("staged aggregate mismatch")
    os.link(pending,path)  # exclusive same-volume publication, never overwrite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",type=Path); p.add_argument("--lease-sha256")
    p.add_argument("--candidate"); p.add_argument("--environment",type=Path); p.add_argument("--environment-sha256")
    p.add_argument("--review",type=Path); p.add_argument("--review-sha256")
    p.add_argument("--rehearse",action="store_true"); p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.worker: return worker(args.worker,args.lease_sha256)
    if not isolated_startup():
        raise ValueError("coordinator requires -I -S -B and an empty private pycache prefix")
    if os.name!="nt" or not args.candidate or args.environment is None or args.output is None:
        p.error("Windows, candidate, environment and fresh external output required")
    output=args.output.absolute()
    if output.exists() or output.is_relative_to(Path("C:/Github")):
        raise ValueError("fresh external evidence directory required")
    frozen=inputs(args.candidate)
    environment.verify(args.environment,args.environment_sha256)
    inv_raw=authority.INVENTORY.read_bytes().replace(b"\r\n",b"\n"); inventory=authority.inventory(inv_raw)
    review=None
    if not args.rehearse:
        if args.review is None or args.review_sha256 is None: raise ValueError("accepted independent review required")
        review=authority.review(environment.regular(args.review).read_bytes(),args.review_sha256,
            candidate=args.candidate,tree=frozen["tree"],inputs_hash=sha256(authority.canonical(frozen)).hexdigest(),
            environment_hash=args.environment_sha256,inventory_hash=sha256(inv_raw).hexdigest())
    elif args.review is not None or args.review_sha256 is not None:
        raise ValueError("rehearsal is not formal authority")
    output.mkdir(parents=True,exist_ok=False); deadline=time.monotonic()+1800
    write(output/"inputs.json",frozen)
    cycles=[]
    try:
        for cycle in (1,2):
            results={}; cycle_dir=output/f"cycle-{cycle}"; cycle_dir.mkdir()
            for offset in range(0,len(inventory["lanes"]),3):
                if time.monotonic()>=deadline: raise TimeoutError("wave deadline")
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures={}
                    for registered in inventory["lanes"][offset:offset+3]:
                        lane=registered["lane"]; directory=cycle_dir/lane; directory.mkdir()
                        lease=dict(schema="G3B_WORKER_LEASE_V1",mode="REHEARSAL" if args.rehearse else "FORMAL",
                            cycle=cycle,lane=lane,candidate=args.candidate,inputs=frozen,
                            environment=str(args.environment.absolute()),environment_sha256=args.environment_sha256,
                            review=review,review_sha256=args.review_sha256,inventory_sha256=sha256(inv_raw).hexdigest())
                        print(f"START cycle {cycle} lane {lane}: {directory}",flush=True)
                        futures[pool.submit(run_child,directory,lease,deadline)]=(directory,registered)
                    for future in as_completed(futures):
                        directory,registered=futures[future]; future.result()
                        results[registered["lane"]]=scientific(directory,registered)
                        print(f"PASS cycle {cycle} lane {registered['lane']}",flush=True)
            cycles.append({lane:results[lane] for lane in authority.LANES})
            if inputs(args.candidate)!=frozen: raise ValueError("candidate changed during wave")
            environment.verify(args.environment,args.environment_sha256)
        authority.compare_cycles(*cycles)
        if time.monotonic()>=deadline: raise TimeoutError("wave deadline")
        value=dict(schema="GE_BEAM3_G3B_CONFIRMATION_V1",candidate=args.candidate,tree=frozen["tree"],
            inputs_sha256=sha256(authority.canonical(frozen)).hexdigest(),environment_sha256=args.environment_sha256,
            review_sha256=args.review_sha256,inventory_sha256=sha256(inv_raw).hexdigest(),scope=authority.SCOPE,
            cycles=2,lanes=[dict(lane=lane,scientific=authority.strict(cycles[0][lane])) for lane in authority.LANES],
            terminal="REHEARSAL_ONLY" if args.rehearse else "PROVISIONAL_GO_GE_BEAM3_G3B_REFERENCE_MIXED_ELASTIC_STATIC_ONLY",
            defaults_changed=False,general_static_parity=False)
        publish(output/("rehearsal.json" if args.rehearse else "aggregate.json"),value)
        print(value["terminal"],flush=True); return 0
    except BaseException as exc:
        write(output/"failure.json",dict(terminal="BLOCKED_G3B_PROCESS_OR_EVIDENCE",
            exception=type(exc).__name__,message=str(exc)))
        raise


if __name__=="__main__": raise SystemExit(main())
