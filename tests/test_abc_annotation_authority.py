"""Cold-import annotation observation preserves exact qualified authority."""

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("family", ["q4", "s3"])
@pytest.mark.parametrize("observer_first", [False, True])
def test_cold_abc_annotations_and_later_mutations(family, observer_first, tmp_path):
    # pytest or its plugins may already have observed ABC in this process.
    # A fresh interpreter exercises both observer/solver import orders.
    code = r'''
from abc import ABC
import inspect,json,sys
observer_first = sys.argv[3] == "1"
if observer_first: assert inspect.get_annotations(ABC) == {}
import anyfesolver
from pathlib import Path
import os
from anyfesolver.current_state_tangent import require_exact_qualified_component_lifecycle_api as guard
from anyfesolver.element_capabilities import ElementCapabilityError

family = sys.argv[1]
record_path = Path(sys.argv[2])
entry = {"family":family,"solver":anyfesolver.__file__,"observer_first":observer_first}
record_path.write_text(json.dumps(entry))
assert Path(anyfesolver.__file__).is_relative_to(Path(os.environ["ANYSOLVER_DEVELOPMENT_SOURCE"]))
owner = anyfesolver.QualifiedE4PLShellElement if family == "q4" else anyfesolver.QualifiedE4PLS3ShellElement
points = ((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)) if family == "q4" else ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
model = anyfesolver.FEModel("cold-abc-"+family)
model.add_material("elastic",1e5,.3)
for node,point in enumerate(points,1): model.add_node(node,*point)
model.add_element(1,owner(1,tuple(range(1,len(points)+1)),"elastic",thickness=.05,reference_normal=(0.,0.,1.)))
before = dict(type.__getattribute__(ABC,"__dict__"))
observed = inspect.get_annotations(ABC)
after = dict(type.__getattribute__(ABC,"__dict__"))
entry.update(added=sorted(set(after)-set(before)),removed=sorted(set(before)-set(after)),
             rebound=[key for key in before if key in after and after[key] is not before[key]])
record_path.write_text(json.dumps(entry))
assert not set(after).symmetric_difference(before)
assert all(after[key] is value for key,value in before.items())
assert guard(model,context="cold ABC annotation") == {"qualified_element_ids":[1],"guarded":True}
cache = "__annotations_cache__" if "__annotations_cache__" in after else "__annotations__"
invoked = []
def forbidden_annotation(*args,**kwargs):
    invoked.append(True)
    raise AssertionError("annotation callback executed")
refused = []
for name,value in ((cache,{"forged":True}),("__annotate_func__",forbidden_annotation),("__annotations_cache___",{})):
    old = dict(type.__getattribute__(ABC,"__dict__"))
    setattr(ABC,name,value)
    try:
        try: guard(model,context="cold ABC mutation")
        except ElementCapabilityError as exc:
            assert "ABC."+name in str(exc),str(exc)
            refused.append(name)
        else: raise AssertionError("ABC mutation admitted: "+name)
    finally:
        if name in old: setattr(ABC,name,old[name])
        else: delattr(ABC,name)
    assert guard(model,context="cold ABC restored") == {"qualified_element_ids":[1],"guarded":True}
assert not invoked
entry.update(observation=observed,refused=refused,callback_calls=len(invoked))
record_path.write_text(json.dumps(entry))
print(json.dumps(entry))
'''
    expected = Path(os.environ.get("ANYSOLVER_DEVELOPMENT_SOURCE", Path(__file__).parents[1]/"src/anyfesolver"))
    environment = os.environ.copy()
    environment["ANYSOLVER_DEVELOPMENT_SOURCE"] = str(expected)
    environment["PYTHONPATH"] = str(expected.parent)+os.pathsep+environment.get("PYTHONPATH", "")
    run = subprocess.run([sys.executable, "-X", "utf8", "-c", code, family, str(tmp_path/"entry.json"), str(int(observer_first))],
                         env=environment, capture_output=True, text=True,
                         timeout=30)
    (tmp_path/"child.stdout.log").write_text(run.stdout)
    (tmp_path/"child.stderr.log").write_text(run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    record = json.loads(run.stdout)
    assert Path(record["solver"]).is_relative_to(expected)
    assert len(record["refused"]) == 3 and record["callback_calls"] == 0


@pytest.mark.parametrize("attribute", ["__annotate__", "__annotate_func__"])
def test_preimport_annotation_callback_is_rejected_without_invocation(attribute, tmp_path):
    expected = Path(os.environ.get("ANYSOLVER_DEVELOPMENT_SOURCE", Path(__file__).parents[1]/"src/anyfesolver"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(expected.parent)+os.pathsep+environment.get("PYTHONPATH", "")
    code = r'''
from abc import ABC
import importlib.util,json,sys
from pathlib import Path
assert Path(importlib.util.find_spec("anyfesolver").origin).is_relative_to(Path(sys.argv[2]))
calls = []
def forbidden(*args,**kwargs):
    calls.append(True)
    raise AssertionError("preimport annotation callback executed")
setattr(ABC,sys.argv[1],forbidden)
try:
    import anyfesolver
except RuntimeError as exc:
    assert "ABC annotation callback is incompatible" in str(exc),str(exc)
else:
    raise AssertionError("preimport callback admitted")
assert not calls
print(json.dumps({"attribute":sys.argv[1],"callback_calls":len(calls)}))
'''
    run = subprocess.run([sys.executable,"-X","utf8","-c",code,attribute,str(expected)],
                         env=environment,capture_output=True,text=True,timeout=30)
    (tmp_path/"child.stdout.log").write_text(run.stdout)
    (tmp_path/"child.stderr.log").write_text(run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr
    assert json.loads(run.stdout)["callback_calls"] == 0
