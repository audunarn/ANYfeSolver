# Third-party notices

ANYfeSolver does not vendor or bundle these packages in its source distribution
or pure-Python wheel. They are installed separately by packaging tools and
remain governed by their own upstream licences. The machine-readable release
inventory is `dependency-licenses.json`.

| Dependency | Requirement | Licence | Distribution status |
| --- | --- | --- | --- |
| NumPy | `numpy>=2.0` | BSD-3-Clause and bundled-component licences | Runtime dependency |
| SciPy | `scipy>=1.13` | BSD-3-Clause and bundled-component licences | Runtime dependency |
| threadpoolctl | `threadpoolctl>=3.5` | BSD-3-Clause | Runtime dependency; not bundled |
| ANYmaterial | `ANYmaterial>=0.2,<0.3` | MPL-2.0 | Runtime dependency; separately distributed |
| ANYmesher | `ANYmesher>=0.4,<0.6` | MPL-2.0 | Runtime dependency; separately distributed |
| ANYfileio | `ANYfileio>=0.3.3,<0.4` | MPL-2.0 | Runtime dependency; separately distributed |
| ANYfeSolver-results | `ANYfeSolver-results==2.2.0a1` | MPL-2.0 | Runtime dependency |
| ANYhydro | `ANYhydro>=0.1.0` | MPL-2.0 | Optional marine excitation dependency |
| ANYloads | `ANYloads>=0.1.0` | MPL-2.0 | Optional marine Morison dependency |
| ANYgeometry | `ANYgeometry>=0.4.3,<0.5` | MPL-2.0 | Optional general-contact dependency; separately distributed |
| Numba | `numba>=0.59` | BSD | Optional accelerator; not bundled |
| pypardiso | `pypardiso>=0.4` | BSD-3-Clause | Optional accelerator; not bundled |
| mpmath | `mpmath==1.3.0` | BSD-3-Clause | Development/research dependency; not bundled |
| pytest | `pytest>=8` | MIT | Development dependency; not bundled |
| psutil | `psutil>=5.9` | BSD-3-Clause | Development resource-monitor dependency; not bundled |
| build | `build>=1.2` | MIT | Development dependency; not bundled |
| setuptools | `setuptools>=77` | MIT | Build and development dependency |
| Twine | `twine>=5` | Apache-2.0 | Development dependency; not bundled |

For complete licence texts and bundled-component notices, consult the exact
dependency distribution installed in the target environment. Earlier ANY
dependency releases retain the licence terms under which they were published;
the coordinated release bands above identify their MPL-2.0 successors.
