"""Independent exact roots for private endpoint-projection strips.

Direct execution avoids importing the solver for the bounded polynomial check;
ordinary pytest collection uses the same production module bytes.
"""
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
import sys

if __name__ == '__main__':
    _path = Path(__file__).parents[1]/'src/anyfesolver/general_contact/native_retained_body_projection.py'
    _spec = importlib.util.spec_from_file_location('body_projection_inventory',_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
else:
    from anyfesolver.general_contact import native_retained_body_projection as _module


class EndpointProjectionInventory(unittest.TestCase):
    def test_complete_junction_omission_accounting_and_refusal(self):
        import numpy as np
        bulk = np.array((1.,2.,3.))
        bound,total,boundary = _module.charged_junction_omission(
            bulk,4.,(.25,.5,.75),(.1,.2,.3),(10.,20.,30.))
        self.assertTrue(np.all(bound >= (1.,2.,7.)))
        self.assertTrue(np.all(total >= bound+(.25,.5,.75)))
        self.assertTrue(np.all(boundary >= bound+(.1,.2,.3)))
        with self.assertRaises(MemoryError):
            _module.charged_junction_omission(bulk,4.,(0.,0.,0.),(0.,0.,0.),(10.,20.,10.))
        for malformed in ((1.,2.,float('nan')),(-1.,2.,3.)):
            with self.assertRaises(ValueError):
                _module.charged_junction_omission(malformed,0.,(0.,0.,0.),(0.,0.,0.),(10.,20.,30.))

    def test_private_omission_keeps_legacy_midpoint_branch(self):
        import ast,inspect
        from anyfesolver.general_contact.native_retained_whole_pair import NativeRetainedWholeCurvePair
        import textwrap
        tree = ast.parse(textwrap.dedent(inspect.getsource(NativeRetainedWholeCurvePair.evaluate)))
        band_loop = next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                         and isinstance(node.iter,ast.Name) and node.iter.id=='junction_bands')
        private = next(node for node in band_loop.body if isinstance(node,ast.If)
                       and '_source_domains is not None' in ast.unparse(node.test))
        self.assertIsInstance(private.body[-1],ast.Continue)
        self.assertIn('charged_junction_omission',ast.unparse(private))
        tail = band_loop.body[band_loop.body.index(private)+1:]
        self.assertIn('integrand(midpoint, nominal_owner)',ast.unparse(ast.Module(body=tail,type_ignores=[])))

    @unittest.skipUnless(__import__('os').environ.get('CONTACT_NATIVE_BODY_CONTACT_ONLY') == '1',
                         'explicit bounded changed-production selector')
    def test_changed_curved_contact_evaluation_only(self):
        import os,json,time
        from dataclasses import replace
        started = time.monotonic()
        report_path = Path(os.environ['CONTACT_APP_RESULT_RECORD'])
        report = dict(phase='imports',fixture=dict(reference_rise_m=.001,separation=.101,body_b_shift_y_m=-.00102))
        def save():
            report['child_elapsed_s'] = time.monotonic()-started
            report_path.write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()),encoding='utf-8')
        save()
        import numpy as np
        from threadpoolctl import threadpool_info
        from examples.general_contact_native_retained_body_pair import analysis
        report.update(phase='construction',pools=threadpool_info(),numba_jit_disabled=os.environ['NUMBA_DISABLE_JIT'],
                      imports={n:str(Path(m.__file__).resolve()) for n,m in tuple(sys.modules.items())
                               if n.startswith('anyfesolver') and getattr(m,'__file__',None)})
        save()
        owner = analysis(separation=.101,reference_rise_m=.001)
        positions = owner.initial.mechanical.positions.copy()
        positions[5:,1] -= .00102
        pose = replace(owner.initial.mechanical,positions=positions)
        diagnostics = {}
        real_evaluate = owner.contact.evaluate
        # Capture the real caller's diagnostics without another evaluation.
        def captured_evaluate(mechanical):
            return real_evaluate(mechanical,_diagnostics=diagnostics)
        owner.contact.evaluate = captured_evaluate
        report.update(phase='actual_contact',mechanical=pose.descriptor(),
                      coordinate_count=owner.context.count,policy=owner.contact.identity,
                      limits=dict(search_cells=owner.contact.maximum_search_cells,
                                  integration_intervals=owner.contact.maximum_integration_intervals,
                                  global_error=owner.contact.tolerances))
        save()
        try:
            result = owner._contact(pose)
            force_error = float(np.linalg.norm(result.beam_a_force_n+result.beam_b_force_n)/.0025)
            moment_error = float(np.linalg.norm(result.beam_a_moment_nm+result.beam_b_moment_nm)/.005)
            report.update(phase='complete',diagnostics=diagnostics,potential_j=result.potential_j,
                          physical_gap_lower_m=result.conservative_lower_gap_m,
                          force_a=result.beam_a_force_n,force_b=result.beam_b_force_n,
                          moment_a=result.beam_a_moment_nm,moment_b=result.beam_b_moment_nm,
                          wrench_normalized_errors=(force_error,moment_error),
                          residual_shape=result.residual.shape,tangent_shape=result.tangent.shape,
                          native_component_bounds=(result.native_residual_component_error,
                                                   result.native_tangent_component_error,
                                                   result.native_spatial_tangent_component_error))
            save()
            self.assertEqual(result.residual.shape,(156,))
            self.assertEqual(result.tangent.shape,(156,156))
            self.assertTrue(np.all(np.isfinite(result.residual)))
            self.assertTrue(np.all(np.isfinite(result.tangent)))
            self.assertGreater(result.potential_j,0.)
            self.assertTrue(np.all(diagnostics['aggregate_error'] <= owner.contact.tolerances))
            self.assertLessEqual(max(force_error,moment_error),1e-6)
        except Exception as exc:
            report.update(phase='failure',failure=type(exc).__name__+': '+str(exc),diagnostics=diagnostics)
            save()
            raise

    def test_independent_near_endpoint_interior_jet(self):
        import numpy as np
        if __name__ == '__main__':
            path = _path.with_name('beam_curve_pair.py')
            spec = importlib.util.spec_from_file_location('body_projection_jet',path)
            beam = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = beam
            spec.loader.exec_module(beam)
        else:
            from anyfesolver.general_contact import beam_curve_pair as beam
        height, x = .09998, 5e-11
        target = np.array(((0.,height,0.),(.5,height,0.),(1.,height,0.)))
        source = np.array(((x,0.,0.),(.5+x,0.,0.),(1.+x,0.,0.)))
        t = _module.strict_projection_station(source,target,0.)
        self.assertGreater(t,0.)
        self.assertAlmostEqual(t,x,delta=1e-24)
        jet = beam._b3_distance_jet_at_station(source,target,0.,.1,t)
        self.assertEqual(jet.target_branch,'interior')
        self.assertAlmostEqual(jet.gap,height-.1,delta=1e-16)
        self.assertAlmostEqual(jet.gradient[0],0.,delta=1e-15)
        self.assertAlmostEqual(jet.gradient[1],-1.,delta=1e-15)
        self.assertAlmostEqual(jet.hessian[0,0],0.,delta=1e-12)
        # Independently, a target midpoint y perturbation changes its Q2
        # tangent slope by 4-8*t. At horizontal contact this is d(normal_x).
        self.assertAlmostEqual(jet.hessian[0,13],4.-8*t,delta=1e-12)
        self.assertAlmostEqual(jet.hessian[13,0],4.-8*t,delta=1e-12)
        # Legacy norm clustering chooses the endpoint at this binary64 pose;
        # it remains unchanged, and is not the oracle for the private jet.
        self.assertEqual(beam.b3_distance_jet(source,target,0.,.1).target_branch,'endpoint')
        source[:,0] -= 2*x
        t = _module.strict_projection_station(source,target,0.)
        self.assertEqual(t,0.)
        jet = beam._b3_distance_jet_at_station(source,target,0.,.1,t)
        d = float(np.hypot(x,height))
        self.assertEqual(jet.target_branch,'endpoint')
        self.assertAlmostEqual(jet.gradient[0],-x/d,delta=1e-15)
        self.assertAlmostEqual(jet.hessian[0,0],(1-(x/d)**2)/d,delta=1e-12)
        import os,json
        if os.environ.get('CONTACT_ENDPOINT_DERIVATIVE_RECORD'):
            from threadpoolctl import threadpool_info
            import scipy
            Path(os.environ['CONTACT_ENDPOINT_DERIVATIVE_RECORD']).write_text(json.dumps(
                dict(helper_path=str(Path(_module.__file__).resolve()),jet_path=str(Path(beam.__file__).resolve()),
                     numpy_path=np.__file__,scipy_path=scipy.__file__,pools=threadpool_info(),
                     numba_imported='numba' in sys.modules,
                     observations=dict(interior_station=x,mixed_H=4.-8*x,exterior_Hxx=jet.hessian[0,0])),indent=2),encoding='utf-8')

    def test_independent_straight_root_and_source_edge(self):
        # x(s)=3s-1; target x(t)=t. Phi endpoint roots are independently
        # s=1/3 and 2/3. Neither is exactly binary64 representable.
        source = ((-1.,.1,0.),(.5,.1,0.),(2.,.1,0.))
        target = ((0.,0.,0.),(.5,0.,0.),(1.,0.,0.))
        bands,count = _module.endpoint_projection_bands(source,target,0.,1.,maximum_cells=256)
        self.assertEqual(len(bands),2)
        self.assertLessEqual(count,256)
        for band,root in zip(bands,(Fraction(1,3),Fraction(2,3))):
            self.assertLessEqual(Fraction(band[0]),root)
            self.assertGreaterEqual(Fraction(band[1]),root)
            self.assertLess(band[1]-band[0],1e-14)
        # Exact endpoint root has a nonempty inward strip, not a cut alone.
        edge = ((0.,.1,0.),(.5,.1,0.),(1.,.1,0.))
        bands,_ = _module.endpoint_projection_bands(edge,target,0.,1.,maximum_cells=256)
        self.assertEqual(bands[0][0],0.)
        self.assertGreater(bands[0][1],0.)
        self.assertEqual(bands[-1][1],1.)

    def test_quadratic_two_roots_and_tangent_refusal(self):
        target = ((0.,0.,0.),(.5,0.,0.),(1.,0.,0.))
        # x(s)=(s-1/4)(s-3/4); both t=0 roots must be inventoried.
        source = ((.1875,.1,0.),(-.0625,.1,0.),(.1875,.1,0.))
        bands,_ = _module.endpoint_projection_bands(source,target,0.,1.,maximum_cells=256)
        self.assertEqual(len(bands),2)
        for band,root in zip(bands,(.25,.75)):
            self.assertLessEqual(band[0],root)
            self.assertGreaterEqual(band[1],root)
        tangent = ((.25,.1,0.),(0.,.1,0.),(.25,.1,0.))
        with self.assertRaisesRegex(ValueError,'tangent'):
            _module.endpoint_projection_bands(tangent,target,0.,1.,maximum_cells=256)
        plateau = ((0.,.1,0.),(0.,.2,0.),(0.,.3,0.))
        with self.assertRaisesRegex(ValueError,'plateau'):
            _module.endpoint_projection_bands(plateau,target,0.,1.,maximum_cells=256)
        with self.assertRaises(MemoryError):
            _module.endpoint_projection_bands(source,target,0.,1.,maximum_cells=0)


if __name__ == '__main__':
    unittest.main()
