"""Exact bounded reuse of the pure angular Taylor kernel."""
from decimal import (Decimal, Inexact, InvalidOperation, localcontext,
                     ROUND_DOWN, ROUND_HALF_EVEN)
from math import pi
from pathlib import Path
import pytest
from anyfesolver.general_contact import native_normal_angles as angles


def _reference_sincos(angle):
    """The original unshared Taylor loop, embedded as the parity oracle."""
    x = Decimal(angle)
    sine = sine_term = x
    cosine = cosine_term = Decimal(1)
    square = -x*x
    for k in range(1, 81):
        sine_term *= square/Decimal(2*k*(2*k+1))
        cosine_term *= square/Decimal((2*k-1)*2*k)
        sine += sine_term
        cosine += cosine_term
    return sine, cosine


def _corpus():
    values = [0.0, 2*pi, -2*pi, pi/2, 0.0491/.05, 1e-20, -1e-20,
              6.283185307179586, 1.5707963267948966, -0.3333333333333333,
              5.0e-324, 0.982, 2.0943951023931953]
    values += [Decimal('1.5'), Decimal('1.50'), Decimal('-0.0491'),
               Decimal('1e-40'), Decimal(pi), Decimal(2)*Decimal(pi)/3,
               Decimal('0.98200000000000000000000000000000000000000000000000')]
    return values


def test_sincos_reuse_is_bit_exact_and_repeats_return_same_tuple():
    angles._sincos_cache.clear()
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        for angle in _corpus():
            first = angles._sincos(angle)
            reference = _reference_sincos(angle)
            assert first[0].as_tuple() == reference[0].as_tuple()
            assert first[1].as_tuple() == reference[1].as_tuple()
            assert angles._sincos(angle) is first


def test_sincos_cache_key_carries_the_effective_context():
    angles._sincos_cache.clear()
    probe = 0.3333333333333333
    with localcontext() as ctx:
        ctx.prec = 50
        base = angles._sincos(probe)
    with localcontext() as ctx:
        ctx.prec = 28
        low = angles._sincos(probe)
        reference = _reference_sincos(probe)
        assert low[0].as_tuple() == reference[0].as_tuple()
        assert low[1].as_tuple() == reference[1].as_tuple()
        assert low[0] != base[0]
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_DOWN
        down = angles._sincos(probe)
        reference = _reference_sincos(probe)
        assert down[0].as_tuple() == reference[0].as_tuple()
        assert down[1].as_tuple() == reference[1].as_tuple()
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.clamp = 1
        clamped = angles._sincos(probe)
        reference = _reference_sincos(probe)
        assert clamped[0].as_tuple() == reference[0].as_tuple()
        assert clamped[1].as_tuple() == reference[1].as_tuple()
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.traps[Inexact] = True
        before = len(angles._sincos_cache)
        with pytest.raises(Inexact):
            angles._sincos(probe)
        assert len(angles._sincos_cache) == before


def test_context_signature_reflects_every_effective_setting():
    with localcontext() as ctx:
        ctx.prec = 50
        base = angles._context_signature()
        ctx.prec = 28
        assert angles._context_signature() != base
        ctx.prec = 50
        ctx.rounding = ROUND_DOWN
        assert angles._context_signature() != base
        ctx.rounding = ROUND_HALF_EVEN
        ctx.clamp = 1
        assert angles._context_signature() != base
        ctx.clamp = 0
        ctx.Emin = -60
        assert angles._context_signature() != base
        ctx.Emin = -999999
        ctx.Emax = 60
        assert angles._context_signature() != base
        ctx.Emax = 999999
        ctx.capitals = 0
        assert angles._context_signature() != base
        ctx.capitals = 1
        ctx.traps[Inexact] = True
        assert angles._context_signature() != base
        ctx.traps[Inexact] = False
        assert angles._context_signature() == base


def test_sincos_cache_stays_bounded_and_survives_clear():
    angles._sincos_cache.clear()
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        for i in range(angles._SINCOS_CACHE_LIMIT + 37):
            angles._sincos(Decimal(i)/Decimal(3))
        assert len(angles._sincos_cache) <= angles._SINCOS_CACHE_LIMIT
        value = angles._sincos(Decimal(7)/Decimal(3))
        reference = _reference_sincos(Decimal(7)/Decimal(3))
        assert value[0].as_tuple() == reference[0].as_tuple()
        assert value[1].as_tuple() == reference[1].as_tuple()


def test_precise_normal_angles_value_parity_with_reference_kernel():
    angles._sincos_cache.clear()
    field = angles.PreciseNormalAngles(.0491, 0., .8, .05, 0.)
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        for angle in (0.3, Decimal(pi)/7, 2*pi, 0.982):
            sine, cosine = _reference_sincos(angle)
            expected = field.offset+field.cosine*cosine+field.sine*sine
            assert field.value(angle).as_tuple() == expected.as_tuple()


def _raised_flags(context):
    return frozenset(flag for flag, value in context.flags.items() if value)


def test_sincos_hit_reproduces_original_context_flags():
    angles._sincos_cache.clear()
    probe = Decimal('0.3333333333333333')
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        ctx.clear_flags()
        warm = angles._sincos(probe)
        warm_flags = _raised_flags(ctx)
        ctx.clear_flags()
        hit = angles._sincos(probe)
        hit_flags = _raised_flags(ctx)
        ctx.clear_flags()
        fresh = _reference_sincos(probe)
        fresh_flags = _raised_flags(ctx)
    assert warm_flags == fresh_flags
    assert hit_flags == fresh_flags
    assert hit_flags
    assert hit is warm
    assert hit[0].as_tuple() == fresh[0].as_tuple()
    assert hit[1].as_tuple() == fresh[1].as_tuple()


def test_sincos_hit_preserves_preexisting_unrelated_flag():
    angles._sincos_cache.clear()
    probe = Decimal('0.982')
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        ctx.clear_flags()
        angles._sincos(probe)
        ctx.clear_flags()
        ctx.flags[InvalidOperation] = True
        hit = angles._sincos(probe)
        hit_flags = _raised_flags(ctx)
        ctx.clear_flags()
        ctx.flags[InvalidOperation] = True
        fresh = _reference_sincos(probe)
        fresh_flags = _raised_flags(ctx)
    assert InvalidOperation in hit_flags
    assert hit_flags == fresh_flags
    assert hit[0].as_tuple() == fresh[0].as_tuple()
    assert hit[1].as_tuple() == fresh[1].as_tuple()


def test_sincos_trapped_raise_preserves_flags_and_stores_no_entry():
    angles._sincos_cache.clear()
    probe = Decimal('0.3333333333333333')
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        ctx.traps[Inexact] = True
        before = len(angles._sincos_cache)
        ctx.clear_flags()
        with pytest.raises(Inexact):
            angles._sincos(probe)
        trapped_flags = _raised_flags(ctx)
        assert Inexact in trapped_flags
        assert len(angles._sincos_cache) == before
        ctx.clear_flags()
        with pytest.raises(Inexact):
            _reference_sincos(probe)
        assert _raised_flags(ctx) == trapped_flags
        ctx.clear_flags()
        with pytest.raises(Inexact):
            angles._sincos(probe)
        assert _raised_flags(ctx) == trapped_flags
        assert len(angles._sincos_cache) == before


def test_sincos_hit_sets_full_loop_flags_after_dirty_warm():
    angles._sincos_cache.clear()
    probe = Decimal('0.3333333333333333')
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        ctx.clear_flags()
        fresh = _reference_sincos(probe)
        fresh_flags = _raised_flags(ctx)
        assert fresh_flags
        ctx.clear_flags()
        for flag in fresh_flags:
            ctx.flags[flag] = True
        warm = angles._sincos(probe)
        assert _raised_flags(ctx) == fresh_flags
        ctx.clear_flags()
        hit = angles._sincos(probe)
        hit_flags = _raised_flags(ctx)
    assert hit_flags == fresh_flags
    assert hit is warm
    assert hit[0].as_tuple() == fresh[0].as_tuple()
    assert hit[1].as_tuple() == fresh[1].as_tuple()


def test_sincos_early_completion_parity_across_contexts():
    from decimal import (ROUND_DOWN, ROUND_HALF_DOWN, ROUND_HALF_UP,
                         ROUND_UP)
    angles._sincos_cache.clear()
    probes = [0.0, 1e-20, 1e-10, 0.001, 0.25, Decimal(1)/Decimal(3), 0.982,
              1.5, 2.0943951023931953, Decimal(pi), 6.283185307179586,
              -0.3333333333333333, -6.283185307179586, Decimal('1e-40'),
              Decimal('1e-73'), 5.0e-324]
    variants = [{}, {'prec': 28}, {'prec': 9}, {'prec': 1},
                {'rounding': ROUND_HALF_UP}, {'rounding': ROUND_HALF_DOWN},
                {'rounding': ROUND_UP}, {'rounding': ROUND_DOWN},
                {'Emin': -100}, {'Emin': -1000}, {'Emax': 9}, {'clamp': 1},
                {'prec': 9, 'Emin': -40}, {'prec': 28, 'Emin': -200}]
    for variant in variants:
        with localcontext() as ctx:
            ctx.prec = angles.PRECISION
            for name, value in variant.items():
                setattr(ctx, name, value)
            for probe in probes:
                angles._sincos_cache.clear()
                ctx.clear_flags()
                produced = angles._sincos(probe)
                produced_flags = _raised_flags(ctx)
                ctx.clear_flags()
                reference = _reference_sincos(probe)
                reference_flags = _raised_flags(ctx)
                assert produced[0].as_tuple() == reference[0].as_tuple(), (variant, probe)
                assert produced[1].as_tuple() == reference[1].as_tuple(), (variant, probe)
                assert produced_flags == reference_flags, (variant, probe)


def test_sincos_early_completion_respects_underflow_trap():
    from decimal import Underflow
    angles._sincos_cache.clear()
    probe = Decimal('1e-40')
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        ctx.Emin = -1000
        ctx.traps[Underflow] = True
        before = len(angles._sincos_cache)
        ctx.clear_flags()
        with pytest.raises(Underflow):
            angles._sincos(probe)
        trapped_flags = _raised_flags(ctx)
        assert len(angles._sincos_cache) == before
        ctx.clear_flags()
        with pytest.raises(Underflow):
            _reference_sincos(probe)
        assert _raised_flags(ctx) == trapped_flags


def test_certified_early_completion_engages_and_directed_modes_never_certify():
    from decimal import ROUND_UP
    angles._sincos_cache.clear()
    small = Decimal('0.001')
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        square = -small*small
        plan = angles._certification_plan(ctx, abs(square))
        assert plan is not None
        sine = sine_term = small
        cosine = cosine_term = Decimal(1)
        engaged = None
        for k in range(1, 81):
            sine_term *= square/Decimal(2*k*(2*k+1))
            cosine_term *= square/Decimal((2*k-1)*2*k)
            sine += sine_term
            cosine += cosine_term
            if (k >= 8 and not k & 3 and angles._tail_certified(
                    ctx, k, abs(square), plan, sine_term, cosine_term,
                    sine, cosine)):
                engaged = k
                break
        assert engaged is not None and engaged < 80
        finished_sine, finished_cosine = sine, cosine
        for k in range(engaged+1, 81):
            sine_term *= square/Decimal(2*k*(2*k+1))
            cosine_term *= square/Decimal((2*k-1)*2*k)
            finished_sine += sine_term
            finished_cosine += cosine_term
        assert finished_sine.as_tuple() == sine.as_tuple()
        assert finished_cosine.as_tuple() == cosine.as_tuple()
        ctx.rounding = ROUND_UP
        assert angles._certification_plan(ctx, abs(square)) is None
        ctx.rounding = ROUND_HALF_EVEN
        assert angles._certification_plan(ctx, Decimal('NaN')) is None
        ctx.clear_flags()
        assert not angles._tail_certified(ctx, 12, abs(square), plan,
                                           sine_term, cosine_term, sine, cosine)


def test_fresh_source_import_raises_no_flags_or_traps_in_adversarial_contexts():
    from decimal import (Clamped, DivisionByZero, FloatOperation, Overflow,
                         Rounded, Subnormal, Underflow)
    source = (Path(__file__).resolve().parents[1]/'src'/'anyfesolver'
              /'general_contact'/'native_normal_angles.py').read_text(encoding='utf-8')
    all_traps = (Inexact, Rounded, Underflow, Subnormal, Clamped, Overflow,
                 DivisionByZero, InvalidOperation, FloatOperation)
    variants = [{'prec': 1, 'traps': (Inexact, Rounded)},
                {'prec': 1, 'traps': all_traps},
                {'prec': 2, 'traps': (Inexact, Rounded)},
                {'prec': 28, 'traps': all_traps},
                {'prec': 1, 'Emin': -40, 'Emax': 9, 'traps': all_traps}]
    for variant in variants:
        with localcontext() as ctx:
            ctx.prec = variant['prec']
            if 'Emin' in variant:
                ctx.Emin = variant['Emin']
            if 'Emax' in variant:
                ctx.Emax = variant['Emax']
            for trap in variant['traps']:
                ctx.traps[trap] = True
            ctx.clear_flags()
            namespace = {}
            exec(compile(source, 'fresh_native_normal_angles', 'exec'), namespace)
            assert _raised_flags(ctx) == frozenset(), variant
            bounds = namespace['_TAIL_SHRINK_BOUNDS']
            for k in range(1, 81):
                integer = (2*k+1)*(2*k+2)
                expected = (0, tuple(int(digit) for digit in str(integer)), -1)
                assert bounds[k-1].as_tuple() == expected, (variant, k)
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        for k in range(1, 81):
            assert angles._TAIL_SHRINK_BOUNDS[k-1] == Decimal((2*k+1)*(2*k+2))/10, k


def test_tail_certification_raises_no_flags_or_traps_beyond_the_loop():
    from decimal import (Clamped, ROUND_HALF_EVEN, Rounded, Subnormal,
                         Underflow)
    angles._sincos_cache.clear()
    with localcontext() as ctx:
        ctx.prec = 9
        ctx.Emin = -40
        ctx.rounding = ROUND_HALF_EVEN
        magnitude = Decimal('1e-30')
        plan = angles._certification_plan(ctx, magnitude)
        assert plan is not None
        # Negligible subnormal terms below the remaining-normal floor: the
        # certification must refuse before any threshold product can go
        # subnormal and raise flags or fire traps the loop would not.
        sine_term = cosine_term = Decimal('1e-45')
        sine = cosine = Decimal('1.23456789e-33')
        for trap in (Underflow, Subnormal, Clamped):
            ctx.traps[trap] = True
        ctx.clear_flags()
        ctx.flags[Inexact] = True
        ctx.flags[Rounded] = True
        assert not angles._tail_certified(ctx, 8, magnitude, plan,
                                          sine_term, cosine_term, sine, cosine)
        assert _raised_flags(ctx) == frozenset((Inexact, Rounded))
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        small = Decimal('0.001')
        square = -small*small
        plan = angles._certification_plan(ctx, abs(square))
        assert plan is not None
        sine = sine_term = small
        cosine = cosine_term = Decimal(1)
        engaged = None
        for k in range(1, 81):
            sine_term *= square/Decimal(2*k*(2*k+1))
            cosine_term *= square/Decimal((2*k-1)*2*k)
            sine += sine_term
            cosine += cosine_term
            if k >= 8 and not k & 3:
                ctx.clear_flags()
                ctx.flags[Inexact] = True
                ctx.flags[Rounded] = True
                before = _raised_flags(ctx)
                certified = angles._tail_certified(ctx, k, abs(square), plan,
                                                   sine_term, cosine_term,
                                                   sine, cosine)
                assert _raised_flags(ctx) == before, k
                if certified:
                    engaged = k
                    break
        assert engaged is not None and engaged < 80


def test_certification_plan_integer_bound_refuses_zero_and_nonfinite():
    with localcontext() as ctx:
        ctx.prec = angles.PRECISION
        for magnitude in (Decimal(0), Decimal('0.000'), Decimal('NaN'),
                          Decimal('sNaN'), Decimal('Infinity'),
                          Decimal('-Infinity')):
            assert angles._certification_plan(ctx, magnitude) is None, magnitude
        magnitude = Decimal('4.386')
        plan = angles._certification_plan(ctx, magnitude)
        assert plan is not None
        assert isinstance(plan[2], int)
        assert plan[2] == 5 - magnitude.adjusted()
        # A magnitude whose binary64 conversion underflows to 0.0 must still
        # carry the exact integer plan; only the proof, never a float, decides.
        tiny = Decimal('2.25e-324')
        plan = angles._certification_plan(ctx, tiny)
        assert plan is not None
        assert plan[2] == 5 - tiny.adjusted()


def test_sincos_parity_at_crossed_low_precision_and_subnormal_magnitudes():
    from decimal import (Clamped, Subnormal, Underflow)
    angles._sincos_cache.clear()
    probes = [Decimal('1.5e-162'), Decimal('2.2e-162'), Decimal('2.6e-162'),
              Decimal('2.6e-324'), Decimal('7.4e-324'), Decimal('1e-154'),
              5.0e-324, 1e-162, 0.0, 2.0943951023931953, Decimal('0.982'),
              Decimal('0.001')]
    variants = [{'prec': 1, 'Emin': -40},
                {'prec': 1, 'Emin': -1000},
                {'prec': 1, 'Emin': -999999},
                {'prec': 2, 'Emin': -999999},
                {'prec': 9, 'Emin': -999999},
                {'prec': 28, 'Emin': -1100},
                {'prec': 50, 'Emin': -999999}]
    for variant in variants:
        with localcontext() as ctx:
            ctx.prec = angles.PRECISION
            for name, value in variant.items():
                setattr(ctx, name, value)
            for trap in (Underflow, Subnormal, Clamped):
                ctx.traps[trap] = True
            for probe in probes:
                angles._sincos_cache.clear()
                ctx.clear_flags()
                try:
                    produced = angles._sincos(probe)
                    produced_error = None
                except ArithmeticError as error:
                    produced, produced_error = None, type(error)
                produced_flags = _raised_flags(ctx)
                ctx.clear_flags()
                try:
                    reference = _reference_sincos(probe)
                    reference_error = None
                except ArithmeticError as error:
                    reference, reference_error = None, type(error)
                reference_flags = _raised_flags(ctx)
                assert produced_error is reference_error, (variant, probe)
                assert produced_flags == reference_flags, (variant, probe)
                if produced is None:
                    continue
                assert produced[0].as_tuple() == reference[0].as_tuple(), (variant, probe)
                assert produced[1].as_tuple() == reference[1].as_tuple(), (variant, probe)
