"""Independent straight centered principal-axis loading reference (SI).

No anyfesolver import or production inertia/work/contact kernel is used here.
The ellipse primitives are independently declared geometry, not inferred from
EllipsoidalGeneralizedSection (a generalized yield-law class).
Prepared source only; this reference grants no execution or qualification.
"""
from dataclasses import dataclass
from math import cos, sin, isfinite, pi, sqrt


@dataclass(frozen=True)
class LoadingInputs:
    length: float = 2.
    density: float = 2700.
    semi_axis_y: float = .02
    semi_axis_z: float = .03
    young: float = 1.e7
    poisson: float = .3
    force_per_length: tuple = (0., .4, 0.)
    couple_per_length: tuple = (.04, 0., 0.)
    initial_translation_velocity: tuple = (0., .1, .02)
    initial_angle: float = .125
    initial_spin: float = 25.
    dt: float = .0025
    steps: int = 4

    def __post_init__(self):
        positive = (self.length, self.density, self.semi_axis_y, self.semi_axis_z, self.young, self.dt)
        if any(type(x) is not float or not isfinite(x) or x <= 0 for x in positive):
            raise ValueError('positive finite declared SI parameters required')
        for row in (self.force_per_length, self.couple_per_length,
                    self.initial_translation_velocity):
            if type(row) is not tuple or len(row) != 3 or any(
                    type(x) is not float or not isfinite(x) for x in row):
                raise ValueError('finite binary64 vector declaration required')
        if type(self.poisson) is not float or not isfinite(self.poisson) or not -1. < self.poisson < .5:
            raise ValueError('declared isotropic Poisson ratio required')
        if any(self.couple_per_length[1:]):
            raise ValueError('reference restricted to longitudinal principal couple')
        if type(self.steps) is not int or not 1 <= self.steps <= 16:
            raise ValueError('bounded reference step count required')
        if any(type(x) is not float or not isfinite(x) for x in
               (self.initial_angle, self.initial_spin)):
            raise ValueError('finite reference angle/rate required')


    @property
    def area(self):
        return pi*self.semi_axis_y*self.semi_axis_z

    @property
    def iy(self):
        return self.area*self.semi_axis_z**2/4

    @property
    def iz(self):
        return self.area*self.semi_axis_y**2/4

    @property
    def mass_per_length(self):
        return self.density*self.area

    @property
    def polar_inertia_per_length(self):
        return self.density*(self.iy+self.iz)

    @property
    def transverse_inertia_y(self):
        return self.density*self.iy

    @property
    def transverse_inertia_z(self):
        return self.density*self.iz

    @property
    def stiffness_diagonal(self):
        g = self.young/(2*(1+self.poisson))
        jt = pi*self.semi_axis_y**3*self.semi_axis_z**3/(self.semi_axis_y**2+self.semi_axis_z**2)
        return (self.young*self.area, (5/6)*g*self.area, (5/6)*g*self.area,
                g*jt, self.young*self.iy, self.young*self.iz)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def rx(angle):
    c, s = cos(angle), sin(angle)
    return ((1., 0., 0.), (0., c, -s), (0., s, c))


def endpoint(inputs, time):
    """Exact rigid invariant solution; t is actual accepted endpoint time."""
    if type(time) is not float or not isfinite(time) or time < 0.:
        raise ValueError('finite nonnegative time required')
    a = tuple(f/inputs.mass_per_length for f in inputs.force_per_length)
    alpha = inputs.couple_per_length[0]/inputs.polar_inertia_per_length
    v = tuple(v0+ai*time for v0, ai in zip(inputs.initial_translation_velocity, a))
    shift = tuple(v0*time+.5*ai*time*time for v0, ai in
                  zip(inputs.initial_translation_velocity, a))
    omega = inputs.initial_spin+alpha*time
    angle = inputs.initial_angle+inputs.initial_spin*time+.5*alpha*time*time
    m, j = inputs.mass_per_length*inputs.length, inputs.polar_inertia_per_length*inputs.length
    p = tuple(m*x for x in v)
    orbital = cross(shift, p)
    h = (orbital[0]+j*omega, orbital[1], orbital[2])
    k = .5*m*dot(v, v)+.5*j*omega*omega
    positions = tuple((x+shift[0], shift[1], shift[2]) for x in
                      (-.5*inputs.length, 0., .5*inputs.length))
    return dict(time=time, positions=positions, rotation=rx(angle), angle=angle,
                translation_velocity=v, angular_velocity=(omega, 0., 0.),
                translation_acceleration=a, angular_acceleration=(alpha, 0., 0.),
                kinetic_energy=k, momentum=p, angular_momentum=h,
                potential_energy=0., shift=shift)


def interval(inputs, start, end):
    """Exact total force/moment impulse and load work about fixed global zero."""
    first, last = endpoint(inputs, start), endpoint(inputs, end)
    if end < start:
        raise ValueError('ordered reference interval required')
    dt = end-start
    force = tuple(inputs.length*x for x in inputs.force_per_length)
    couple = tuple(inputs.length*x for x in inputs.couple_per_length)
    # Integral shift(t) cross F dt; do not equate all moment to torque alone.
    integral_shift = tuple(.5*v0*(end*end-start*start)+
        (f/inputs.mass_per_length)*(end**3-start**3)/6 for v0, f in
        zip(inputs.initial_translation_velocity, inputs.force_per_length))
    arm_impulse = cross(integral_shift, force)
    force_impulse = tuple(x*dt for x in force)
    moment_impulse = tuple(c*dt+x for c, x in zip(couple, arm_impulse))
    shift = tuple(b-a for a, b in zip(first['shift'], last['shift']))
    work = dot(force, shift)+couple[0]*(last['angle']-first['angle'])
    return dict(force=force, pure_couple=couple, force_impulse=force_impulse,
                moment_impulse=moment_impulse, work=work)


def stage(inputs, start, end):
    """Independent discrete midpoint pose, distinct from continuum midtime pose.

    For commuting rotations, cells are Rx(mean endpoint angles). Position is
    endpoint average: difference from continuous midpoint is a*h^2/8.
    Velocity is exact continuous midtime velocity under constant acceleration.
    """
    first, last = endpoint(inputs, start), endpoint(inputs, end)
    middle = endpoint(inputs, float((start+end)/2))
    shift = tuple((a+b)/2 for a, b in zip(first['shift'], last['shift']))
    angle = (first['angle']+last['angle'])/2
    force = tuple(inputs.length*x for x in inputs.force_per_length)
    couple = tuple(inputs.length*x for x in inputs.couple_per_length)
    moment = tuple(a+b for a,b in zip(cross(shift, force), couple))
    return dict(shift=shift, angle=angle, rotation=rx(angle),
                velocity=middle['translation_velocity'], omega=middle['angular_velocity'],
                force=force, moment=moment)


def scales(inputs):
    """Positive input-frozen SI scales; never selected from observed errors."""
    t = inputs.steps*inputs.dt
    end = endpoint(inputs, float(t))
    load = interval(inputs, 0., float(t))
    norm = lambda row: sqrt(dot(row, row))
    return dict(length_m=norm(end['shift']), angle_rad=abs(end['angle']-inputs.initial_angle),
        velocity_m_s=norm(end['translation_acceleration'])*t,
        omega_rad_s=abs(end['angular_acceleration'][0])*t,
        acceleration_m_s2=norm(end['translation_acceleration']),
        alpha_rad_s2=abs(end['angular_acceleration'][0]),
        force_n=norm(load['force']), moment_nm=norm(load['pure_couple']),
        momentum_ns=norm(load['force_impulse']), angular_momentum_nms=norm(load['moment_impulse']),
        energy_j=abs(load['work']))
