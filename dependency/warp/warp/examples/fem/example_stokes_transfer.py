# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Stokes Transfer
#
# This example computes a 2D weakly-compressible Stokes flow around
# a moving object, including:
# - defining active cells from a mask, and restricting the computation domain to those
# - utilizing the PicQuadrature to integrate over unstructured particles 
###########################################################################

import math
import warp as wp
import numpy as np

import warp.fem as fem

from warp.utils import array_cast
from warp.fem.utils import array_axpy
from warp.sparse import bsr_transposed, bsr_mm, bsr_axpy, bsr_mv

# Import example utilities
# Make sure that works both when imported as module and run as standalone file
try:
    from .bsr_utils import bsr_cg
    from .plot_utils import Plot
except ImportError:
    from bsr_utils import bsr_cg
    from plot_utils import Plot

wp.init()


@fem.integrand
def vel_from_particles_form(s: fem.Sample, particle_vel: wp.array(dtype=wp.vec2), v: fem.Field):
    vel = particle_vel[s.qp_index]
    return wp.dot(vel, v(s))


@fem.integrand
def viscosity_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return nu * wp.ddot(fem.D(u, s), fem.D(v, s))


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def scalar_mass_form(
    s: fem.Sample,
    p: fem.Field,
    q: fem.Field,
):
    return p(s) * q(s)


@fem.integrand
def div_form(
    s: fem.Sample,
    u: fem.Field,
    q: fem.Field,
):
    return q(s) * fem.div(u, s)


@fem.integrand
def cell_activity(s: fem.Sample, domain: fem.Domain, c1: wp.vec2, c2: wp.vec2, radius: float):
    pos = domain(s)
    if wp.length(pos - c1) < radius:
        return 0.0
    if wp.length(pos - c2) < radius:
        return 0.0
    return 1.0


@wp.kernel
def inverse_array_kernel(m: wp.array(dtype=wp.float64)):
    m[wp.tid()] = wp.float64(1.0) / m[wp.tid()]


class Example:
    def __init__(self, stage=None, quiet=False):
        self._quiet = quiet

        self.res = 50
        self.cell_size = 1.0 / self.res

        self.vel = 1.0
        self.viscosity = 100.0
        self.compliance = 0.01
        self.bd_strength = 100000.0

        geo = fem.Grid2D(res=wp.vec2i(self.res))

        # Displacement boundary conditions are defined by two circles going in opposite directions
        # Sample particles along those
        circle_radius = 0.15
        c1_center = wp.vec2(0.25, 0.5)
        c2_center = wp.vec2(0.75, 0.5)
        particles, particle_areas, particle_velocities = self._gen_particles(circle_radius, c1_center, c2_center)

        # Disable cells that are interior to the circles
        cell_space = fem.make_polynomial_space(geo, degree=0)
        activity = cell_space.make_field()
        fem.interpolate(
            cell_activity,
            dest=activity,
            values={"c1": c1_center, "c2": c2_center, "radius": circle_radius - self.cell_size},
        )

        # Explicitly define the active geometry partition from those cells
        self._active_partition = fem.ExplicitGeometryPartition(geo, wp.array(activity.dof_values.numpy(), dtype=int))
        if not self._quiet:
            print("Active cells:", self._active_partition.cell_count())

        # Function spaces -- Q1 for vel, Q0 for pressure
        u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
        p_space = fem.make_polynomial_space(geo, degree=0)

        self._active_space_partition = fem.make_space_partition(
            space=u_space, geometry_partition=self._active_partition
        )
        self._active_p_space_partition = fem.make_space_partition(
            space=p_space, geometry_partition=self._active_partition
        )

        self._u_field = u_space.make_field()
        self._p_field = p_space.make_field()

        # Particle-based quadrature rule over active cells
        domain = fem.Cells(geometry=self._active_partition)
        self._pic_quadrature = fem.PicQuadrature(domain, particles, particle_areas)
        self._particle_velocities = particle_velocities

        self.renderer = Plot(stage)

    def step(self):
        u_space = self._u_field.space
        p_space = self._p_field.space

        # Weakly-enforced boundary condition on particles
        u_test = fem.make_test(space=u_space, space_partition=self._active_space_partition)
        u_trial = fem.make_trial(space=u_space, space_partition=self._active_space_partition)

        u_rhs = fem.integrate(
            vel_from_particles_form,
            quadrature=self._pic_quadrature,
            fields={"v": u_test},
            values={"particle_vel": self._particle_velocities},
            output_dtype=wp.vec2d,
        )
        u_bd_matrix = fem.integrate(mass_form, quadrature=self._pic_quadrature, fields={"u": u_trial, "v": u_test})

        # Viscosity
        u_visc_matrix = fem.integrate(
            viscosity_form,
            fields={"u": u_trial, "v": u_test},
            values={"nu": self.viscosity},
        )

        # Pressure-velocity coupling
        p_test = fem.make_test(space=p_space, space_partition=self._active_p_space_partition)
        p_trial = fem.make_trial(space=p_space, space_partition=self._active_p_space_partition)

        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})
        inv_p_mass_matrix = fem.integrate(scalar_mass_form, fields={"p": p_trial, "q": p_test})
        wp.launch(
            kernel=inverse_array_kernel,
            dim=inv_p_mass_matrix.values.shape,
            device=inv_p_mass_matrix.values.device,
            inputs=[inv_p_mass_matrix.values],
        )

        # Assemble linear system
        u_matrix = u_visc_matrix
        bsr_axpy(u_bd_matrix, u_matrix, alpha=self.bd_strength)

        div_matrix_t = bsr_transposed(div_matrix)
        gradient_matrix = bsr_mm(div_matrix_t, inv_p_mass_matrix)
        bsr_mm(gradient_matrix, div_matrix, u_matrix, alpha=1.0 / self.compliance, beta=1.0)

        array_axpy(u_rhs, u_rhs, alpha=0.0, beta=self.bd_strength)

        # Solve for displacement
        u_res = wp.zeros_like(u_rhs)
        bsr_cg(u_matrix, x=u_res, b=u_rhs, quiet=self._quiet)

        # Compute pressure from displacement
        div_u = bsr_mv(A=div_matrix, x=u_res)
        p_res = bsr_mv(A=inv_p_mass_matrix, x=div_u, alpha=-1)

        # Copy to fields
        u_nodes = wp.indexedarray(self._u_field.dof_values, indices=self._active_space_partition.space_node_indices())
        p_nodes = wp.indexedarray(self._p_field.dof_values, indices=self._active_p_space_partition.space_node_indices())

        array_cast(in_array=u_res, out_array=u_nodes)
        array_cast(in_array=p_res, out_array=p_nodes)

    def render(self):
        self.renderer.add_surface("pressure", self._p_field)
        self.renderer.add_surface_vector("velocity", self._u_field)

    def _gen_particles(self, circle_radius, c1_center, c2_center):
        """Generate some particles along two circles defining velocity boundary conditions"""

        # Generate particles defining the transfer displacement
        particles_per_circle = int(2.0 * math.pi * circle_radius * self.res)

        angles = np.linspace(0, 2.0 * math.pi, particles_per_circle, endpoint=False)

        n_particles = 2 * particles_per_circle
        particles = np.empty((n_particles, 2), dtype=float)

        particles[:particles_per_circle, 0] = c1_center[0] + circle_radius * np.cos(angles)
        particles[:particles_per_circle, 1] = c1_center[1] + circle_radius * np.sin(angles)
        particles[particles_per_circle:, 0] = c2_center[0] + circle_radius * np.cos(angles)
        particles[particles_per_circle:, 1] = c2_center[1] + circle_radius * np.sin(angles)

        particle_areas = np.ones(n_particles) * self.cell_size**2
        particle_velocities = np.zeros_like(particles)
        particle_velocities[:particles_per_circle, 0] = self.vel
        particle_velocities[particles_per_circle:, 0] = -self.vel

        particles = wp.array(particles, dtype=wp.vec2)
        particle_areas = wp.array(particle_areas, dtype=float)
        particle_velocities = wp.array(particle_velocities, dtype=wp.vec2)

        return particles, particle_areas, particle_velocities


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": False})

    example = Example()
    example.step()
    example.render()

    example.renderer.plot(streamlines=["velocity"])
