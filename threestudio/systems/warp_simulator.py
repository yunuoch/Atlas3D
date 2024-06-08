import os

import warp as wp
import warp.sim
import warp.sim.render

import torch
import math
import numpy as np

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .warp_utils import *


class WarpSimulator:

    def __init__(self, frame_dt=1.0 / 50.0):

        self.frame_dt = frame_dt
        self.sim_time = 0.0

        self.builder = wp.sim.ModelBuilder()
        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.renderer = None

        self.use_z_up = False  #

    def load_mesh_and_finalize(self, mesh):
        X = mesh.v_pos.detach().cpu().numpy()
        F = mesh.t_pos_idx.detach().cpu().numpy()
        b = self.builder.add_body(origin=wp.transform_identity(
            dtype=wp.float32))
        self.wp_mesh = wp.sim.Mesh(X, F.flatten())
        self.X_wp = wp.array(X, dtype=wp.vec3, device="cuda:0")
        self.x_wp = wp.array(X, dtype=wp.vec3, device="cuda:0")
        self.F = F
        self.builder.add_shape_mesh(body=b, mesh=self.wp_mesh)

        self.initialize()

    def load_mesh(self, X, F):
        b = self.builder.add_body(origin=wp.transform_identity(
            dtype=wp.float32))
        self.wp_mesh = wp.sim.Mesh(X, F.flatten())
        self.X_wp = wp.array(X, dtype=wp.vec3, device="cuda:0")
        self.x_wp = wp.array(X, dtype=wp.vec3, device="cuda:0")
        self.F = F
        self.builder.add_shape_mesh(body=b, mesh=self.wp_mesh)

    def initialize(self):
        if self.use_z_up:
            self.builder.set_ground_plane((0.0, 0.0, 1.0), 0.0)
        self.model = self.builder.finalize(device=None, requires_grad=True)
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 2.0
        self.model.soft_contact_kf = 1.0e3
        self.model.gravity = wp.vec3(0.0, -9.81, 0.0)
        if self.use_z_up:
            self.model.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.model.ground = True

        self.states = [self.model.state(requires_grad=True)]

    def initialize_com(self):

        self.model.body_com.zero_()
        compute_solid_com_warp(
            self.wp_mesh.mesh.points,
            wp.array(self.F, dtype=wp.vec3i, device="cuda:0"),
            self.model.body_com)

    def advance_to(self,
                   total_time,
                   dt,
                   return_transform_at=-1.0,
                   output_mesh=False,
                   output_usd=False,
                   folder_name="sim",
                   verbose=False):
        self.sim_time = 0.0
        total_frame = int(total_time / dt)
        render_every_k_frame = int(self.frame_dt / dt)
        render_frame = 0

        if output_usd and self.renderer is None:
            usd_path = os.path.join(os.path.dirname(__file__),
                                    "outputs/" + folder_name + ".usd")
            self.renderer = wp.sim.render.SimRenderer(self.model, usd_path)

        for f in range(total_frame):
            if output_mesh:
                if (f % render_every_k_frame == 0):
                    current_state = self.states[f]
                    transform7 = wp.to_torch(current_state.body_q)[0]
                    L = transform7.tolist()
                    t = wp.transform(L[0], L[1], L[2], L[3], L[4], L[5], L[6])
                    self.output_mesh(
                        'outputs/' + folder_name + '/' + str(render_frame) +
                        '.obj', t, verbose)
                    render_frame = render_frame + 1

            if output_usd:
                self.render_usd(f)

            wp.sim.collide(self.model, self.states[f])
            self.states[f].clear_forces()
            self.states.append(self.model.state(requires_grad=True))
            self.integrator.simulate(self.model, self.states[f],
                                     self.states[f + 1], dt)
            self.sim_time += dt

        if output_usd:
            self.renderer.save()

        if return_transform_at < 0.0:
            return self.states[-1].body_q
        return self.states[int(return_transform_at / dt)].body_q

    def update_x(self):
        transform7 = wp.to_torch(self.states[-1].body_q)[0]
        L = transform7.tolist()
        t = wp.transform(L[0], L[1], L[2], L[3], L[4], L[5], L[6])
        wp.launch(
            transform_vertices,
            dim=self.X_wp.shape[0],
            inputs=[self.X_wp, t, self.x_wp],
        )

    def output_mesh(self, filename, transform, verbose=False):
        wp.launch(
            transform_vertices,
            dim=self.X_wp.shape[0],
            inputs=[self.X_wp, transform, self.x_wp],
        )
        save_obj(filename, self.x_wp.numpy(), self.F, verbose)

    def render_usd(self, f):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.states[f])
        self.renderer.end_frame()


@wp.kernel
def copy_vec3_array(origin: wp.array(dtype=wp.vec3),
                    target: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target[tid] = origin[tid] * 1.0


class DiffSim(torch.autograd.Function):
    # Create a class instance of torch.autograd.Function is not allowed
    # thus simulator is defined as a static variable
    sim = None

    param = {
        "total_time": 2.0,
        "dt": 1e-3,
        "return_transform_at": -1.0,
        "output_mesh": False,
        "output_usd": False,
        "folder_name": sim,
        "verbose": False,
    }
    tape = wp.Tape()

    @staticmethod
    def forward(ctx, mesh_x, mesh_f):
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        DiffSim.tape = wp.Tape()
        with DiffSim.tape:
            # load mesh to simulator and initialize
            np_mesh_x = mesh_x.detach().cpu().numpy()
            np_mesh_f = mesh_f.detach().cpu().numpy()
            DiffSim.sim.load_mesh(np_mesh_x, np_mesh_f)
            DiffSim.sim.initialize()

            # convert torch tensor to warp array
            wp_mesh_x = wp.from_torch(mesh_x,
                                      requires_grad=True,
                                      dtype=wp.vec3)

            # assign mesh_x to simulator
            wp.launch(copy_vec3_array,
                      dim=(len(mesh_x)),
                      inputs=[
                          wp_mesh_x,
                          DiffSim.sim.model.shape_geo_src[0].mesh.points
                      ])

            # recalculate the center of mass and inertia
            DiffSim.sim.initialize_com()

            ctx.transform = DiffSim.sim.advance_to(
                total_time=DiffSim.param["total_time"],
                dt=DiffSim.param["dt"],
                return_transform_at=DiffSim.param["return_transform_at"],
                output_mesh=DiffSim.param["output_mesh"],
                output_usd=DiffSim.param["output_usd"],
                folder_name=DiffSim.param["folder_name"],
                verbose=DiffSim.param["verbose"])

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return wp.to_torch(ctx.transform)

    @staticmethod
    def backward(ctx, adj_transform):
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        # map incoming Torch grads to our output variables
        ctx.transform.grad = wp.from_torch(adj_transform, dtype=wp.transformf)

        DiffSim.tape.backward()

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        # mesh_x_grad = wp.to_torch(ctx.mesh_x.grad)
        mesh_x_grad = wp.to_torch(
            DiffSim.sim.model.shape_geo_src[0].mesh.points.grad)

        return mesh_x_grad, None
