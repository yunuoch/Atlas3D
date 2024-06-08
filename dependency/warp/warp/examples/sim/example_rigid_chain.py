# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Chain
#
# Shows how to set up a chain of rigid bodies connected by different joint
# types using wp.sim.ModelBuilder(). There is one chain for each joint
# type, including fixed joints which act as a flexible beam.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage):
        self.chain_length = 8
        self.chain_width = 1.0
        self.chain_types = [
            wp.sim.JOINT_REVOLUTE,
            wp.sim.JOINT_FIXED,
            wp.sim.JOINT_BALL,
            wp.sim.JOINT_UNIVERSAL,
            wp.sim.JOINT_COMPOUND,
        ]

        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        self.frame_dt = 1.0 / 100.0

        episode_duration = 5.0  # seconds
        self.episode_frames = int(episode_duration / self.frame_dt)

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        for c, t in enumerate(self.chain_types):
            # start a new articulation
            builder.add_articulation()

            for i in range(self.chain_length):
                if i == 0:
                    parent = -1
                    parent_joint_xform = wp.transform([0.0, 0.0, c * 1.0], wp.quat_identity())
                else:
                    parent = builder.joint_count - 1
                    parent_joint_xform = wp.transform([self.chain_width, 0.0, 0.0], wp.quat_identity())

                # create body
                b = builder.add_body(origin=wp.transform([i, 0.0, c * 1.0], wp.quat_identity()), armature=0.1)

                # create shape
                builder.add_shape_box(
                    pos=wp.vec3(self.chain_width * 0.5, 0.0, 0.0),
                    hx=self.chain_width * 0.5,
                    hy=0.1,
                    hz=0.1,
                    density=10.0,
                    body=b,
                )

                joint_type = t

                if joint_type == wp.sim.JOINT_REVOLUTE:
                    joint_limit_lower = -np.deg2rad(60.0)
                    joint_limit_upper = np.deg2rad(60.0)
                    builder.add_joint_revolute(
                        parent=parent,
                        child=b,
                        axis=(0.0, 0.0, 1.0),
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                        limit_lower=joint_limit_lower,
                        limit_upper=joint_limit_upper,
                        target_ke=0.0,
                        target_kd=0.0,
                        limit_ke=1e5,
                        limit_kd=1.0,
                    )

                elif joint_type == wp.sim.JOINT_UNIVERSAL:
                    builder.add_joint_universal(
                        parent=parent,
                        child=b,
                        axis_0=wp.sim.JointAxis((1.0, 0.0, 0.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        axis_1=wp.sim.JointAxis((0.0, 0.0, 1.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

                elif joint_type == wp.sim.JOINT_BALL:
                    builder.add_joint_ball(
                        parent=parent,
                        child=b,
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

                elif joint_type == wp.sim.JOINT_FIXED:
                    builder.add_joint_fixed(
                        parent=parent,
                        child=b,
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

                elif joint_type == wp.sim.JOINT_COMPOUND:
                    builder.add_joint_compound(
                        parent=parent,
                        child=b,
                        axis_0=wp.sim.JointAxis((1.0, 0.0, 0.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        axis_1=wp.sim.JointAxis((0.0, 1.0, 0.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        axis_2=wp.sim.JointAxis((0.0, 0.0, 1.0), -np.deg2rad(60.0), np.deg2rad(60.0)),
                        parent_xform=parent_joint_xform,
                        child_xform=wp.transform_identity(),
                    )

        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)

        self.renderer = None
        if stage:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=20.0)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_graph = wp.get_device().is_cuda
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = "example_rigid_chain.usd"

    example = Example(stage_path)

    for _ in range(example.episode_frames):
        example.step()
        example.render()

    if example.renderer:
        example.renderer.save()
