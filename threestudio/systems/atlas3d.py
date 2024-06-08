import os
from dataclasses import dataclass, field

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *

import warp as wp
import torch
import numpy as np
from .loss_func import *
from .mesh_utils import *

wp.init()


@threestudio.register("atlas3d-system")
class Atlas3DSystem(BaseLift3DSystem):

    @dataclass
    class Config(BaseLift3DSystem.Config):
        refinement: bool = False
        coarse_type: str = "magic3d"
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)
        visualize_samples: bool = False
        simulator: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        # Enable to keep only the largest connected component
        self.clean_debris = False

        # Default simulation time
        self.sim_time = self.cfg.simulator.sim_time
        self.sim_time_max = self.cfg.simulator.sim_time_max

        # Base rotation matrix for aligning with ground
        R1 = rotation_matrix_3d_z(self.cfg.simulator.init_rot_z / 180.0 *
                                  np.pi)
        R2 = rotation_matrix_3d_y(self.cfg.simulator.init_rot_y / 180.0 *
                                  np.pi)
        R3 = rotation_matrix_3d_x(self.cfg.simulator.init_rot_x / 180.0 *
                                  np.pi)
        self.base_R = R3 @ R2 @ R1

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.prompt_processor = threestudio.find(
            self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.guidance = threestudio.find(self.cfg.guidance_type)(
            self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        if self.cfg.coarse_type == "magic3d":
            return self.train_refine_magic3d(batch, batch_idx)
        else:
            raise NotImplementedError

    def diff_sim(self, v_pos, faces):
        from .warp_simulator import WarpSimulator, DiffSim

        sim = WarpSimulator()
        sim.use_z_up = True

        DiffSim.sim = sim
        DiffSim.param["total_time"] = self.sim_time
        DiffSim.param["dt"] = 1e-3
        DiffSim.param[
            "return_transform_at"] = -1  # <0 means return the last frame
        DiffSim.param["output_mesh"] = False
        DiffSim.param["output_usd"] = False
        DiffSim.param["folder_name"] = "sim/" + str(self.true_global_step)

        tn = DiffSim.apply(v_pos, faces)[0]
        t0 = wp.to_torch(sim.states[0].body_q)[0]

        # loss from simulation
        loss_sim = rotation_matrix_loss(t0, tn)

        if loss_sim < 1.0e-3 and self.sim_time < self.sim_time_max:
            self.sim_time += 0.1

        return loss_sim

    def train_refine_magic3d(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()
        guidance_out = self.guidance(out["comp_rgb"],
                                     prompt_utils,
                                     **batch,
                                     rgb_as_latents=False)

        loss = 0.0

        # Add loss for guidance
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                # value *= 0.0  # Disable guidance loss
                loss_sds = value
                loss += value * self.C(self.cfg.loss[name.replace(
                    "loss_", "lambda_")])

        if self.cfg.refinement:
            mesh = out["mesh"]
            # Add loss for normal consistency
            loss_normal_consistency = mesh.normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency)

        if self.cfg.refinement:
            mesh = out["mesh"]

            # Clean debris
            if self.clean_debris:
                remove_debris(mesh)

            # Rotate to align with ground
            v_pos = torch.mm(mesh.v_pos, self.base_R.T)

            # Move vertices to above the ground
            v_pos = set_above_ground(v_pos, 0.0)
            faces = mesh.t_pos_idx.detach()

            # simulation loss
            if self.C(self.cfg.loss.lambda_sim) > 0 and batch_idx % 10 == 0:
                loss_sim = self.diff_sim(v_pos, faces)
                self.log("train/loss_sim", loss_sim)
                loss += loss_sim * self.C(self.cfg.loss.lambda_sim)

            # stability loss
            if self.C(self.cfg.loss.lambda_stability) > 0:
                loss_stability = stability_loss(
                    X=v_pos,
                    F=faces,
                    theta_y=0.1,
                    num_samples=20,
                    y_samples=10,
                    is_solid=True,
                )
                self.log("train/loss_stability", loss_stability)
                loss += loss_stability * self.C(self.cfg.loss.lambda_stability)

            # bottom laplacion loss
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = bottom_laplacian_loss(
                    v_pos, faces, percent=0.02, method="uniform")
                self.log("train/loss_laplacian_smoothness",
                         loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness)

        self.log("train/loss", loss)
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log("train_params/sim_time", self.sim_time)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {
                        "data_format": "HWC"
                    },
                },
            ] + ([{
                "type": "rgb",
                "img": out["comp_normal"][0],
                "kwargs": {
                    "data_format": "HWC",
                    "data_range": (0, 1)
                },
            }] if "comp_normal" in out else []) + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {
                        "cmap": None,
                        "data_range": (0, 1)
                    },
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_start(self):
        if self.cfg.refinement:
            mesh = self.geometry.isosurface()
            self.t_mask = None
            if self.clean_debris:
                self.t_mask = remove_debris(mesh)

    def test_step(self, batch, batch_idx):
        if self.cfg.refinement:
            out = self.renderer.render_with_transformation(
                self.base_R,
                torch.zeros(3).to("cuda"), self.t_mask, **batch)
        else:
            out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-vis/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {
                        "data_format": "HWC"
                    },
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

        # Run simulation and render
        if self.cfg.refinement and batch_idx == 119:  # on last test step
            mesh = self.geometry.isosurface()
            if self.clean_debris:
                self.t_mask = remove_debris(mesh)
            vertices = mesh.v_pos.detach().cpu().numpy()
            triangles = mesh.t_pos_idx.detach().cpu().numpy().reshape(-1)
            vertices = vertices @ self.base_R.detach().cpu().numpy().T
            base_t = torch.tensor([0.0, 0.0,
                                   -np.min(vertices[:, 2])]).to("cuda")
            vertices[:, 2] -= np.min(vertices[:, 2])

            from .warp_simulator import WarpSimulator
            from .torch_utils import quaternion_to_matrix

            sim = WarpSimulator()
            sim.use_z_up = True
            sim.load_mesh(vertices, triangles)
            sim.initialize()
            sim.initialize_com()
            sim.advance_to(
                total_time=2.0,
                dt=1e-3,
                return_transform_at=-1,
                output_mesh=False,
                output_usd=False,
                folder_name="it" + str(self.true_global_step) + "-sim",
            )

            step_per_frame = int(sim.frame_dt / 1e-3)
            num_frames = len(sim.states) // step_per_frame
            for i in range(num_frames):
                qn = wp.to_torch(sim.states[i * step_per_frame].body_q)[0]
                tn = qn[:3]
                Rn = quaternion_to_matrix(qn[3:])
                R = Rn @ self.base_R
                t = Rn @ base_t + tn
                t = t - base_t  # move back to original position
                out = self.renderer.render_with_transformation(
                    R, t, self.t_mask, **batch)
                self.save_image_grid(
                    f"it{self.true_global_step}-sim/{i}.png",
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {
                                "data_format": "HWC"
                            },
                        },
                    ],
                    name="test_step",
                    step=self.true_global_step,
                )

            self.save_img_sequence(
                f"it{self.true_global_step}-sim",
                f"it{self.true_global_step}-sim",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="sim",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-vis",
            f"it{self.true_global_step}-vis",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
