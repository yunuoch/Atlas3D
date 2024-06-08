import torch
import math
import numpy as np

from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian

from .torch_utils import *


def rotation_matrix_3d_z(angle):
    """
    Create a 3D rotation matrix for rotation around the z-axis by a given angle in radians.
    """
    # angle = torch.tensor([angle], device = "cuda")
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    matrix = torch.tensor(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0],
         [0.0, 0.0, 1.0]],
        device="cuda",
    )
    return matrix


def rotation_matrix_3d_y(angle):
    """
    Create a 3D rotation matrix for rotation around the y-axis by a given angle in radians.
    """
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    matrix = torch.tensor(
        [[cos_theta, 0.0, sin_theta], [0.0, 1.0, 0.0],
         [-sin_theta, 0.0, cos_theta]],
        device="cuda",
    )
    return matrix


def rotation_matrix_3d_x(angle):
    """
    Create a 3D rotation matrix for rotation around the x-axis by a given angle in radians.
    """
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cos_theta, -sin_theta],
         [0.0, sin_theta, cos_theta]],
        device="cuda",
    )
    return matrix


def compute_solid_mesh_com_torch(vertices, faces):
    # vertices has shape (n,3),
    # faces has shape (m,3),
    vertices_1 = vertices[faces[:, 0], :]  # num_tets by 3
    vertices_2 = vertices[faces[:, 1], :]  # num_tets by 3
    vertices_3 = vertices[faces[:, 2], :]  # num_tets by 3

    tets_coms = (vertices_1 + vertices_2 + vertices_3) / 4.0
    cross1and2 = torch.linalg.cross(vertices_1, vertices_2)
    dot3 = torch.linalg.vecdot(cross1and2, vertices_3)
    com = torch.mul(torch.unsqueeze(dot3, 1),
                    tets_coms).sum(0) / torch.sum(dot3)
    # vol = torch.sum(dot3) / 6.0
    # print("vol: ", vol)
    return com


def _stability_loss(X,
                    F=None,
                    theta_y=0.1,
                    num_samples=20,
                    y_samples=10,
                    is_solid=False):
    if is_solid:
        com = compute_solid_mesh_com_torch(X, F)
    else:
        com = torch.mean(X, 0)
    shift_to_origin = X - com
    original_com_to_min = -shift_to_origin[:, 2].min()

    loss = torch.tensor(0.0, device="cuda")
    z_angles = torch.zeros(num_samples + 1, 3, device="cuda")
    z_angles[:, 2] = torch.linspace(0,
                                    2.0 * torch.pi,
                                    num_samples + 1,
                                    device="cuda")
    y_angles = torch.zeros(y_samples, 3, device="cuda")
    y_angles[:, 1] = torch.linspace(theta_y / y_samples,
                                    theta_y,
                                    y_samples,
                                    device="cuda")

    for i in range(y_samples):
        # apply rotation around z-axis
        rotation_matrix_z = axis_angle_to_matrix(z_angles)[:-1]
        new_vertices = torch.matmul(shift_to_origin, rotation_matrix_z)

        # apply rotation around y-axis
        rotation_matrix_y = axis_angle_to_matrix(y_angles[i])
        new_vertices = torch.matmul(new_vertices, rotation_matrix_y)

        rotated_com_to_min, _ = torch.min(new_vertices[:, :, 2], 1)
        rotated_com_to_min = -rotated_com_to_min

        height_diff = original_com_to_min - rotated_com_to_min
        height_diff = torch.clamp(height_diff, min=0.0)
        loss += torch.sum(height_diff)

    loss = loss / (num_samples * y_samples)
    return loss


def stability_loss(X,
                   F=None,
                   theta_y=0.1,
                   num_samples=20,
                   y_samples=10,
                   iter=1,
                   is_solid=True):
    loss = 0.0
    for i in range(0, iter):
        smoothed_X = X.clone()
        smoothed_X = laplacian_smooth(smoothed_X,
                                      F,
                                      smoothed_X,
                                      iter=i,
                                      method="cot")

        loss += _stability_loss(smoothed_X, F, theta_y, num_samples, y_samples,
                                is_solid)

    return loss


def rotation_matrix_loss(t0, tn):
    q0 = t0[3:]
    qn = tn[3:]
    r0 = quaternion_to_matrix(q0)
    rn = quaternion_to_matrix(qn)

    delta = (rn - r0).view(-1)
    loss = torch.dot(delta, delta)
    return loss


def quat_difference_loss(t0, tn):
    qn = tn[3:]
    q0 = t0[3:]

    delta = qn - q0
    loss = torch.dot(delta, delta)
    return loss


def position_difference_loss(t0, tn, X):
    tn = tn.view(-1)
    t0 = t0.view(-1)
    Xn = transform_vertices(X, tn)
    X0 = transform_vertices(X, t0)
    delta = (Xn - X0).view(-1)
    loss = torch.dot(delta, delta) / X.shape[0]
    return loss


def bottom_laplacian_loss(vertices,
                          faces,
                          percent=2.e-2,
                          method: str = "uniform"):
    meshes = Meshes(verts=[vertices], faces=[faces])
    if meshes.isempty():
        return torch.tensor([0.0],
                            dtype=torch.float32,
                            device=meshes.device,
                            requires_grad=True)

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # for vertices with z axis less than height, scale the weights with a factor of 10
    max_z = torch.max(verts_packed[:, 2])
    min_z = torch.min(verts_packed[:, 2])
    height = min_z + percent * (max_z - min_z)
    weights = torch.where(verts_packed[:, 2] < height, weights, 0.)
    weights = weights / torch.sum(weights)

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w

    # loss = loss.norm(dim=1)
    # WARNING: calculate the z axis loss only
    loss = torch.abs(loss[:, 2])

    loss = loss * weights
    return loss.sum() / N


def laplacian_smooth(vertices,
                     faces,
                     attribute,
                     iter=3,
                     method: str = "uniform"):
    assert vertices.shape[0] == attribute.shape[0]
    meshes = Meshes(verts=[vertices], faces=[faces])
    if meshes.isempty():
        return torch.tensor([0.0],
                            dtype=torch.float32,
                            device=meshes.device,
                            requires_grad=True)

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

        if method == "uniform":
            for i in range(iter):
                attribute = L.mm(attribute) + attribute
        elif method == "cot":
            # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
            for i in range(iter):
                attribute = L.mm(attribute) * norm_w
        elif method == "cotcurv":
            raise NotImplementedError
        #     # pyre-fixme[61]: `norm_w` may not be initialized here.
        #     loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w

    return attribute
