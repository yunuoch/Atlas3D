import torch


def normalize_mesh(mesh: torch.Tensor) -> torch.Tensor:
    """
    Normalize the mesh to fit in a unit cube.

    Args:
        mesh: Mesh vertices as tensor of shape (..., 3).

    Returns:
        Normalized mesh vertices as tensor of shape (..., 3).
    """
    min_v = mesh.min(-2).values
    max_v = mesh.max(-2).values
    center = (min_v + max_v) / 2
    scale = (max_v - min_v).max(-1).values
    return (mesh - center) / scale


def set_above_ground(vertices: torch.Tensor, height: float) -> None:
    """
    Set the vertices of a mesh above the ground plane.

    Args:
        vertices: Mesh vertices as tensor of shape (N, 3).
        height: Height above the ground plane.
    """
    new_vertices = vertices.clone()
    new_vertices[:, 2] = vertices[:, 2] - vertices[:, 2].min().detach() + height

    return new_vertices


def transform_vertices(vertices: torch.Tensor,
                       transformation: torch.Tensor) -> torch.Tensor:
    """
    Transform vertices by a transformation matrix.

    Args:
        vertices: Mesh vertices as tensor of shape (..., 3).
        transformation: Transformation matrix as tensor of shape (7) with first three as translation and last four as quaternion.

    Returns:
        Transformed vertices as tensor of shape (..., 3).
    """
    transformation = transformation.view(-1)
    translation = transformation[:3]
    quaternion = transformation[3:]
    rotation = quaternion_to_matrix(quaternion)

    rotated_vertices = torch.mm(vertices, rotation.T)
    translated_vertices = rotated_vertices + translation

    return translated_vertices


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last one,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
