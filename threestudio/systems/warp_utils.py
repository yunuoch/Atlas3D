import warp as wp

from .warp_functions import *


@wp.kernel
def compute_mesh_com_solid_undivided(vertices: wp.array(dtype=wp.vec3f), faces: wp.array(dtype=wp.vec3i), com: wp.array(dtype=wp.vec3f), volume: wp.array(dtype=float)):
    tid = wp.tid()
    v1 = vertices[faces[tid][0]]
    v2 = vertices[faces[tid][1]]
    v3 = vertices[faces[tid][2]]
    first_pt = wp.vec3(0.0, 0.0, 0.0)
    vol = compute_tet_volume(first_pt, v1, v2, v3)
    wp.atomic_add(com, 0,vol * compute_tet_com(v1, v2, v3))
    wp.atomic_add(volume, 0, vol)

@wp.kernel
def compute_mesh_com_solid_divide_by_total_vol(com_in: wp.array(dtype=wp.vec3f), volume: wp.array(dtype=float), com_out: wp.array(dtype=wp.vec3f)):
    com_out[0] = com_in[0] / volume[0]

def compute_solid_com_warp(vertices: wp.array(dtype=wp.vec3f), faces: wp.array(dtype=wp.vec3i), com: wp.array(dtype=wp.vec3)):
    com.zero_()
    com_temp = wp.zeros_like(com)
    volume = wp.zeros(shape=1, dtype=float, device="cuda")
    wp.launch(kernel=compute_mesh_com_solid_undivided,
              dim=faces.shape[0],
              inputs=[vertices, faces, com_temp, volume])
    
    wp.launch(kernel=compute_mesh_com_solid_divide_by_total_vol,
              dim=1,
              inputs=[com_temp, volume, com])
    

@wp.kernel
def compute_avg(
    x: wp.array(dtype=wp.vec3), n: float, total: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    wp.atomic_add(total, 0, x[tid] / n)

@wp.kernel
def compute_sum(x: wp.array(dtype=wp.vec3), total: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    wp.atomic_add(total, 0, x[tid])


@wp.kernel
def devided_by_n(n: float, dummy: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    com[0] = dummy[0] / n


def compute_hollow_com(x: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    total = wp.zeros_like(com)
    wp.launch(compute_sum, 
              dim = x.shape[0],
              inputs=[x, total])
    wp.launch(devided_by_n, 
              dim = 1,
              inputs=[float(x.shape[0]), total, com])
    
@wp.kernel
def loss_stability(transform: wp.array(dtype=wp.transformf),
                   transformX: wp.array(dtype=wp.transformf), w_t: float,
                   w_r: float, loss: wp.array(dtype=float)):
    delta_t = wp.transform_get_translation(
        transform[0]) - wp.transform_get_translation(transformX[0])
    delta_r = wp.transform_get_rotation(
        transform[0]) - wp.transform_get_rotation(transformX[0])
    loss[0] = w_t * wp.dot(delta_t, delta_t) + w_r * wp.dot(delta_r, delta_r)


@wp.kernel
def loss_rot_diff(transform: wp.array(dtype=wp.transformf),
                  transformX: wp.array(dtype=wp.transformf),
                  loss: wp.array(dtype=float)):
    delta = wp.transform_get_rotation(
        transform[0]) - wp.transform_get_rotation(transformX[0])
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def extract_translation(transform: wp.array(dtype=wp.transform),
                        translation: wp.array(dtype=wp.vec3)):
    translation[0] = wp.transform_get_translation(transform[0])


@wp.kernel
def loss_dist(transform: wp.array(dtype=wp.transformf), target: wp.vec3,
              loss: wp.array(dtype=float)):
    '''
    Compute the loss as the squared distance between the position and the target
    '''
    delta = wp.transform_get_translation(transform[0]) - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel()
def assign_vx(v: wp.array(dtype=wp.spatial_vectorf), vx: wp.array(dtype=float,
                                                                  ndim=1)):
    '''
    Assign the velocity to the spatial vector
    '''
    v[0] = wp.spatial_vector(0.0, 0.0, 0.0, vx[0], vx[1], vx[2])


@wp.kernel()
def assign_body_qd(qd_src: wp.array(dtype=float, ndim=1),
                   qd_dst: wp.array(dtype=wp.spatial_vectorf)):
    '''
    Assign the velocity to the spatial vector
    '''
    qd_dst[0] = wp.spatial_vector(qd_src[0], qd_src[1], qd_src[2], qd_src[3],
                                  qd_src[4], qd_src[5])



@wp.kernel
def clamp_x(x: wp.array(dtype=wp.vec3), min_x: float, max_x: float):
    tid = wp.tid()
    x[tid][0] = wp.clamp(x[tid][0], min_x, max_x)
    x[tid][1] = wp.clamp(x[tid][1], min_x, max_x)
    x[tid][2] = wp.clamp(x[tid][2], min_x, max_x)


@wp.kernel
def transform_vertices(x_in: wp.array(dtype=wp.vec3),
                       t: wp.types.transformation(dtype=wp.float32),
                       x_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    x_out[tid] = wp.transform_point(t, x_in[tid])


import numpy as np
import os
import openmesh as om


def save_obj(file_name, vertices, faces, verbose=False):
    """
    Save a mesh as an OBJ file.

    Parameters:
        file_name (str): Name of the OBJ file to save.
        vertices (list): List of vertex coordinates. Each vertex should be a tuple of (x, y, z) coordinates.
        faces (list): List of faces. Each face is a list of vertex indices that make up the face.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        for vertex in vertices:
            f.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))

        for face in faces:
            f.write("f")
            for vertex_index in face:
                f.write(" {}".format(vertex_index +
                                     1))  # OBJ format uses 1-based indexing
            f.write("\n")
    if verbose:
        print("File " + file_name + " is saved.")
