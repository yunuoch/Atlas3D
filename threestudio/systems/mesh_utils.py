def remove_debris(mesh):
    # Keep only the largest connected component of the mesh
    import open3d as o3d
    import torch
    import numpy as np

    np_vertices = mesh.v_pos.detach().cpu().numpy()
    np_triangles = mesh.t_pos_idx.detach().cpu().numpy()
    o3dmesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np_vertices),
        o3d.utility.Vector3iVector(np_triangles))
    t_clus, clus_n_t, clus_area = o3dmesh.cluster_connected_triangles()
    t_mask = np.asarray(t_clus) == np.argmax(clus_n_t)
    v_dele = np.unique(np_triangles[~t_mask])
    mesh.t_pos_idx = mesh.t_pos_idx[t_mask]
    tmp_v_pos = mesh.v_pos.clone()
    tmp_v_pos[v_dele, :] = torch.mean(tmp_v_pos.detach(), dim=0)
    mesh.v_pos.data = tmp_v_pos

    return t_mask
