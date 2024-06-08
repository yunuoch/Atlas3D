# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import warp as wp

wp.init()


@wp.kernel
def arange(out: wp.array(dtype=int)):
    tid = wp.tid()
    out[tid] = tid


device = "cuda:0"
cmds = []

n = 10
arrays = []

for i in range(5):
    arrays.append(wp.zeros(n, dtype=int, device=device))

# setup CUDA graph
wp.capture_begin()

# launch kernels and keep command object around
for i in range(5):
    cmd = wp.launch(arange, dim=n, inputs=[arrays[i]], device=device, record_cmd=True)
    cmds.append(cmd)

graph = wp.capture_end()

# ---------------------------------------

ref = np.arange(0, n, dtype=int)
wp.capture_launch(graph)

for i in range(5):
    print(arrays[i].numpy())


# ---------------------------------------

n = 16
arrays = []

for i in range(5):
    arrays.append(wp.zeros(n, dtype=int, device=device))

# update graph params
for i in range(5):
    cmd.set_dim(n)
    cmd.set_param(arrays[i])

    cmd.update_graph()


wp.capture_launch(graph)
wp.synchronize()

ref = np.arange(0, n, dtype=int)

for i in range(5):
    print(arrays[i].numpy())
