# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

version = "1.0.2"

verify_fp = False  # verify inputs and outputs are finite after each launch
verify_cuda = False  # if true will check CUDA errors after each kernel launch / memory operation
print_launches = False  # if true will print out launch information

mode = "release"
verbose = False  # print extra informative messages
verbose_warnings = False  # whether file and line info gets included in Warp warnings
quiet = False  # suppress all output except errors and warnings

cache_kernels = True
kernel_cache_dir = None  # path to kernel cache directory, if None a default path will be used

cuda_output = (
    None  # preferred CUDA output format for kernels ("ptx" or "cubin"), determined automatically if unspecified
)

ptx_target_arch = 70  # target architecture for PTX generation, defaults to the lowest architecture that supports all of Warp's features

enable_backward = True  # whether to compiler the backward passes of the kernels

llvm_cuda = False  # use Clang/LLVM instead of NVRTC to compile CUDA

enable_graph_capture_module_load_by_default = True  # Default value of force_module_load for capture_begin()

enable_mempools_at_init = True  # Whether CUDA devices will be initialized with mempools enabled (if supported)

max_unroll = 16
