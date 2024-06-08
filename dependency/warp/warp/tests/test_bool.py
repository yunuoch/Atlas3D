# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()

TRUE_CONSTANT = wp.constant(True)


@wp.func
def identity_function(input_bool: wp.bool, plain_bool: bool):
    return input_bool and plain_bool


@wp.kernel
def identity_test(data: wp.array(dtype=wp.bool)):
    i = wp.tid()

    data[i] = data[i] and True
    data[i] = data[i] and wp.bool(True)
    data[i] = data[i] and not False
    data[i] = data[i] and not wp.bool(False)
    data[i] = identity_function(data[i], True)

    if data[i]:
        data[i] = True
    else:
        data[i] = False

    if not data[i]:
        data[i] = False
    else:
        data[i] = True

    if data[i] and True:
        data[i] = True
    else:
        data[i] = False

    if data[i] or False:
        data[i] = True
    else:
        data[i] = False


def test_bool_identity_ops(test, device):
    rng = np.random.default_rng(123)

    dim_x = 10

    rand_np = rng.random(dim_x) > 0.5

    data_array = wp.array(data=rand_np, device=device)

    test.assertEqual(data_array.dtype, wp.bool)

    wp.launch(identity_test, dim=data_array.shape, inputs=[data_array], device=device)

    assert_np_equal(data_array.numpy(), rand_np)


@wp.kernel
def check_compile_constant(result: wp.array(dtype=wp.bool)):
    if TRUE_CONSTANT:
        result[0] = TRUE_CONSTANT
    else:
        result[0] = False


def test_bool_constant(test, device):
    compile_constant_value = wp.zeros(1, dtype=wp.bool, device=device)
    wp.launch(check_compile_constant, 1, inputs=[compile_constant_value], device=device)
    test.assertTrue(compile_constant_value.numpy()[0])


def test_bool_constant_vec(test, device):

    vec3bool = wp.vec(length=3, dtype=wp.bool)
    bool_selector_vec = wp.constant(vec3bool([True, False, True]))

    @wp.kernel
    def sum_from_bool_vec(sum_array: wp.array(dtype=wp.int32)):
        i = wp.tid()

        if bool_selector_vec[0]:
            sum_array[i] = sum_array[i] + 1
        if bool_selector_vec[1]:
            sum_array[i] = sum_array[i] + 2
        if bool_selector_vec[2]:
            sum_array[i] = sum_array[i] + 4

    result_array = wp.zeros(10, dtype=wp.int32, device=device)

    wp.launch(sum_from_bool_vec, result_array.shape, inputs=[result_array], device=device)

    assert_np_equal(result_array.numpy(), np.full(result_array.shape, 5))


def test_bool_constant_mat(test, device):

    mat22bool = wp.mat((2, 2), dtype=wp.bool)
    bool_selector_mat = wp.constant(mat22bool([True, False, False, True]))

    @wp.kernel
    def sum_from_bool_mat(sum_array: wp.array(dtype=wp.int32)):
        i = wp.tid()

        if bool_selector_mat[0, 0]:
            sum_array[i] = sum_array[i] + 1
        if bool_selector_mat[0, 1]:
            sum_array[i] = sum_array[i] + 2
        if bool_selector_mat[1, 0]:
            sum_array[i] = sum_array[i] + 4
        if bool_selector_mat[1, 1]:
            sum_array[i] = sum_array[i] + 8

    result_array = wp.zeros(10, dtype=wp.int32, device=device)

    wp.launch(sum_from_bool_mat, result_array.shape, inputs=[result_array], device=device)

    assert_np_equal(result_array.numpy(), np.full(result_array.shape, 9))


devices = get_test_devices()


class TestBool(unittest.TestCase):
    pass


add_function_test(TestBool, "test_bool_identity_ops", test_bool_identity_ops, devices=devices)
add_function_test(TestBool, "test_bool_constant", test_bool_constant, devices=devices)
add_function_test(TestBool, "test_bool_constant_vec", test_bool_constant_vec, devices=devices)
add_function_test(TestBool, "test_bool_constant_mat", test_bool_constant_mat, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
