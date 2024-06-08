# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from typing import Any


@wp.kernel
def sgd_step_kernel(
    g: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    lr: float,
    weight_decay: float,
    momentum: float,
    damping: float,
    nesterov: int,
    t: int,
    params: wp.array(dtype=Any),
):
    i = wp.tid()
    gt = g[i]
    if weight_decay != 0.0:
        gt += weight_decay * params[i]
    if momentum != 0.0:
        bt = b[i]
        if t > 0:
            bt = momentum * bt + (1.0 - damping) * gt
        else:
            bt = gt
        if nesterov == 1:
            gt += momentum * bt
        else:
            gt = bt
        b[i] = bt
    params[i] = params[i] - lr * gt


class SGD:
    """An implementation of the Stochastic Gradient Descent Optimizer
    It is designed to mimic Pytorch's version.
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """

    def __init__(self, params=None, lr=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        self.b = []  # momentum buffer
        self.set_params(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.t = 0

    def set_params(self, params):
        self.params = params
        if params is not None and type(params) == list and len(params) > 0:
            if len(self.b) != len(params):
                self.b = [None] * len(params)
            for i in range(len(params)):
                param = params[i]
                if self.b[i] is None or self.b[i].shape != param.shape or self.b[i].dtype != param.dtype:
                    self.b[i] = wp.zeros_like(param)

    def reset_internal_state(self):
        for b_i in self.b:
            b_i.zero_()
        self.t = 0

    def step(self, grad):
        assert self.params is not None
        for i in range(len(self.params)):
            SGD.step_detail(
                grad[i], self.b[i], self.lr, self.momentum, self.dampening, self.weight_decay, self.nesterov, self.t, self.params[i]
            )
        self.t = self.t + 1

    @staticmethod
    def step_detail(g, b, lr, momentum, dampening, weight_decay, nesterov, t, params):
        assert params.dtype == g.dtype
        assert params.dtype == b.dtype
        assert params.shape == g.shape
        kernel_inputs = [g, b, lr, momentum, dampening, weight_decay, int(nesterov), t, params]
        wp.launch(
            kernel=sgd_step_kernel,
            dim=len(params),
            inputs=kernel_inputs,
            device=params.device,
        )
