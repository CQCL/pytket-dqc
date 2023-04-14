# Copyright 2023 Quantinuum and The University of Tokyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pytket.circuit import OpType  # type: ignore

distributable_1q_op_types = [
    OpType.Rz,
    OpType.X,
    OpType.Z
]

distributable_op_types = distributable_1q_op_types + [
    OpType.CU1,
]


def is_diagonal(op):
    # Boolean function that determines
    # if a given command has an associated matrix representation
    # (in the computational basis) that is diagonal.
    # This function uses the fastest answer presented here
    # https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python

    array = op.get_unitary().round(12)  # To stay consistent with TKET team.
    i, j = array.shape
    test = array.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


def is_antidiagonal(op):
    # See is_diagonal() for discussion
    array = np.flip(op.get_unitary(), 0).round(12)
    i, j = array.shape
    test = array.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


def is_distributable(op):
    return is_antidiagonal(op) or is_diagonal(op)
