import numpy as np
from pytket.circuit import OpType  # type: ignore

distributable_1q_op_types = [
    OpType.Rz,
    OpType.X,
    OpType.U1,
]

distributable_op_types = distributable_1q_op_types + [
    OpType.CU1,
    OpType.CRz,
    OpType.CZ,
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
