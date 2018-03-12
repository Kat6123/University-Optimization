#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from basic_simplex import basic_phase, SimplexException


class LinearException(SimplexException):
    def __init__(self, k):
        self.k = k
        super(SimplexException, self).__init__("Linear Dependency")


def get_auxiliary(A, b, c):
    _A, _b, _c = A, b, c
    var_num = A.shape[1]
    aux_num = A.shape[0]

    neg_index = np.where(b < 0)
    _A[neg_index] = A[neg_index] * (-1)
    _b[neg_index] = b[neg_index] * (-1)

    _A = np.append(_A, np.eye(aux_num), axis=1)

    _c = np.zeros(var_num + aux_num)
    _c[var_num:] = -1

    return _A, _b, _c, var_num, aux_num


def get_native_not_basis_index(k, i, A, J_b, var_num):
    A_b_inverse = np.linalg.inv(A[:, J_b])

    for j in range(var_num):
        if j not in J_b:
            l_v = A_b_inverse.dot(A[:, j])
            if l_v[k] != 0:
                return j

    raise LinearException(k)


def initial_phase(A, b, c):
    while True:
        _A, _b, _c, var_num, aux_num = get_auxiliary(A, b, c)

        aux_indexes = np.arange(var_num, aux_num + var_num)

        x_b = np.zeros(var_num + aux_num)
        x_b[var_num:] = b
        J_b = np.arange(var_num, aux_num + var_num)

        basic_phase(_A, _b, _c, x_b, J_b)

        if np.any(x_b[var_num:]):
            raise SimplexException("Task is incompatible")

        if not np.any(np.in1d(aux_indexes, J_b)):
            return A, b, c, x_b[:var_num], J_b

        extra_aux_indexes = np.where(np.in1d(J_b, aux_indexes))[0]
        try:
            for k in extra_aux_indexes:
                i = J_b[k] - var_num
                j = get_native_not_basis_index(k, i, _A, J_b, var_num)
                J_b[k] = j
        except LinearException as ex:
            k = ex.k
            A = np.delete(A, k, axis=0)
            b = np.delete(b, k)
            continue

        break
    return A, b, c, x_b[:var_num], J_b


def main():
    A = np.array([
        [0, 1, 4, 1, 0, -3, 5, 0],
        [1, -1, 0, 1, 0, 0, 1, 0],
        [0, 7, -1, 0, -1, 3, 8, 0],
        [1, 1, 1, 1, 0, 3, -3, 1]
    ])
    c = np.array([-5, -2, 3, -4, -6, 0, -1, -5])
    b = np.array([6, 10, -2, 15])

    A, b, c, x_b, J_b = initial_phase(A, b, c)
    print("Initial plan: ", x_b, J_b + 1)
    basic_phase(A, b, c, x_b, J_b)
    print("Optimal plan: ", x_b, J_b + 1)


if __name__ == '__main__':
    main()
