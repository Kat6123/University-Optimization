#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from inverse_matrix import find_inverse, SimplexException


def basic_phase(A, b, c, x_b, J_b):
    iteration = check_basic_iter(A, b, c, x_b, J_b)
    while not iteration["result"]:
        iteration = check_basic_iter(A, b, c, x_b, J_b, iteration["info"])


def check_basic_iter(A, b, c, x_b, J_b, args=None):
    A_b = A[:, J_b]
    if args is None:
        A_b_inverse = np.linalg.inv(A_b)
    else:
        A_b_inverse = find_inverse(*args)
    c_b = c[J_b]
    u = c_b.dot(A_b_inverse)
    delta = u.dot(A) - c
    delta_n = np.delete(delta, J_b)

    if np.all([delta_n >= 0]):          # Check if plan is optimal
        return {"result": True}

    delta_negative = delta_n[delta_n < 0]   # Find j as first negative index
    j_0 = np.where(delta == delta_negative[0])[0][0]

    z = A_b_inverse.dot(A[:, j_0])
    tetta = np.full(z.shape, np.inf)
    z_pos_index = np.where(z > 0)[0]
    new_tetta = [x_b[J_b[i]] / z[i] for i in z_pos_index]
    tetta[z_pos_index] = new_tetta

    tetta_0 = tetta.min()           # Find tetta_0

    if np.isinf(tetta_0):           # Check if all tetta infinum
        raise SimplexException("Unlimited function")
    s = np.where(tetta == tetta_0)[0][0]
    j_0_s = J_b[s]

    new_x_basis = [
        (x_b[j] - tetta_0 * z[i]) if i != s else tetta_0
        for (i, j) in enumerate(J_b)]

    J_b[s] = j_0                    # Create new basis

    x_b[J_b] = new_x_basis
    x_b[j_0_s] = 0

    return {"result": False,
            "info": (A_b, A_b_inverse, A[:, j_0], s, A_b.shape[0])}


def main():
    A = np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1]
    ])
    c = np.array([1, 1, 0, 0, 0])
    b = np.array([1, 3, 2])
    x_b = np.array([0, 0, 1, 3, 2])
    J_b = np.array([3, 4, 5]) - 1
    basic_phase(A, b, c, x_b, J_b)
    print("Optimal plan: ", x_b, J_b + 1)


if __name__ == '__main__':
    main()
