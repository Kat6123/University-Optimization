#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class LimitException(Exception):
    def __init__(self, msg):
        super(LimitException, self).__init__(msg)


def get_H_matrix(A, D_E, J_E_op):
    A_E = A[:, J_E_op]
    A_E_T = A_E.transpose()
    zero_matrix = np.full((A_E.shape[0], A_E.shape[0]), 0)

    top = np.concatenate((D_E, A_E_T), axis=1)
    bottom = np.concatenate((A_E, zero_matrix), axis=1)

    return np.concatenate((top, bottom))


def construct_l_direction(A, D, J_E_op, j_0, size):
    D_E = D[np.ix_(J_E_op, J_E_op)]
    H = get_H_matrix(A, D_E, J_E_op)

    b_E = np.concatenate((D[J_E_op, j_0], A[:, j_0])) * (-1)

    H_inv = np.linalg.inv(H)
    y = H_inv.dot(b_E)

    l_vector = np.full((size, ), 0, dtype='float')
    l_vector[j_0] = 1

    l_vector[J_E_op] = y[:J_E_op.shape[0]]
    l_vector = np.around(l_vector, decimals=15)
    print("L: ", l_vector)
    return l_vector


def calc_estimate_vector(A, D, c, J_op, x):
    c_x = c + x.dot(D)

    A_op_inverse = np.linalg.inv(A[:, J_op])
    c_op_x = c_x[J_op]

    u_x = -c_op_x.dot(A_op_inverse)

    delta = u_x.dot(A) + c_x

    return delta


def _calc_theta_j0(D, l_vector, delta, j_0):
    small_delta = l_vector.dot(D).dot(l_vector)
    if small_delta > 0:
        return abs(delta[j_0]) / small_delta
    else:
        return np.inf


def find_min_theta(D, J_E_op, delta, l_v, x, j_0):
    theta_j0 = _calc_theta_j0(D, l_v, delta, j_0)

    theta_list = [(j, -x[j] / l_v[j]) if l_v[j] < 0
                  else (j, np.inf) for j in J_E_op]
    theta_list.append((j_0, theta_j0))
    print("Theta_list: ", theta_list)
    return min(theta_list, key=lambda t: t[1])


def build_new_J(J_op, J_E_op, j_0, j_E, A):
    if j_0 == j_E:
        return J_op, np.insert(J_E_op, J_E_op.size, j_E)
    elif j_E in np.setdiff1d(J_E_op, J_op):
        return J_op, np.setdiff1d(J_E_op, [j_E])
    elif j_E in J_op:
        s = np.argwhere(J_op == j_E)[0][0]
        j_plus_set = np.setdiff1d(J_E_op, J_op)
        A_op_inverse = np.linalg.inv(A[:, J_op])

        J_op_new = np.copy(J_op)
        J_E_op_new = np.copy(J_E_op)

        if np.array_equal(J_op, J_E_op):
            J_op_new[s] = j_0
            J_E_op_new[s] = j_0
            return J_op_new, J_E_op_new

        for j_plus in j_plus_set:
            mul = A_op_inverse.dot(A[:, j_plus])
            if mul[s] != 0:
                break
        else:
            J_op_new[s] = j_0
            J_E_op_new[s] = j_0
            return J_op_new, J_E_op_new

        J_op_new[s] = j_plus
        print("j_plus", j_plus)
        return J_op_new, np.setdiff1d(J_E_op, [j_E])


def find_j0(delta, J_E_op):
    for (i, val) in enumerate(delta):
        if i not in J_E_op and val < 0:
            return i


def make_iter(A, b, c, D, J_op, J_E_op, x):
    delta = calc_estimate_vector(A, D, c, J_op, x)
    print("delta: ", delta, J_E_op)
    delta[J_E_op] = 0
    if np.all(delta >= 0):
        return (True, x, J_op, J_E_op)

    j_0 = find_j0(delta, J_E_op)
    print("j_0: ", j_0)
    l_vector = construct_l_direction(A, D, J_E_op, j_0, x.shape[0])
    # print(l_vector)
    j_E, min_theta = find_min_theta(D, J_E_op, delta, l_vector, x, j_0)
    if np.isinf(min_theta):
        raise LimitException("Unlimited function")

    x_new = x + min_theta * l_vector
    # print("New x ", x_new)
    J_op_new, J_E_op_new = build_new_J(J_op, J_E_op, j_0, j_E, A)
    # print(J_op_new, J_E_op_new)
    return (False, x_new, J_op_new, J_E_op_new)


def solve_quadratic_task(A, b, c, D, J_op, J_E_op, x):
    # add not to make_iter
    result, x_n, J_op_n, J_E_op_n = make_iter(A, b, c, D, J_op, J_E_op, x)
    print(result, x_n, J_op_n, J_E_op_n)
    while not result:
        result, x_n, J_op_n, J_E_op_n = make_iter(
            A, b, c, D, J_op_n, J_E_op_n, x_n)
        print(result, x_n, J_op_n, J_E_op_n)

    return x_n


def main():
    # Lab task
    # A = np.array([
    #     [1, 0, 2, 1],
    #     [0, 1, -1, 2]
    # ])
    # D = np.array([
    #     [2, 1, 1, 0],
    #     [1, 1, 0, 0],
    #     [1, 0, 1, 0],
    #     [0, 0, 0, 0]
    # ])
    # c = np.array([-8, -6, -4, -6])
    # b = np.array([2, 3])
    #
    # J_op = np.array([1, 2])
    # J_E_op = np.array([1, 2])
    # x = np.array([2, 3, 0, 0], dtype='float')

    # Task0
    # A = np.array([
    #     [1, 2, 0, 1, 0, 4, -1, -3],
    #     [1, 3, 0, 0, 1, -1, -1, 2],
    #     [1, 4, 1, 0, 0, 2, -2, 0]
    # ], dtype='float')
    # B = np.array([
    #     [1, 1, -1, 0, 3, 4, -2, 1],
    #     [2, 6, 0, 0, 1, -5, 0, -1],
    #     [-1, 2, 0, 0, -1, 1, 1, 1]
    # ], dtype='float')
    # d = np.array([7, 3, 3], dtype='float')
    #
    # D = B.transpose().dot(B)
    # c = -d.dot(B)
    # b = np.array([4, 5, 6], dtype='float')
    #
    # J_op = np.array([3, 4, 5])
    # J_E_op = np.array([3, 4, 5])
    # x = np.array([0, 0, 6, 4, 5, 0, 0, 0], dtype='float')
    #

    # Task1
    # A = np.array([
    #     [11, 0, 0, 1, 0, -4, -1, 1],
    #     [1, 1, 0, 0, 1, -1, -1, 1],
    #     [1, 1, 1, 0, 1, 2, -2, 1]
    # ], dtype='float')
    # B = np.array([
    #     [1, -1, 0, 3, -1, 5, -2, 1],
    #     [2, 5, 0, 0, -1, 4, 0, 0],
    #     [-1, 3, 0, 5, 4, -1, -2, 1]
    # ], dtype='float')
    # d = np.array([6, 10, 9], dtype='float')
    #
    # D = B.transpose().dot(B)
    # c = -d.dot(B)
    # b = np.array([8, 2, 5], dtype='float')
    #
    # J_op = np.array([1, 2, 3])
    # J_E_op = np.array([1, 2, 3])
    # x = np.array([0.7273, 1.2727, 3.0000, 0, 0, 0, 0, 0], dtype='float')

    # Task2
    # A = np.array([
    #     [2, -3, 1, 1, 3, 0, 1, 2],
    #     [-1, 3, 1, 0, 1, 4, 5, -6],
    #     [1, 1, -1, 0, 1, -2, 4, 8]
    # ], dtype='float')
    # B = np.array([
    #     [1, 0, 0, 3, -1, 5, 0, 1],
    #     [2, 5, 0, 0, 0, 4, 0, 0],
    #     [-1, 9, 0, 5, 2, -1, -1, 5]
    # ], dtype='float')
    #
    # D = B.transpose().dot(B)
    # c = np.array([-13, -217, 0, -117, -27, -71, 18, -99], dtype='float')
    #
    # b = np.array([8, 4, 14], dtype='float')
    #
    # J_op = np.array([2, 5, 8])
    # J_E_op = np.array([2, 5, 8])
    # x = np.array([0, 2, 0, 0, 4, 0, 0, 1], dtype='float')

    # Task3
    # A = np.array([
    #     [0, 2, 1, 4, 3, 0, -5, -10],
    #     [-1, 3, 1, 0, 1, 3, -5, -6],
    #     [1, 1, 1, 0, 1, -2, -5, 8]
    # ], dtype='float')
    # b = np.array([6, 4, 14], dtype='float')
    # c = np.array([1, 3, -1, 3, 5, 2, -2, 0], dtype='float')
    #
    # D = np.eye(8)
    # D[2][2] = 0
    # D[6][6] = 0
    # print(D)
    # J_op = np.array([2, 5, 8])
    # J_E_op = np.array([2, 5, 8])
    # x = np.array([0, 2, 0, 0, 4, 0, 0, 1], dtype='float')

    optimal = solve_quadratic_task(A, b, c, D, J_op - 1, J_E_op - 1, x)
    print(c.dot(optimal) + 0.5 * optimal.dot(D).dot(optimal))


if __name__ == '__main__':
    main()
