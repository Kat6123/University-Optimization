#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def north_west_corner_method(a, b, C):
    U_B = []
    X = np.full(C.shape, 0)

    i = j = 0

    while i < a.size and j < b.size:
        X[i, j] = min(a[i], b[j])
        a[i] -= X[i, j]
        b[j] -= X[i, j]
        U_B.append((i, j))
        if a[i] == 0:
            i += 1
        else:
            j += 1
        # print(X)
        # print("A: ", a)
        # print("B: ", b)
        # print("U_B: ", U_B)
    return X, U_B


def find_potentials(X, U_B, C):
    m, n = X.shape
    A = np.full((len(U_B) + 1, m + n), 0)
    B = np.full((len(U_B) + 1, ), 0)
    for s in xrange(len(U_B)):
        i, j = U_B[s]
        A[s][i] = 1
        A[s][m + j] = 1
        B[s] = C[i][j]

    A[s + 1][1] = 1
    B[s + 1] = 0
    return np.linalg.solve(A, B)


def _get_ordered_signed_cycle(U_cycle, start):
    SIGN = ["+", "-"]
    U_ordered = [(start, "+")]
    del U_cycle[U_cycle.index(start)]

    while len(U_cycle) != 0:
        current_point = U_ordered[-1][0]
        current_sign = U_ordered[-1][1]

        for point in U_cycle:
            if point[0] == current_point[0] or point[1] == current_point[1]:
                sign_index = SIGN.index(current_sign) - 1
                U_ordered.append((point, SIGN[sign_index]))

                point_index = U_cycle.index(point)
                del U_cycle[point_index]
                break

    # print(U_ordered)
    return U_ordered


def find_cycle(U_B, shape):
    M = np.full(shape, 0)
    for i, j in U_B:
        M[i][j] = 1

    was_found = True

    while was_found:
        # print(M)
        was_found = False
        for m in xrange(shape[0]):
            if list(M[m]).count(1) == 1:
                M[m].fill(-1)
                was_found = True
        for n in xrange(shape[1]):
            # print(M[:, n])
            # print(list(M[:, n]).count(1))
            if list(M[:, n]).count(1) == 1:
                M[:, n].fill(-1)
                was_found = True

    U_cycle = []
    for m in xrange(shape[0]):
        for n in xrange(shape[1]):
            if M[m, n] == 1:
                U_cycle.append((m, n))

    return _get_ordered_signed_cycle(U_cycle, U_B[-1])


def find_min_thetta(X, U_cycle):
    U_x = []
    for point, sign in U_cycle:
        if sign == "-":
            U_x.append((X[point], point))

    return min(U_x, key=lambda val: val[0])


def update_X(X, U_cycle, thetta):
    for point, sign in U_cycle:
        if sign == "+":
            X[point] += thetta
        else:
            X[point] -= thetta


def potential_method(X, U_B, C):
    end = False
    m, n = X.shape
    i_n, j_n = -1, -1

    u_v = find_potentials(X, U_B, C)

    for i in xrange(m):
        if not end:
            for j in xrange(n):
                if (i, j) not in U_B and u_v[i] + u_v[m + j] > C[i, j]:
                    end = True
                    i_n, j_n = i, j
                    break
    if not end:
        return True

    U_B.append((i_n, j_n))
    U_cycle = find_cycle(U_B, X.shape)
    thetta, j_star = find_min_thetta(X, U_cycle)
    update_X(X, U_cycle, thetta)

    # print(U_B)
    del_index = U_B.index(j_star)
    del U_B[del_index]

    return False


def solve_transport_task(a, b, C):
    X, U_B = north_west_corner_method(a, b, C)
    print("Iteration #0")
    print(X, U_B)
    iteration = 0

    while not potential_method(X, U_B, C):
        iteration += 1
        print("\nIteration #{}".format(iteration))
        print("X: ", X)
        print("U_B: ", U_B)

    return X


def cost(C, X):
    temp = np.multiply(C, X)
    # print(temp)
    return np.sum(temp)


def fix_balance_condition(a, b, C):
    diff = np.sum(a) - np.sum(b)
    if diff:
        if diff > 0:
            b = np.append(b, diff)
            print(C)
            print(np.zeros(a.size))
            C = np.c_[C, np.zeros(a.size)]

        else:
            a = np.append(a, abs(diff))
            C = np.r_[C, np.zeros(b.size)]
            # C[-1] = np.zeros(b.size)

    return a, b, C


def main():
    # a = np.array([100, 300, 300])
    # b = np.array([300, 200, 200])
    # C = np.array([
    #     [8, 4, 1],
    #     [8, 4, 3],
    #     [9, 7, 5]
    # ])

    # a = np.array([20, 30, 25])
    # b = np.array([10, 10, 10, 10, 10])
    # C = np.array([
    #     [2, 8, -5, 7, 10],
    #     [11, 5, 8, -8, -4],
    #     [1, 3, 7, 4, 2]
    # ])

    a = np.array([53, 20, 45, 38])
    b = np.array([15, 31, 10, 3, 18])
    C = np.array([
        [3, 0, 3, 1, 6],
        [2, 4, 10, 5, 7],
        [-2, 5, 3, 2, 9],
        [1, 3, 5, 1, 9]
    ])

    # b = np.array([20, 5, 6, 11])
    # a = np.array([13, 5, 7, 9, 10])
    # C = np.array([
    #     [2, 6, 8, -3],
    #     [3, 2, 12, 4],
    #     [7, 2, 5, 7],
    #     [9, 2, 14, 9],
    #     [8, 7, 8, 8]
    # ])

    a, b, C = fix_balance_condition(a, b, C)
    print(a, b, C)
    X = solve_transport_task(a, b, C)
    print("\nOptimal plan: \n{}".format(X))
    print("\nCost: \n{}".format(cost(C, X)))


if __name__ == '__main__':
    main()
