#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


def iterate(y, J_b, A, b, c):
    n = A.shape[1]
    A_b_inverse = np.linalg.inv(A[:, J_b])

    kappa = np.zeros(n)
    kappa[J_b] = np.dot(A_b_inverse, b)

    if np.all(kappa >= 0):
        return kappa

    basic_ind = np.argmin(kappa)
    j_basic_ind = np.where(J_b == basic_ind)[0][0]

    delta_y = A_b_inverse[:, j_basic_ind]

    mask = np.setdiff1d(np.arange(n), J_b)
    mu = np.dot(delta_y, A[:, mask])

    if not np.any(mu < 0):
        print("Problem is incompatible")
        raise Exception()

    mu_cond = np.array([(i, m) for (i, m) in zip(mask, mu) if m < 0])

    sigma_min_ind, sigma_min = mu_cond[0]
    for ind, m in mu_cond:
        ind = int(ind)
        sigma = (c[ind] - np.dot(A[:, ind], y)) / m
        if sigma < sigma_min:
            sigma_min = sigma
            sigma_min_ind = ind

    J_b[j_basic_ind] = sigma_min_ind
    y += sigma_min * delta_y


def main():
    A = np.array([
        [-2, -1, 1, -7, 0, 0, 0, 2],
        [4, 2, 1, 0, 1, 5, -1, -5],
        [1, 1, 0, -1, 0, 3, -1, 1]])
    b = np.array([-2, -4, -2])
    c = np.array([5, 2, 3, -16, 1, 3, -3, -12])
    J_b = np.array([1, 2, 3]) - 1

    y = np.dot(c[J_b], np.linalg.inv(A[:, J_b]))

    result = iterate(y, J_b, A, b, c)
    while result is None:
        result = iterate(y, J_b, A, b, c)

    print("Optimal plan: ", result)


if __name__ == '__main__':
    main()
