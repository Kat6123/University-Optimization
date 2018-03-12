#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class SimplexException(Exception):
    def __init__(self, msg):
        super(SimplexException, self).__init__(msg)


def check_inverse(inv, x, pos):
    l_v = inv.dot(x)
    return l_v[pos] != 0


def find_inverse(matrix, inverse, x, pos, dim):
    if not check_inverse(inverse, x, pos):
        raise SimplexException("Inverse matrix doesn't exist")

    l1 = inverse.dot(x)
    l_pos = l1[pos]

    l1[pos] = -1

    l2 = l1 * (-1 / l_pos)
    E = np.eye(dim)
    E[:, pos] = np.ndarray.transpose(l2)

    return E.dot(inverse)
