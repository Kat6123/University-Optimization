#!/usr/bin/env python
# -*- coding: utf-8 -*-


from scipy.optimize import linprog
import numpy as np

PRECISION = 3


# Calculate and return gradient in the test vector
def _calc_gradient(c, D, x_test):
    return np.array([c[i] + D[i].dot(x_test) for i in xrange(x_test.size)])


def _sift_active_constraints(constraints, x_test):
    active_indexes = []
    for i in xrange(len(constraints)):
        constraint_val = (constraints[i]["c"].dot(x_test) +
                          0.5 * x_test.dot(constraints[i]["D"]).dot(x_test) +
                          constraints[i]["l"])
        if round(constraint_val, PRECISION) == 0:
            active_indexes.append(i)

    return active_indexes


def _generate_bounds(x_test):
    return [(0, 1) if x == 0 else (-1, 1) for x in x_test]


def generate_linear_task(c, D, constraints, x_test):
    c = _calc_gradient(c, D, x_test)

    active_indexes = _sift_active_constraints(constraints, x_test)
    print(active_indexes)
    b = np.zeros((len(active_indexes), ))

    A = np.empty((x_test.size, ), float)
    for i in active_indexes:
        grad = _calc_gradient(constraints[i]["c"], constraints[i]["D"], x_test)
        A = np.vstack((A, grad))
    A = np.delete(A, (0), axis=0)

    bounds = _generate_bounds(x_test)
    return c, A, b, bounds


def _find_alpha(grad, l_0, x_slater, x_test):
    a = grad.dot(l_0)
    b = (x_slater - x_test).dot(grad)

    if b > 0:
        return -a/(2*float(b))
    else:
        return 1


def objective_f(c, D, x):
    return c.dot(x) + 0.5 * x.dot(D).dot(x)


def is_valid_plan(constraints, x):
    for c in constraints:
        val = c["c"].dot(x) + 0.5 * x.dot(c["D"]).dot(x) + c["l"]
        if val > 0:
            return False

    if np.any(x < 0):
        return False

    return True


def build_better_plan(grad, l_0, x_slater, x_test, c, D, constraints):
    alpha = _find_alpha(grad, l_0, x_slater, x_test)

    t = 1.
    current_obj = objective_f(c, D, x_test)
    while True:
        new_x = x_test + t * l_0 + alpha * t * (x_slater - x_test)
        new_obj = objective_f(c, D, new_x)
        if is_valid_plan(constraints, new_x) and new_obj < current_obj:
            return new_x

        t /= 2


# Constraints is a list of dicts such as [{"c": c1, "D": D1, "l": l1)}, ...]
#
# If the necessary optimality condition is satisfied,
# then we return thetest plan 'x_test' else a new plan
# with better value of the objective function
def solve_convex_problem(c, D, constraints, x_slater, x_test):
    c_lin, A, b, bounds = generate_linear_task(c, D, constraints, x_test)

    print(c_lin)
    res = linprog(c_lin, A_ub=A, b_ub=b, bounds=bounds)
    fun, l_0 = res["fun"], res["x"]

    if fun == 0:
        return x_test
    else:
        # Here c is used as gradient of the objective function
        return build_better_plan(c_lin, l_0, x_slater, x_test,
                                 c, D, constraints)


def main():
    # Task_0
    # c = np.array([-3, -3])
    # D = np.array([
    #     [2, 1],
    #     [1, 2]
    # ])
    # constraints = [
    #     {"c": np.array([1, -1]), "D": np.array([[1, 0], [0, 1]]),
    #         "l": -1},
    #     {"c": np.array([-1, 1]), "D": np.array([[1, 0.5], [0.5, 1]]),
    #         "l": -1.5},
    # ]
    # x_slater = np.array([0, 0])
    # x_test = np.array([0, 1])

    # Task_1
    c = np.array([-1, -1, -1, -1, -2, 0, -2, -3])
    B = np.array([
        [2, 1, 0, 4, 0, 3, 0, 0],
        [0, 4, 0, 3, 1, 1, 3, 2],
        [1, 3, 0, 5, 0, 4, 0, 4]
    ])
    D = B.transpose().dot(B)

    l_v = [0, -51.75, -436.75, -33.7813, -303.3750, -41.75]
    c_1 = np.array([0, 60, 80, 0, 0, 0, 40, 0])
    c_2 = np.array([2, 0, 3, 0, 2, 0, 3, 0])
    c_3 = np.array([0, 0, 80, 0, 0, 0, 0, 0])
    c_4 = np.array([0, -2, 1, 2, 0, 0, -2, 1])
    c_5 = np.array([-4, -2, 6, 0, 4, -2, 60, 2])

    B_1 = np.array([
        [0, 0, 0.5, 2.5, 1.0, 0, -2.5, -2.0],
        [0.5, 0.5, -0.5, 0, 0.5, -0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5, 0, 0.5, 1, 2.5, 4.0]
    ])
    B_2 = np.array([
        [1., 2., -1.5, 3., -2.5, 0, -1., -0.5],
        [-1.5, -0.5, -1., 2.5, 3.5, 3.0, -1.5, -0.5],
        [1.5, 2.5, 1.0, 1.0, 2.5, 1.5, 3.0, 0]
    ])
    B_3 = np.array([
        [0.75, 0.5, -1.0, 0.25, 0.25, 0, 0.25, 0.75],
        [-1., 1., 1., 0.75, 0.75, 0.5, 1., -0.75],
        [0.5, -0.25, 0.5, 0.75, 0.5, 1.25, -0.75, -0.25]
    ])
    B_4 = np.array([
        [1.5, -1.5, -1.5, 2., 1.5, 0, 0.5, -1.5],
        [-0.5, -2.5, -0.5, -1.0, -2.5, 2.5, 1., 2.],
        [-2.5, 1., -2., -1.5, -2.5, 0.5, 2.5, -2.5]
    ])
    B_5 = np.array([
        [1., 0.25, -0.5, 1.25, 1.25, -0.5, 0.25, -0.75],
        [-1., -0.75, -0.75, 0.5, -0.25, 1.25, 0.25, -0.5],
        [0, 0.75, 0.5, -0.5, -1., 1., -1., 1.]
    ])

    constraints = (
        {"c": c_1, "D": B_1.transpose().dot(B_1), "l": l_v[1]},
        {"c": c_2, "D": B_2.transpose().dot(B_2), "l": l_v[2]},
        {"c": c_3, "D": B_3.transpose().dot(B_3), "l": l_v[3]},
        {"c": c_4, "D": B_4.transpose().dot(B_4), "l": l_v[4]},
        {"c": c_5, "D": B_5.transpose().dot(B_5), "l": l_v[5]}
    )
    x_test = np.array([1, 0, 0, 2, 4, 2, 0, 0])
    x_slater = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    x = solve_convex_problem(c, D, constraints, x_slater, x_test)

    print("Base plan: {}".format(x_test))
    print("Objective func value: {}".format(objective_f(c, D, x_test)))
    if (x_test == x).all():
        print("Plan satisfies the necessary optimality condition")
    else:
        print("\nA new plan with the best objective func value was generated")
        print("\nNew plan: {}".format(x))
        print("Objective func value: {}".format(objective_f(c, D, x)))


if __name__ == '__main__':
    main()
