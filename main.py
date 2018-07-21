######################################################
# Suboptiaml solver for the set/vertex cover problem (CITATION)
# O(n^3) time with O(log n) suboptimality.
# We solve a linear 
# Problem:
#    minimize c^T x
#    subject to Ax >= e
# A is a set of sets. Each column represent a set of elements.
# e = (1, 1, ..., 1)
# x is a 0-1 vector with size of n
# Dual of the problem
#    maximize e^T y
#    subject to y^T A <= c 
# Js = {j | y^T Aj = cj}
# Solution <- x(Js)
# Then we greedily remove a redundant element from the solution.
#######################################################
import numpy as np
from scipy.optimize import linprog

def find_cover(A, c):
    # First we solve a dual of the linear relaxation.
    dual_c = -1 * np.ones(A.shape[0])
    dual_A = A.transpose()
    dual_b = c
    print(dual_A)
    ret = linprog(c=dual_c, A_ub=dual_A, b_ub=dual_b)
    y = ret.x
    costs = np.matmul(y.transpose(), A)
    print("y=", y)
    print("costs=", costs)
    print("c=", c)
    # x(Js) will be a feasible solution with a bound of O(log n).
    Js = []
    x = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        if costs[j] == c[j]:
            Js.append(j)
            x[j] = 1
    return x

def remove_redundant(A, cover):
    # Greedily remove a set from the cover.
    # b = Ax - e (b is a vector of overly covered elements)
    # Find a set in x which is a subset of b.
    # Remove it greedily.
    # Repeat this process until no set can be removed.

    c = cover
    while True:
        b = np.matmul(A, c) - np.ones(A.shape[0])
        print("b =", b)
        removed = False
        for j in range(A.shape[1]):
            if (A[:, j] <= b).all():
                c[j] = 0
                removed = True
                break
        if not removed:
            # No more set can be removed
            print("No more set can be removed.")
            break
                
    return c

# Main function
A = np.array([[1, 0], [1, 1], [0, 1]])

c = np.array([1, 1])

cover = find_cover(A, c)
print("Cover = ", cover)

c = remove_redundant(A, cover)
