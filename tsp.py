import numpy as np

def two_opt(route, mat: np.matrix):
    def opt2CostDelta(mat, i, ni, j, nj):
        return mat[i, j] + mat[ni, nj] - mat[i, ni] - mat[j, nj]

    imp = True
    changed = False
    n = len(route)
    while imp:
        imp = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if opt2CostDelta(mat, route[i], route[i + 1], route[j], route[j + 1]) < 0:
                    route[i + 1:j + 1] = route[j:i:-1]
                    changed = imp = True
        for i in range(1, n - 2):
            if opt2CostDelta(mat, route[i], route[i + 1], route[-1], route[0]) < 0:
                route[:] = np.roll(route, -1)
                route[i: -1] = route[n - 2:i - 1:-1]
                route[:] = np.roll(route, 1)
                changed = imp = True
    return changed


def three_opt(route, mat: np.matrix):
    imp = True
    changed = False
    n = len(route)
    while imp:
        imp = False
        for i in range(n - 2):
            ri = route[i]
            rni = route[i + 1]
            for j in range(i + 2, n - 1):
                rj = route[j]
                rnj = route[j + 1]
                for k in range(j + 2, n - 1):
                    rk = route[k]
                    rnk = route[k + 1]
                    rem = mat[ri, rni] + mat[rj, rnj] + mat[rk, rnk]
                    if mat[ri, rnj] + mat[rk, rni] + mat[rj, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[j + 1:k + 1]) + list(route[i + 1:j + 1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        break
                    if mat[ri, rnj] + mat[rk, rj] + mat[rni, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[j + 1:k + 1]) + list(route[j:i:-1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        break

                    if mat[ri, rk] + mat[rnj, rni] + mat[rj, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[k:j:-1]) + list(route[i + 1:j + 1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        # print('3opt 3')
                        break

                    if mat[ri, rj] + mat[rni, rk] + mat[rnj, rnk] < rem:
                        r = list(route[0:i + 1]) + list(route[j:i:-1]) + list(route[k:j:-1]) + list(
                            route[k + 1:])
                        route[:] = r
                        changed = imp = True
                        break
                if imp:
                    break
    return changed
