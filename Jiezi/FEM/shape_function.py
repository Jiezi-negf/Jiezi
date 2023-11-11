# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


def shape_function(dof_coord):
    """
    compute the coefficients of four shape functions of specific cell based on its degree of freedom information
    :param dof_coord: the coordinates of every dof of one specific cell_vol
    data format is [[x1,y1,z1],[x2,...],...,[x4,...]]
    :return: the coefficients of four shape functions of specific cell_vol
    N = a + bx + cy + dz
    data format is four lists. a[x,x,x,x] b[...] c[...] d[...]
    """
    # coord_x, coord_y, coord_z = coord
    x = [xx[0] for xx in dof_coord]
    y = [xx[1] for xx in dof_coord]
    z = [xx[2] for xx in dof_coord]
    a = [float] * 4
    b = [float] * 4
    c = [float] * 4
    d = [float] * 4
    for j in range(4):
        index = [0, 1, 2, 3]
        index.pop(j)
        k, l, m = index
        co1 = x[k] * (y[l] * z[m] - y[m] * z[l]) - y[k] * (x[l] * z[m] - x[m] * z[l]) + z[k] * (
                    x[l] * y[m] - x[m] * y[l])
        co2 = (y[l] * z[m] - y[m] * z[l]) - (y[k] * z[m] - y[m] * z[k]) + (y[k] * z[l] - y[l] * z[k])
        co3 = (x[l] * z[m] - x[m] * z[l]) - (x[k] * z[m] - x[m] * z[k]) + (x[k] * z[l] - x[l] * z[k])
        co4 = (x[l] * y[m] - x[m] * y[l]) - (x[k] * y[m] - x[m] * y[k]) + (x[k] * y[l] - x[l] * y[k])
        det_C = co1 - co2 * x[j] + co3 * y[j] - co4 * z[j]
        # det_D = co1 - co2 * coord_x + co3 * coord_y - co4 * coord_z
        a[j] = co1 / det_C
        b[j] = -co2 / det_C
        c[j] = co3 / det_C
        d[j] = -co4 / det_C
        co_shapefunc = [a, b, c, d]
    return co_shapefunc
