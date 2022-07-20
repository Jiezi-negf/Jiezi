# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import copy

from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA import operator as op
from Jiezi.Physics.common import *
import numpy as np
from scipy.optimize import root, fsolve


def ef_solver(u_init, N_GP_T, cell_co, dos, density_n, ef_init, mark_list, z_length, E_list, TOL):
    """
    loop all the cell.
    In every loop, first call the gauss_xyz to get four gauss_point_xyz of the specific cell
    then based on the gauss_xyz, find the values of dos and density_n
    then compute the ef on every gauss point of cell and store them.
    :param u_init:
            u_init[cell_index] = [u_dof1, u_dof2, u_dof3, u_dof4]
            at the beginning of poisson solver, u_init is the initial value of newton iteration
    :param N_GP_T:
            values of shape functions on GPs, cooperate with u_init to get values of u on GPs
            N_GP_T[cell_index] = vec, vec is vector_numpy(4), which is a column vector
    :param cell_co:
            coefficients of shape functions in each cell
            cell_co[cell_index] = [a[x,x,x,x] b[...] c[...] d[...]]
    :param dos: from the output of GF, dos[ee][zz] = D(E(ee),z)
    :param density_n: from the output of GF, dos[zz] = n(z)
    :param ef_init:
            initial value of ef in the iteration of ef_solver
            ef_init[cell_index] = [value1, value2, value3, value4], may be[0,0,0,0]
    :param mark_list: the mark value list of all cells
    :param z_length:
            the whole length of the tube along the z axis, used by "def find_dos" and "def find_n"
    :param E_list: energy sampling points set, which is same as the GF solver
    :param TOL: tolerance of ef_solver interation
    :return:
            1 ef: ef[cell_index] = [value1, value2, value3, value4]

            2 dos_GP_list: dos value on each gauss point
            dos_GP_list[cell_index] = dos_cell_i, dos_cell_i[GP_index] = dos_GP,
            dos_GP[ee] = dos of value E_list[ee].
    """
    # every ef[i] stores a list [], which stores the values of gf on four GPs of cell i
    ef = [None] * len(cell_co)
    dos_GP_list = [None] * len(cell_co)
    n_GP_list = [None] * len(cell_co)
    u_GP_list = [None] * len(cell_co)
    for cell_index in range(len(cell_co)):
        dos_cell_i = [None] * 4
        n_cell_i  = [None] * 4
        gauss_point_xyz = gauss_xyz(cell_co[cell_index])
        # u_GP_cell is a list, the length of which is 4, which stores the phi value on the four GP of cell_index
        u_GP_cell = [None] * 4
        for GP_index in range(4):
            u_init_vec = vector_numpy(4)
            u_init_vec.copy(np.array([u_init[cell_index]]).transpose())
            u_GP_cell[GP_index] = op.vecdotvec(N_GP_T[GP_index], u_init_vec)
        u_GP_list[cell_index] = u_GP_cell

        for GP_index in range(4):
            coord = gauss_point_xyz[GP_index]
            dos_GP = find_dos(coord, dos, z_length)
            n_GP = find_n(coord, density_n, z_length)
            dos_cell_i[GP_index] = dos_GP
            n_cell_i[GP_index] = n_GP
        n_GP_list[cell_index] = n_cell_i
        dos_GP_list[cell_index] = dos_cell_i

    for cell_index in range(len(cell_co)):
        if mark_list[cell_index] != 1:
            ef_cell_i = [-1e2] * 4
            ef[cell_index] = ef_cell_i
            continue
        else:
            ef_cell_i = [None] * 4
            for GP_index in range(4):
                ef_i = brent(E_list, u_GP_list[cell_index][GP_index],
                             dos_GP_list[cell_index][GP_index], n_GP_list[cell_index][GP_index],
                             E_list[0], E_list[len(E_list) - 1], 1e-5, 1e-5)
                ef_cell_i[GP_index] = ef_i
                # # only compute the first point of every cell, reduce the amount of computation
                # if GP_index == 0:
                #     for i in range(1, 4):
                #         ef_cell_i[i] = ef_i
                # break
            ef[cell_index] = ef_cell_i

    # # this part uses the newton method to solve ef
    # for cell_index in range(len(cell_co)):
    #     ef_cell_i = [None] * 4
    #     dos_cell_i = [None] * 4
    #     gauss_point_xyz = gauss_xyz(cell_co[cell_index])
    #     # u_GP_cell is a list, the length of which is 4, which stores the phi value on the four GP of cell_index
    #     u_GP_cell = [None] * 4
    #     for i in range(4):
    #         u_init_vec = vector_numpy(4)
    #         u_init_vec.copy(np.array([u_init[cell_index]]).transpose())
    #         u_GP_cell[i] = op.vecdotvec(N_GP_T[i], u_init_vec)
    #
    #     for GP_index in range(4):
    #         u_GP = u_GP_cell[GP_index]
    #         coord = gauss_point_xyz[GP_index]
    #         dos_GP = find_dos(coord, dos, z_length)
    #         n_GP = find_n(coord, density_n, z_length)
    #         dos_cell_i[GP_index] = dos_GP
    #
    #
    #         # ef_i = ef_init[cell_index][GP_index]
    #         # while 1:
    #         #     ef_i1 = ef_i - KT * (integral_ef_up(u_GP, dos_GP, E_list, ef_i) - n_GP) \
    #         #             / integral_ef_bottom(u_GP, dos_GP, E_list, ef_i)
    #         #     error = abs(ef_i1 - ef_i)
    #         #     if error < TOL:
    #         #         break
    #         #     ef_i = ef_i1
    #         ef_i = brent(E_list, u_GP, dos_GP, n_GP, E_list[0], E_list[len(E_list) - 1], 1e-5, 1e-5)
    #         ef_cell_i[GP_index] = ef_i
    #
    #     ef[cell_index] = ef_cell_i
    #     dos_GP_list[cell_index] = dos_cell_i

    return ef, dos_GP_list


def gauss_xyz(co_shapefunc):
    """
    give the four sets of coefficients of four shape functions of one cell
    return the coordinates of four gauss points in physical space xyz
    :param co_shapefunc: [[a1,a2,a3,a4],[b1,b2...]...]
    :return: gauss_point_xyz: [[(x), (y), (z)],[(x),(y),(z)],...]
    """
    gauss_point_xyz = [None] * 4
    p = 0.58541
    q = 0.138197
    gauss_point_base = [p, q, q, q]
    for i in range(4):
        temp = gauss_point_base.copy()
        temp[i] = p
        gauss_point = vector_numpy(3)
        # gauss_point is a column vector
        gauss_point.copy([temp[1:]])
        gauss_point = gauss_point.trans()

        ma_0 = matrix_numpy(4, 4)
        ma_0.copy(co_shapefunc)
        # transpose to ma_0 = [[a1,b1,c1,d1],[a2,b2,c2,d2],...,[a4,b4,c4,d4]]
        ma_0 = ma_0.trans()
        # ma_1 is [[b2,c2,d2],[b3,c3,d3],[b4,c4,d4]]
        # ma is inv([[b2,c2,d2],[b3,c3,d3],[b4,c4,d4]])
        ma_1 = matrix_numpy(3, 3)
        ma_1.copy(ma_0.get_value(1, 4, 1, 4))
        ma = op.inv(ma_1)
        # vec_0 is the first column of ma_0
        # vec_0 = [[a2],[a3],[a4]]
        vec_0 = vector_numpy(3)
        vec_0.copy(ma_0.get_value(1, 4, 0, 1))
        # gauss_point = [[alpha],[beta],[gamma]]
        # vec is [[alpha],[beta],[gamma]]-[[a2],[a3],[a4]]
        vec = op.addvec(gauss_point, vec_0.nega())
        # [[gauss_point_x],[gauss_point_y],[gauss_point_z]] equals ma * vec
        gauss_point_xyz[i] = op.matmulvec(ma, vec).get_value().transpose()[0].tolist()
    return gauss_point_xyz


def find_dos(coord, dos, z_length):
    """
    based on the GPs' coordination on xyz space, get the layer which it belongs to
    then get the dos which is just a function of energy.
    :param coord: coordination of one gauss point
    :param dos: from the output of GF, dos[ee][zz] = D(E(ee),z)
    :param z_length: the whole length of the tube along the z axis
    :return: dos_GP[ee1, ee2, ...], the index of which is energy
    """
    x, y, z = coord
    length_layer = z_length / len(dos[0])
    layer_index = int(z.real // length_layer)
    dos_GP = [row[layer_index] for row in dos]
    return dos_GP


def find_n(coord, density_n, z_length):
    """
    based on the GPs' coordination on xyz space, get the layer which it belongs to.
    then get the density of electrons.
    :param coord: coordination of one gauss point
    :param density_n: from the output of GF, dos[zz] = n(z)
    :param z_length: the whole length of the tube along the z axis
    :return: n_GP, a float type data
    """
    x, y, z = coord
    length_layer = z_length / len(density_n)
    layer_index = int(z.real // length_layer)
    n_GP = density_n[layer_index].real
    return n_GP


def integral_ef_up(start, dos_GP, E_list, ef_i):
    result = 0.0
    E_step = E_list[1] - E_list[0]
    diff = start - E_list[0]
    start_normal = int(diff // E_step)
    if diff <= 0:
        start_index = 0
    else:
        start_index = start_normal
    end_index = len(E_list)
    for ee in range(start_index, end_index - 1):
        result += (dos_GP[ee] * fermi(E_list[ee] - ef_i)
                   + dos_GP[ee + 1] * fermi(E_list[ee + 1] - ef_i)) * E_step / 2
    return result


def integral_ef_bottom(start, dos_GP, E_list, ef_i):
    result = 0.0
    E_step = E_list[1] - E_list[0]
    diff = start - E_list[0]
    start_normal = int(diff // E_step)
    if diff <= 0:
        start_index = 0
    else:
        start_index = start_normal
    end_index = len(E_list)
    for ee in range(start_index, end_index - 1):
        result += (dos_GP[ee] * fermi(E_list[ee] - ef_i) * (1 - fermi(E_list[ee] - ef_i))
                   + dos_GP[ee + 1] * fermi(E_list[ee + 1] - ef_i) * (1 - fermi(E_list[ee + 1] - ef_i))
                   ) * E_step / 2
    return result


def func_F(E_list, phi, dos, density_n, ef):
    result = 0.0
    E_step = E_list[1] - E_list[0]
    diff = -phi - E_list[0]
    start_normal = int(diff.real // E_step)
    if diff <= 0:
        start_index = 0
    elif -phi > E_list[len(E_list) - 1]:
        start_index = len(E_list)
    else:
        start_index = start_normal
    end_index = len(E_list)
    for ee in range(start_index, end_index - 1):
        result += (dos[ee] * fermi(E_list[ee] - ef)
                   + dos[ee + 1] * fermi(E_list[ee + 1] - ef)) * E_step / 2
    result -= density_n
    return result


def func_derivative_F(E_list, phi, dos, ef):
    result = 0.0
    E_step = E_list[1] - E_list[0]
    diff = -phi - E_list[0]
    start_normal = int(diff.real // E_step)
    if diff <= 0:
        start_index = 0
    elif -phi > E_list[len(E_list) - 1]:
        start_index = len(E_list)
    else:
        start_index = start_normal
    end_index = len(E_list)
    for ee in range(start_index, end_index - 1):
        result += (dos[ee] * fermi(E_list[ee] - ef) * (1 - fermi(E_list[ee] - ef))
                   + dos[ee + 1] * fermi(E_list[ee + 1] - ef) * (1 - fermi(E_list[ee + 1] - ef))) * E_step / 2
    result = result / KT
    return result

# def func_F(E_list, phi, dos, density_n, ef):
#     result = ef**3 - 2*ef**2 - +5*ef - 15
#     return result
# def f(ef):
#     res = ef**3 - 2*ef**2 - +5*ef - 15
#     return res


def secant(E_list, phi, dos, density_n, ef_0, ef_1):
    f_0 = func_F(E_list, phi, dos, density_n, ef_0)
    f_1 = func_F(E_list, phi, dos, density_n, ef_1)
    ef_2 = ef_0 - f_0 * (ef_1 - ef_0) / (f_1 - f_0)
    return ef_2


def IQI(E_list, phi, dos, density_n, ef_0, ef_1, ef_2):
    f_0 = func_F(E_list, phi, dos, density_n, ef_0)
    f_1 = func_F(E_list, phi, dos, density_n, ef_1)
    f_2 = func_F(E_list, phi, dos, density_n, ef_2)
    result = f_1 * f_2 / (f_0 - f_1) / (f_0 - f_2) * ef_0 \
             + f_0 * f_2 / (f_1 - f_0) / (f_1 - f_2) * ef_1 \
             + f_0 * f_1 / (f_2 - f_0) / (f_2 - f_1) * ef_2
    return result


def swap(x1, x2):
    return x2, x1


def between(a, b, x):
    if a <= x <= b or b <= x <= a:
        return True
    else:
        return False


def brent(E_list, phi, dos, density_n, a, b, tol_brent_residual=1e-5, tol_brent_bisection=1e-5):
    f_a = func_F(E_list, phi, dos, density_n, a)
    f_b = func_F(E_list, phi, dos, density_n, b)
    # test if a and b are set reasonably
    if f_a * f_b > 0:
        print("ERROR: the sign of f(a) and f(b) must be different!")
    # make sure f(b) is closer to 0 than f(a), if not, swap a and b
    if abs(f_a) < abs(f_b):
        a, b = swap(a, b)
    # initialize c and d, they both equal to a at the first loop
    c = a
    d = a
    # set this variable to count the number of iteration
    iter_brent = 0
    # set the flag initial value false, represent that the bisection method has not been used
    flag_bisection = False
    # start the loop
    while 1:
        # make sure f(b) is closer to 0 than f(a), if not, swap a and b
        if abs(func_F(E_list, phi, dos, density_n, a)) < abs(func_F(E_list, phi, dos, density_n, b)):
            a, b = swap(a, b)
        # test if the iteration is converged
        if abs(a - b) < tol_brent_bisection or abs(func_F(E_list, phi, dos, density_n, b)) < tol_brent_residual:
            # print("the number of iteration in brent method is:", iter_brent)
            return b
        # compute f(a), f(b), f(c), m, k
        f_a = func_F(E_list, phi, dos, density_n, a)
        f_b = func_F(E_list, phi, dos, density_n, b)
        f_c = func_F(E_list, phi, dos, density_n, c)
        m = (a + b) / 2
        f_m = func_F(E_list, phi, dos, density_n, m)
        k = (3 * a + b) / 4
        # compute s
        if f_a != f_c and f_b != f_c:
            s = IQI(E_list, phi, dos, density_n, a, b, c)
        elif b != c and f_b != f_c:
            s = secant(E_list, phi, dos, density_n, b, c)
        else:
            s = m
        f_s = func_F(E_list, phi, dos, density_n, s)
        # compute flag_ill
        if flag_bisection is True:
            flag_ill = abs(s - b) < (1 / 2) * abs(b - c)
        else:
            flag_ill = (c == d or abs(s - b) < (1 / 2) * abs(c - d))
        # renew the a b c
        if flag_ill and between(k, b, s) and f_s * f_a < 0:
            flag_bisection = False
            d = copy.deepcopy(c)
            c = copy.deepcopy(b)
            b = copy.deepcopy(s)
            iter_brent += 1
            continue
        if flag_ill and between(k, b, s) and f_s * f_a > 0:
            flag_bisection = False
            d = copy.deepcopy(c)
            c = copy.deepcopy(b)
            a = copy.deepcopy(b)
            b = copy.deepcopy(s)
            iter_brent += 1
            continue
        if f_m * f_a < 0:
            flag_bisection = True
            d = copy.deepcopy(c)
            c = copy.deepcopy(b)
            b = copy.deepcopy(m)
            iter_brent += 1
            continue
        else:
            flag_bisection = True
            d = copy.deepcopy(c)
            c = copy.deepcopy(b)
            a = copy.deepcopy(b)
            b = copy.deepcopy(m)
            iter_brent += 1
            continue



