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
from Jiezi.FEM import fermiIntegral
# import matplotlib.pyplot as plt


@ time_it
def ef_solver(u_init, N_GP_T, dos_GP_list, n_GP_list, p_GP_list, ef_init_n, ef_init_p,
              cnt_cell_list, E_list, Ec, Eg, TOL_ef):
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
            N_GP_T[GP_index] = vec, vec is vector_numpy(4), which is a column vector
    :param dos_GP_list: dos value on each gauss point
            dos_GP_list[cell_index] = dos_cell_i, dos_cell_i[GP_index] = dos_GP,
            dos_GP[ee] = dos of value E_list[ee].
    :param n_GP_list: n on each gauss point
            n_GP_list[cell_index] = n_cell_i, n_cell_i[GP_index] = n
    :param p_GP_list: p on each gauss point
            p_GP_list[cell_index] = p_cell_i, p_cell_i[GP_index] = p
    :param ef_init:
            initial value of ef in the iteration of ef_solver
            ef_init[cell_index] = [value1, value2, value3, value4], may be[0,0,0,0]
    :param mark_list: the mark value list of all cells
    :param E_list: energy sampling points set, which is same as the GF solver
    :param Ec: energy of the bottom of the conduction band
    :param Eg: energy of the band gap
    :param TOL_ef: tolerance of ef_solver interation
    :return:
            1 ef_n: ef_n[cell_index] = [value1, value2, value3, value4]
            2 ef_p: ef_p[cell_index] = [value1, value2, value3, value4]
            3 ef_flag: TRUE express that all the ef has been solved successfully
    """

    # set ef_flag TRUE, if ef of every point is computed successfully, ef_flag keep TRUE, otherwise the value
    # will be set FALSE
    ef_flag = True
    E_start = E_list[0]
    E_step = E_list[1] - E_list[0]
    zero_index = - int(E_start // E_step)
    E_list_n = E_list[zero_index:]
    E_list_p = E_list[0:zero_index + 1]
    for cnt_cell_index in cnt_cell_list:
        if not ef_flag:
            break
        for GP_index in range(4):
            # u_GP_cell is a list, the length of which is 4, which stores the phi value on the four GP of cell_index
            u_init_vec = vector_numpy(4, "float")
            u_init_vec.copy(u_init[cnt_cell_index].reshape(u_init[cnt_cell_index].shape[0], 1))
            u_GP = op.vecdotvec(N_GP_T[GP_index], u_init_vec)
            ef_n_i = brent("n", zero_index, E_list, E_list_n, E_step, Ec, Eg, u_GP,
                           dos_GP_list[cnt_cell_index][GP_index], max((1e-22, n_GP_list[cnt_cell_index][GP_index])),
                           E_list[0]-10, E_list[len(E_list) - 1]+10, 0, TOL_ef)
            ef_p_i = brent("p", zero_index, E_list, E_list_p, E_step, Ec, Eg, u_GP,
                           dos_GP_list[cnt_cell_index][GP_index], max((1e-22, p_GP_list[cnt_cell_index][GP_index])),
                           E_list[0]-10, E_list[len(E_list) - 1]+10, 0, TOL_ef)
            # ef_p_i = 1e2
            # test if the function has root, if there is no root in the interval, break the loop
            if ef_n_i == None:
                print("ef_solver failed during ef_n solution")
                ef_flag = False
                break
            if ef_p_i == None:
                print("ef_solver failed during ef_p solution")
                ef_flag = False
                break
            ef_init_n[cnt_cell_index, GP_index] = ef_n_i
            ef_init_p[cnt_cell_index, GP_index] = ef_p_i
            # only compute the first point of every cell, reduce the amount of computation
            if GP_index == 0:
                for i in range(1, 4):
                    ef_init_n[cnt_cell_index, i] = ef_n_i
                    ef_init_p[cnt_cell_index, i] = ef_p_i
            break

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
    return ef_init_n, ef_init_p, ef_flag


# def integral_ef_n_up(start, dos_GP, E_list, ef_i):
#     """
#     if I use newton iteration to solve Ef, this function will be used
#     """
#     result = 0.0
#     E_step = E_list[1] - E_list[0]
#     diff = start - E_list[0]
#     start_normal = int(diff // E_step)
#     if diff <= 0:
#         start_index = 0
#     else:
#         start_index = start_normal
#     end_index = len(E_list)
#     for ee in range(start_index, end_index - 1):
#         result += (dos_GP[ee] * fermi(E_list[ee] - ef_i)
#                    + dos_GP[ee + 1] * fermi(E_list[ee + 1] - ef_i)) * E_step / 2
#     return result
#
#
# def integral_ef_n_bottom(start, dos_GP, E_list, ef_i):
#     """
#     if I use newton iteration to solve Ef, this function will be used
#     """
#     result = 0.0
#     E_step = E_list[1] - E_list[0]
#     diff = start - E_list[0]
#     start_normal = int(diff // E_step)
#     if diff <= 0:
#         start_index = 0
#     else:
#         start_index = start_normal
#     end_index = len(E_list)
#     for ee in range(start_index, end_index - 1):
#         result += (dos_GP[ee] * fermi(E_list[ee] - ef_i) * (1 - fermi(E_list[ee] - ef_i))
#                    + dos_GP[ee + 1] * fermi(E_list[ee + 1] - ef_i) * (1 - fermi(E_list[ee + 1] - ef_i))
#                    ) * E_step / 2
#     return result


def func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef):
    dos_np = np.array(dos)
    E_list_np = np.array(E_list_np)
    if flag_np == "n":
        result = np.trapz(2 * dos_np[zero_index:] * fermi(E_list_np - phi - ef), dx=E_step) - density_np
    else:
        result = np.trapz(2 * dos_np[0:zero_index + 1] * (1 - fermi(E_list_np - phi - ef)), dx=E_step) - density_np
    # print("ef_solver func_F np.trapz:", result)
    # if flag_np == "n":
    #     diff = Ec - E_list[0]
    #     start_normal = int(diff.real // E_step)
    #     end_index = len(E_list) - 1
    #     if diff <= 0:
    #         start_index = 0
    #     elif Ec > E_list[end_index]:
    #         start_index = end_index
    #     else:
    #         start_index = start_normal
    #     for ee in range(start_index, end_index):
    #         result += (dos[ee] * fermi(E_list[ee] - phi - ef)
    #                    + dos[ee + 1] * fermi(E_list[ee + 1] - phi - ef)) * E_step / 2
    #     result -= density_np
    # else:
    #     diff = Ec - Eg - E_list[0]
    #     start_index = 0
    #     end_normal = int(diff.real // E_step)
    #     if diff <= 0:
    #         end_index = 0
    #     elif Ec - Eg > E_list[len(E_list) - 1]:
    #         end_index = len(E_list) - 1
    #     else:
    #         end_index = end_normal
    #     for ee in range(start_index, end_index):
    #         result += (dos[ee] * (1 - fermi(E_list[ee] - phi - ef))
    #                    + dos[ee + 1] * (1 - fermi(E_list[ee + 1] - phi - ef))) * E_step / 2
    #     result -= density_np

    # res = 0.0
    # if flag_np == "n":
    #     for ee in range(0, len(E_list) - 1):
    #         if E_list[ee] > 0:
    #             res += (dos[ee] * fermi(E_list[ee] - phi - ef)
    #                        + dos[ee + 1] * fermi(E_list[ee + 1] - phi - ef)) * E_step / 2
    # else:
    #     for ee in range(0, len(E_list) - 1):
    #         if E_list[ee] < 0:
    #             res += (dos[ee] * (1 - fermi(E_list[ee] - phi - ef))
    #                        + dos[ee + 1] * (1 - fermi(E_list[ee + 1] - phi - ef))) * E_step / 2
    # res -= density_np
    # print("ef_solver fun_F loop:", res)
    return result


# def func_derivative_F(E_list, phi, dos, ef):
#     """
#     if I use newton iteration to solve Ef, this function will be used
#     """
#     result = 0.0
#     E_step = E_list[1] - E_list[0]
#     diff = -phi - E_list[0]
#     start_normal = int(diff.real // E_step)
#     if diff <= 0:
#         start_index = 0
#     elif -phi > E_list[len(E_list) - 1]:
#         start_index = len(E_list)
#     else:
#         start_index = start_normal
#     end_index = len(E_list)
#     for ee in range(start_index, end_index - 1):
#         result += (dos[ee] * fermi(E_list[ee] - ef) * (1 - fermi(E_list[ee] - ef))
#                    + dos[ee + 1] * fermi(E_list[ee + 1] - ef) * (1 - fermi(E_list[ee + 1] - ef))) * E_step / 2
#     result = result / KT
#     return result

# # for testing the function "brent"
# def func_F(E_list, phi, dos, density_n, ef):
#     result = ef**3 - 2*ef**2 - +5*ef - 15
#     return result
# def f(ef):
#     res = ef**3 - 2*ef**2 - +5*ef - 15
#     return res


def secant(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_0, ef_1):
    f_0 = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_0)
    f_1 = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_1)
    ef_2 = ef_0 - f_0 * (ef_1 - ef_0) / (f_1 - f_0)
    return ef_2


def IQI(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_0, ef_1, ef_2):
    f_0 = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_0)
    f_1 = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_1)
    f_2 = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, ef_2)
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


def brent(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, a, b,
          tol_brent_residual=0, tol_brent_bisection=1e-11):
    """
    a robust algorithm for seeking roots of equation
    """
    f_a = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, a)
    f_b = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, b)
    # test if there is root between a and b, i.e. if the equation has solution
    if flag_np == "n" and f_a > 0 and f_b > 0:
        return
    if flag_np == "n" and f_a < 0 and f_b < 0:
        return
    if flag_np == "p" and f_a > 0 and f_b > 0:
        return
    if flag_np == "p" and f_a < 0 and f_b < 0:
        return
    # if f_a * f_b > 0:
    #     print("ERROR: the sign of f(a) and f(b) must be different!", "a:", a, "fa:", f_a, "b:", b, "f_b:", f_b)
    #     # print(phi, dos, density_n)
    #     # plt.plot(E_list, dos)
    #     # plt.show()
    #     return
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
        if abs(func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, a)) < \
                abs(func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, b)):
            a, b = swap(a, b)
        # test if the iteration is converged
        if abs(a - b) < tol_brent_bisection or abs(func_F(flag_np, zero_index, E_list, E_list_np, E_step,
                                                          Ec, Eg, phi, dos, density_np, b)) \
                < tol_brent_residual:
            # print("the number of iteration in brent method is:", iter_brent)
            return b
        # compute f(a), f(b), f(c), m, k
        f_a = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, a)
        f_b = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, b)
        f_c = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, c)
        m = (a + b) / 2
        f_m = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, m)
        k = (3 * a + b) / 4
        # compute s
        if f_a != f_c and f_b != f_c:
            s = IQI(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, a, b, c)
        elif b != c and f_b != f_c:
            s = secant(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, b, c)
        else:
            s = m
        f_s = func_F(flag_np, zero_index, E_list, E_list_np, E_step, Ec, Eg, phi, dos, density_np, s)
        # compute flag_ill
        if flag_bisection is True:
            flag_ill = abs(s - b) < (1 / 2) * abs(b - c)
        else:
            flag_ill = (c == d or abs(s - b) < (1 / 2) * abs(c - d))
        # renew a b c
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

@time_it
def ef_solver_analytical(u_init, N_GP_T, Nc, Nv, n_GP_list, p_GP_list, cnt_cell_list, Ec, Ev, ef_init_n, ef_init_p):
    for cnt_cell_index in cnt_cell_list:
        for GP_index in range(4):
            # u_GP_cell is a list, the length of which is 4, which stores the phi value on the four GP of cell_index
            u_init_vec = vector_numpy(4, "float")
            u_init_vec.copy(u_init[cnt_cell_index].reshape(u_init[cnt_cell_index].shape[0], 1))
            u_GP = op.vecdotvec(N_GP_T[GP_index], u_init_vec)
            ef_n_i = Ec - u_GP + KT * np.log(max((1e-22, n_GP_list[cnt_cell_index][GP_index])) / Nc)
            ef_p_i = Ev - u_GP - KT * np.log(max((1e-22, p_GP_list[cnt_cell_index][GP_index])) / Nv)
            ef_init_n[cnt_cell_index, GP_index] = ef_n_i
            ef_init_p[cnt_cell_index, GP_index] = ef_p_i
            # only compute the first point of every cell, reduce the amount of computation
            if GP_index == 0:
                for i in range(1, 4):
                    ef_init_n[cnt_cell_index, i] = ef_n_i
                    ef_init_p[cnt_cell_index, i] = ef_p_i
            break
    return ef_init_n, ef_init_p

@time_it
def ef_solver_degenerate(u_init, N_GP_T, Nc, Nv, n_GP_list, p_GP_list, cnt_cell_list, Ec, Ev, ef_init_n, ef_init_p):
    for cnt_cell_index in cnt_cell_list:
        for GP_index in range(4):
            # u_GP_cell is a list, the length of which is 4, which stores the phi value on the four GP of cell_index
            u_init_vec = vector_numpy(4, "float")
            u_init_vec.copy(u_init[cnt_cell_index].reshape(u_init[cnt_cell_index].shape[0], 1))
            u_GP = op.vecdotvec(N_GP_T[GP_index], u_init_vec)
            ef_n_i = Ec - u_GP + KT * fermiIntegral.inverseFermiIntegral(
                max((1e-22, n_GP_list[cnt_cell_index][GP_index])) / Nc)
            ef_p_i = Ev - u_GP - KT * fermiIntegral.inverseFermiIntegral(
                max((1e-22, p_GP_list[cnt_cell_index][GP_index])) / Nv)
            ef_init_n[cnt_cell_index, GP_index] = ef_n_i
            ef_init_p[cnt_cell_index, GP_index] = ef_p_i
            # only compute the first point of every cell, reduce the amount of computation
            if GP_index == 0:
                for i in range(1, 4):
                    ef_init_n[cnt_cell_index, i] = ef_n_i
                    ef_init_p[cnt_cell_index, i] = ef_p_i
            break
    return ef_init_n, ef_init_p

