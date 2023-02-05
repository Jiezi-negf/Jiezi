# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
from Jiezi.FEM.shape_function import shape_function
from Jiezi.Physics.common import *
import numpy as np


def assembly_nonlinear(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list,
                       u_k_cell, dof_amount, ef_n, ef_p, dos_GP_list, E_list, Ec, Eg, doping_GP_list):
    # create A_total and b_total to store the final matrix and vector in Ax=b
    A_total = matrix_numpy(dof_amount, dof_amount)
    b_total = vector_numpy(dof_amount)

    for cell_index in range(len(info_mesh)):
        # create a variable A_cell to store one part of A in Ax=b
        # create a variable b_cell to store one part of b in Ax=b
        A_cell = matrix_numpy(4, 4)
        b_cell = vector_numpy(4)

        # u_in_f[:] is the variable which will be regard as the input to the f_DFD and f
        # u_in_f[i] = N_GP_T * u_k, row dot column is a number, u_in_f stores four numbers
        u_in_f = [None] * 4
        u_cell_column = vector_numpy(4)
        for i in range(4):
            u_cell_column.copy(u_k_cell[cell_index].reshape(u_k_cell[cell_index].shape[0], 1))
            u_in_f[i] = op.vecdotvec(N_GP_T[i], u_cell_column)

        # add the left first term to A_cell
        A_cell = op.addmat(A_cell, cell_long_term[cell_index])

        # compute and add the left second term to A_cell
        for i in range(4):
            A_cell = op.addmat(A_cell, op.scamulmat(
                func_f_DFD(ef_n[cell_index, i], ef_p[cell_index, i], Ec, Eg,
                           dos_GP_list[cell_index][i], u_in_f[i], E_list),
                cell_NNTJ[cell_index][i]).nega())

        # compute and add the right first term to b_cell
        b_cell = op.addvec(b_cell, op.matmulvec(cell_long_term[cell_index], u_cell_column).nega())

        # compute and add the right second term to b_cell
        for i in range(4):
            b_cell = op.addvec(b_cell, op.scamulvec(
                func_f(ef_n[cell_index, i], ef_p[cell_index, i], Ec, Eg, doping_GP_list[cell_index][i],
                       dos_GP_list[cell_index][i], u_in_f[i], E_list),
                cell_NJ[cell_index][i]))

        # assembly the local matrix A_cell and local vector b_cell to the total matrix A and total vector b
        # cell_dof_index stores the index of dof in the specific cell, such as [23, 677, 89, 2]
        cell_dof_index = list(info_mesh[cell_index].keys())
        for i in range(4):
            for j in range(4):
                value_matA = A_cell.get_value(i, j) + A_total.get_value(cell_dof_index[i], cell_dof_index[j])
                A_total.set_value(cell_dof_index[i], cell_dof_index[j], value_matA)
            value_vecb = b_cell.get_value(i) + b_total.get_value(cell_dof_index[i])
            b_total.set_value((cell_dof_index[i], 0), value_vecb)

    # based on the Dirichlet list and Dirichlet boundary condition to adjust the final matrix and final vector
    for type_i in range(len(Dirichlet_list)):
        for i in range(len(Dirichlet_list[type_i])):
            D_index = Dirichlet_list[type_i][i]
            vec_zero = vector_numpy(dof_amount)
            # # set the column where the dirichlet point locates in to 0
            # A_total.set_block_value(0, dof_amount, D_index, D_index + 1, vec_zero)
            # set the row where the dirichlet point locates in to 0
            A_total.set_block_value(D_index, D_index + 1, 0, dof_amount, vec_zero.trans())
            # set the element about the Dirichlet point to 1
            A_total.set_value(D_index, D_index, 1.0)
            # set the value of b on Dirichlet point to its value based on the Dirichlet boundary condition
            b_total.set_value((D_index, 0), 0)
    # print("finish the assembly process")
    return A_total, b_total


def assembly_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                    dof_amount, doping_GP_list, n_GP_list, p_GP_list):
    # create A_total and b_total to store the final matrix and vector in Ax=b
    A_total = matrix_numpy(dof_amount, dof_amount)
    b_total = vector_numpy(dof_amount)

    for cell_index in range(len(info_mesh)):
        # create a variable A_cell to store one part of A in Ax=b
        # create a variable b_cell to store one part of b in Ax=b
        A_cell = matrix_numpy(4, 4)
        b_cell = vector_numpy(4)

        # add the left first term to A_cell
        A_cell = op.addmat(A_cell, cell_long_term[cell_index])

        # compute and add the right second term to b_cell
        for i in range(4):
            f = - n_GP_list[cell_index][i] + p_GP_list[cell_index][i] + doping_GP_list[cell_index][i]
            b_cell = op.addvec(b_cell, op.scamulvec(f, cell_NJ[cell_index][i]))

        # assembly the local matrix A_cell and local vector b_cell to the total matrix A and total vector b
        # cell_dof_index stores the index of dof in the specific cell, such as [23, 677, 89, 2]
        cell_dof_index = list(info_mesh[cell_index].keys())
        for i in range(4):
            for j in range(4):
                value_matA = A_cell.get_value(i, j) + A_total.get_value(cell_dof_index[i], cell_dof_index[j])
                A_total.set_value(cell_dof_index[i], cell_dof_index[j], value_matA)
            value_vecb = b_cell.get_value(i) + b_total.get_value(cell_dof_index[i])
            b_total.set_value((cell_dof_index[i], 0), value_vecb)

    # based on the Dirichlet list and Dirichlet boundary condition to adjust the final matrix and final vector
    for type_i in range(len(Dirichlet_list)):
        for i in range(len(Dirichlet_list[type_i])):
            D_index = Dirichlet_list[type_i][i]
            vec_zero = vector_numpy(dof_amount)
            # # set the column where the dirichlet point locates in to 0
            # A_total.set_block_value(0, dof_amount, D_index, D_index + 1, vec_zero)
            # set the row where the dirichlet point locates in to 0
            A_total.set_block_value(D_index, D_index + 1, 0, dof_amount, vec_zero.trans())
            # set the element about the Dirichlet point to 1
            A_total.set_value(D_index, D_index, 1.0)
            # set the value of b on Dirichlet point to its value based on the Dirichlet boundary condition
            b_total.set_value((D_index, 0), Dirichlet_value[type_i])
    # print("finish the assembly process")
    return A_total, b_total


def func_f_DFD(ef_n, ef_p, Ec, Eg, dos, u, E_list):
    f_DFD_n = 0.0
    f_DFD_p = 0.0
    E_step = E_list[1] - E_list[0]
    # compute f_DFD_n
    diff = Ec - E_list[0]
    end_index = len(E_list)
    if diff <= 0:
        start_index = 0
    elif Ec < E_list[len(E_list) - 1]:
        start_index = int(diff.real // E_step)
    else:
        start_index = end_index - 1
    for ee in range(start_index, end_index - 1):
        f_DFD_n += - 1 / KT * (dos[ee] * fermi(E_list[ee] - u - ef_n) * (1 - fermi(E_list[ee] - u - ef_n))
                               + dos[ee + 1] * fermi(E_list[ee + 1] - u - ef_n) * (
                                           1 - fermi(E_list[ee + 1] - u - ef_n))) \
                   * E_step / 2
    # compute f_DFD_p
    diff = Ec - Eg - E_list[0]
    start_index = 0
    if diff <= 0:
        end_index = 0
    elif Ec - Eg > E_list[len(E_list) - 1]:
        end_index = len(E_list) - 1
    else:
        end_index = int(diff.real // E_step)
    for ee in range(start_index, end_index):
        f_DFD_p += 1 / KT * (dos[ee] * fermi(E_list[ee] - u - ef_p) * (fermi(E_list[ee] - u - ef_p) - 1)
                             + dos[ee + 1] * fermi(E_list[ee + 1] - u - ef_p) * (fermi(E_list[ee + 1] - u - ef_p) - 1)
                             ) * E_step / 2
    f_DFD = f_DFD_n + f_DFD_p
    return f_DFD


def func_f(ef_n, ef_p, Ec, Eg, doping, dos, u, E_list):
    result = 0.0
    E_step = E_list[1] - E_list[0]
    # compute the first term which is about n
    diff = Ec - E_list[0]
    end_index = len(E_list)
    if diff <= 0:
        start_index = 0
    elif Ec < E_list[len(E_list) - 1]:
        start_index = int(diff.real // E_step)
    else:
        start_index = end_index - 1
    for ee in range(start_index, end_index - 1):
        result += - (dos[ee] * fermi(E_list[ee] - u - ef_n)
                     + dos[ee + 1] * fermi(E_list[ee + 1] - u - ef_n)) * E_step / 2
    # compute the second term which is about p
    diff = Ec - Eg - E_list[0]
    start_index = 0
    if diff <= 0:
        end_index = 0
    elif Ec - Eg > E_list[len(E_list) - 1]:
        end_index = len(E_list) - 1
    else:
        end_index = int(diff.real // E_step)
    for ee in range(start_index, end_index):
        result += (dos[ee] * (1 - fermi(E_list[ee] - u - ef_p))
                   + dos[ee + 1] * (1 - fermi(E_list[ee + 1] - u - ef_p))) * E_step / 2
    # compute the third term which is about doping concentration
    result += doping
    return result
