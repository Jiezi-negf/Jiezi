# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA import operator as op
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.FEM.shape_function import shape_function
import numpy as np
from Jiezi.Physics.common import *


def func_N(point_coord: list):
    """
    :param point_coord: one gauss point's coordinate (gauss_x, gauss_y, gauss_z)
    :return: value: list, four shape functions' value on this gauss point
    """
    alpha, beta, gamma = point_coord
    value = [1 - alpha - beta - gamma, alpha, beta, gamma]
    return value


def mark_zone(info_mesh, geo_para):
    mark_rule = {"air_inter": 0, "cnt": 1, "air_outer": 2, "gate_oxide": 3}
    mark_list = [None] * len(info_mesh)
    r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation = geo_para
    for cell_index in range(len(info_mesh)):
        cell_i = info_mesh[cell_index]
        dof_coord = list(cell_i.values())
        for dof_x, dof_y, dof_z in dof_coord:
            r = math.sqrt(dof_x ** 2 + dof_y ** 2)
            if r < r_inter:
                mark_list[cell_index] = mark_rule["air_inter"]
                break
            if r_outer > r > r_inter:
                mark_list[cell_index] = mark_rule["cnt"]
                break
            if r > r_outer and z_translation < dof_z < (z_total - z_translation):
                mark_list[cell_index] = mark_rule["gate_oxide"]
                break
            if r > r_outer and (dof_z < z_translation or dof_z > (z_total - z_translation)):
                mark_list[cell_index] = mark_rule["air_outer"]
                break
    return mark_list


def epsilon(mark_list, cell_index):
    mark_value = mark_list[cell_index]
    eps_list = [epsilon_air, epsilon_cnt, epsilon_air, epsilon_oxide]
    return eps_list[mark_value]


def N_gausspoint():
    """
    N_GP is a list, N_GP[i] = vec^T
    N_GP_T is a list, N_GP[i] = vec
    NNT is a list, NNT[i] is a matrix, the value of which = vec^T * vec
    vec = [N1(gauss_i_x,gauss_i_y,gauss_i_z), N2(gauss_i_x,...),...]
    """
    N_GP = [vector_numpy] * 4
    N_GP_T = [vector_numpy] * 4
    NNT = [matrix_numpy] * 4
    # define the first Gauss point coordinate series
    p = 0.58541
    q = 0.138197
    gauss_point_base = [p, q, q, q]
    # create the four gauss points
    for i in range(4):
        vec = vector_numpy(4)
        temp = gauss_point_base.copy()
        temp[i] = p
        gauss_point = temp[1:]
        vec.copy([func_N(gauss_point)])
        N_GP_T[i] = vec
        N_GP[i] = vec.trans()
        NNT[i] = op.vecmulvec(N_GP[i], N_GP_T[i])
    return N_GP, N_GP_T, NNT


def isDirichlet(point_coordinate, geo_para):
    r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation = geo_para
    flag = 0
    x, y, z = point_coordinate
    if abs(x**2 + y**2 - r_oxide**2) < 1e-6 and z_translation - 1e-6 < z < (z_total - z_translation)+1e-6:
        flag = 1
    return flag


def constant_parameters(info_mesh, geo_para):
    """
    :param info_mesh: [{vertex1:[x,y,z],vertex2:[x,y,z],... },{},...]
    :param geo_para: geometrical parameters
    :return: N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, Dirichlet_list
        1 N_GP, N_GP_T, NNT
        2 coefficients of shape functions: [[[a1,a2,a3,a4],[b1,b2,b3,b4],[c..],[d...]],[],...]
        3 determinant of Jacobian matrix |J_e|: [|J_e of cell1|, |J_e of cell2|, ... ]
        4 long term (the frst term on the left hand)
        5 compute N*|J|*weight_quadrature*volume and N*N^T*|J|*weight_quadrature*volume
        6 find all points on the Dirichlet boundary
    """
    volume = 1 / 6
    weight_quadrature = 1 / 4
    # mark_list: the output of "def mark_zone", which can mark the zone by int number
    mark_list = mark_zone(info_mesh, geo_para)

    # 1 call the N_gausspoint to compute N, N^T, NN^T
    N_GP, N_GP_T, NNT = N_gausspoint()

    N_alpha = vector_numpy(4)
    N_alpha.copy([[-1, 1, 0, 0]])
    N_beta = vector_numpy(4)
    N_beta.copy([[-1, 0, 1, 0]])
    N_gamma = vector_numpy(4)
    N_gamma.copy([[-1, 0, 0, 1]])

    cell_co = [None] * len(info_mesh)
    cell_long_term = [None] * len(info_mesh)
    cell_NJ = [None] * len(info_mesh)
    cell_NNTJ = [None] * len(info_mesh)
    Dirichlet_list = []

    for cell_index in range(len(info_mesh)):
        # cell_i is a dict, {vertex1:[x,y,z],vertex2:[x,y,z],... }
        cell_i = info_mesh[cell_index]
        dof_number = list(cell_i.keys())
        dof_coord = list(cell_i.values())
        eps_cell_index = epsilon(mark_list, cell_index)

        # 2 compute the coefficients of the shape functions
        co_shapefunc = shape_function(dof_coord)
        # co_shapefunc_bcd234 = np.array(co_shapefunc)[1:4, 1:4]
        cell_co[cell_index] = co_shapefunc

        # 3 compute the determinant of Jacobian matrix |J_e|
        # as there will be zero element in co_shapefunc, i need to compute the reciprocal element by element
        # jacobian = np.zeros((3, 3))
        # for i in range(3):
        #     for j in range(3):
        #         if not co_shapefunc_bcd234[i][j] == 0:
        #             jacobian[i, j] = 1 / co_shapefunc_bcd234[i, j]
        # det_jacobian = np.linalg.det(jacobian)
        jacob_temp = np.array(dof_coord).transpose()
        jacob2 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                jacob2[i, j] = jacob_temp[i, j + 1] - jacob_temp[i, 0]
        det_jacobian = np.linalg.det(jacob2)

        # 4 compute the long term
        alpha_xyz = vector_numpy(3)
        beta_xyz = vector_numpy(3)
        gamma_xyz = vector_numpy(3)
        alpha_xyz.copy([[ele[1] for ele in co_shapefunc[1:]]])
        beta_xyz.copy([[ele[2] for ele in co_shapefunc[1:]]])
        gamma_xyz.copy([[ele[3] for ele in co_shapefunc[1:]]])
        g_alpha = op.vecdotvec(alpha_xyz, alpha_xyz.trans())
        g_beta = op.vecdotvec(beta_xyz, beta_xyz.trans())
        g_gamma = op.vecdotvec(gamma_xyz, gamma_xyz.trans())
        g_alphabeta = op.vecdotvec(alpha_xyz, beta_xyz.trans())
        g_alphagamma = op.vecdotvec(alpha_xyz, gamma_xyz.trans())
        g_betagamma = op.vecdotvec(beta_xyz, gamma_xyz.trans())
        long_term = op.scamulmat(eps_cell_index * det_jacobian * volume,
                                 op.addmat(op.scamulmat(g_alpha,
                                                        op.vecmulvec(N_alpha.trans(),
                                                                     N_alpha)),
                                           op.scamulmat(g_beta,
                                                        op.vecmulvec(N_beta.trans(),
                                                                     N_beta)),
                                           op.scamulmat(g_gamma,
                                                        op.vecmulvec(N_gamma.trans(),
                                                                     N_gamma)),
                                           op.scamulmat(g_alphabeta,
                                                        op.addmat(
                                                            op.vecmulvec(N_alpha.trans(),
                                                                         N_beta),
                                                            op.vecmulvec(N_beta.trans(),
                                                                         N_alpha))),
                                           op.scamulmat(g_alphagamma,
                                                        op.addmat(
                                                            op.vecmulvec(N_alpha.trans(),
                                                                         N_gamma),
                                                            op.vecmulvec(N_gamma.trans(),
                                                                         N_alpha))),
                                           op.scamulmat(g_betagamma,
                                                        op.addmat(
                                                            op.vecmulvec(N_beta.trans(),
                                                                         N_gamma),
                                                            op.vecmulvec(N_gamma.trans(),
                                                                         N_beta)))
                                           ))
        cell_long_term[cell_index] = long_term

        # 5 compute N*|J|*weight_quadrature*volume and N*N^T*|J|*weight_quadrature*volume
        NJ_singlecell = [None] * 4
        NNTJ_singlecell = [None] * 4
        for i in range(4):
            NJ_singlecell[i] = op.scamulvec(weight_quadrature * det_jacobian * volume, N_GP[i])
            NNTJ_singlecell[i] = op.scamulmat(weight_quadrature * det_jacobian * volume, NNT[i])
        cell_NJ[cell_index] = NJ_singlecell
        cell_NNTJ[cell_index] = NNTJ_singlecell

        # 6 find the Dirichlet points
        for i in range(4):
            point_coord = dof_coord[i]
            if isDirichlet(point_coord, geo_para) == 1:
                Dirichlet_list.append(dof_number[i])
    Dirichlet_list = list(set(Dirichlet_list))
    return N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, Dirichlet_list

