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
    """
    classify each cell to its own area and give each one a number(0/1/2/3)
    :param info_mesh: information of the mesh consists of the composition of the points of each cell and coordinate
            of each point and the order of each cell
    :param geo_para: parameters used to define the geometry
    :return: mark_list: the area number of each cell
             cnt_cell_list: the cell index of all the cell which is classified into CNT area
    """
    mark_rule = {"air_inter": 0, "cnt": 1, "air_outer": 2, "gate_oxide": 3}
    mark_list = [None] * len(info_mesh)
    cnt_cell_list = []
    r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation, z_isolation = geo_para
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
                cnt_cell_list.append(cell_index)
                break
            if r > r_outer and z_translation < dof_z < (z_total - z_translation):
                mark_list[cell_index] = mark_rule["gate_oxide"]
                break
            if r > r_outer and (dof_z < z_translation or dof_z > (z_total - z_translation)):
                mark_list[cell_index] = mark_rule["air_outer"]
                break
    return mark_list, cnt_cell_list


def epsilon(mark_list, cell_index):
    mark_value = mark_list[cell_index]
    eps_list = [epsilon_air, epsilon_cnt, epsilon_air_outer, epsilon_oxide]
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
    """
    judge if the given point is on the Dirichlet boundary, if it is, determine whcih part of Dirichlet boundary it's
    belong to
    :param point_coordinate: coordinate of the given point, (x, y, z)
    :param geo_para: parameters used to define the geometry
    :return: flag==1 indicates that the given point is belong to the electrode of electrostatic doing near the source
    flag==2 indicates that the given point is belong to the electrode of gate
    flag==3 indicates that the given point is belong to the electrode of electrostatic doing near the drain
    """
    r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation, z_isolation = geo_para
    flag = 0
    tol = 1e-6
    x, y, z = point_coordinate
    if abs(math.sqrt(x ** 2 + y ** 2) - r_oxide) < tol:
        if z < z_translation - z_isolation + tol:
            flag = 1
        if z_translation - tol < z < z_total - z_translation + tol:
            flag = 2
        if z > z_total - z_translation + z_isolation - tol:
            flag = 3
    # if abs(math.sqrt(x**2 + y**2) - r_oxide) < tol and z < z_translation + tol:
    #     flag = 0
    # if abs(math.sqrt(x ** 2 + y ** 2) - r_oxide) < tol and z_translation - tol < z < (
    #         z_total - z_translation) + tol:
    #     flag = 1
    # if abs(math.sqrt(x**2 + y**2) - r_oxide) < tol and z > z_total - z_translation + tol:
    #     flag = 2
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
        6 Classify each cell into its own area and assign an area number
        7 get the cell index of all the cell which is classified into CNT area
        8 find all points on the Dirichlet boundary
    """
    volume = 1 / 6
    weight_quadrature = 1 / 4
    # mark_list: the output of "def mark_zone", which can mark the zone by int number
    mark_list, cnt_cell_list = mark_zone(info_mesh, geo_para)

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
    Dirichlet_source_list = []
    Dirichlet_gate_list = []
    Dirichlet_drain_list = []

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
                Dirichlet_source_list.append(dof_number[i])
                continue
            if isDirichlet(point_coord, geo_para) == 2:
                Dirichlet_gate_list.append(dof_number[i])
                continue
            if isDirichlet(point_coord, geo_para) == 3:
                Dirichlet_drain_list.append(dof_number[i])
                continue
    Dirichlet_source_list = list(set(Dirichlet_source_list))
    Dirichlet_gate_list = list(set(Dirichlet_gate_list))
    Dirichlet_drain_list = list(set(Dirichlet_drain_list))
    Dirichlet_list.append(Dirichlet_source_list)
    Dirichlet_list.append(Dirichlet_gate_list)
    Dirichlet_list.append(Dirichlet_drain_list)
    return N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, cnt_cell_list, Dirichlet_list


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
    based on the GPs' coordinate on xyz space, get the layer which it belongs to
    then get the dos which is just a function of energy.
    :param coord: coordinate of one gauss point
    :param dos: from the output of GF, dos[ee][zz] = D(E(ee),z)
    :param z_length: the whole length of the tube along the z axis
    :return: dos_GP[ee1, ee2, ...], the index of which is energy
    """
    x, y, z = coord
    length_layer = z_length / len(dos[0])
    layer_index = int(z.real // length_layer)
    dos_GP = [row[layer_index] for row in dos]
    return dos_GP


def find_np(coord, density_np, z_length):
    """
    based on the GPs' coordinate on xyz space, get the layer which it belongs to.
    then get the density of electrons or holes.
    :param coord: coordinate of one gauss point
    :param density_np: from the output of GF, n[zz] = n(z), p[zz] = p(z)
    :param z_length: the whole length of the tube along the z axis
    :return: np_GP, a float type data
    """
    x, y, z = coord
    length_layer = z_length / len(density_np)
    layer_index = int(z.real // length_layer)
    np_GP = density_np[layer_index].real
    return np_GP


def get_coord_GP_list(cell_co):
    """
    get coordinate of every Gauss Point of every cell
    :param cell_co: coefficients of every cell
    :return: coord_GP_list is a 3-level list, coord_GP_list[cell_index] = [GP1[x, y, z], GP2[x, y, z]...]
    """
    coord_GP_list = [None] * len(cell_co)
    for cell_index in range(len(cell_co)):
        coord_GP_list[cell_index] = gauss_xyz(cell_co[cell_index])
    return coord_GP_list


def get_dos_GP_list(coord_GP_list, dos, z_length, mark_list):
    """
    get dos of every Gauss Point of every cell
    :param coord_GP_list: coord_GP_list[cell_index] = [GP1[x, y, z], GP2[x, y, z]...]
    :param dos: from the output of GF, dos[ee][zz] = D(E(ee),z)
    :param z_length: the whole length of the tube along the z axis
    :return: dos_GP_list, a 3-level list, dos_GP_list[cell_index] = [dos[:][GP1], dos[:][GP2], ...]
    """
    dos_GP_list = [None] * len(coord_GP_list)
    num_energy = len(dos)
    for cell_index in range(len(coord_GP_list)):
        if mark_list[cell_index] != 1:
            # if this cell is not in CNT region, then set dos to zero
            dos_GP_list_i = [[0.0] * num_energy] * 4
        else:
            # if this cell is in CNT region, then set dos value based on NEGF output
            dos_GP_list_i = [None] * 4
            for GP_index in range(4):
                dos_GP_list_i[GP_index] = find_dos(coord_GP_list[cell_index][GP_index], dos, z_length)
        dos_GP_list[cell_index] = dos_GP_list_i
    return dos_GP_list


def get_np_GP_list(coord_GP_list, density_np, z_length, mark_list):
    """
    get n or p of every Gauss Point of every cell
    :param coord_GP_list: coord_GP_list[cell_index] = [GP1[x, y, z], GP2[x, y, z]...]
    :param density_np: from the output of GF, n[zz] = n(z), p[zz] = p(z)
    :param z_length: the whole length of the tube along the z axis
    :return: np_GP_list, a 3-level list, np_GP_list[cell_index] = [np[GP1], np[GP2], ...]
    """
    np_GP_list = [None] * len(coord_GP_list)
    for cell_index in range(len(coord_GP_list)):
        if mark_list[cell_index] != 1:
            np_GP_list_i = [0.0] * 4
        else:
            np_GP_list_i = [None] * 4
            for GP_index in range(4):
                np_GP_list_i[GP_index] = find_np(coord_GP_list[cell_index][GP_index], density_np, z_length)
        np_GP_list[cell_index] = np_GP_list_i
    return np_GP_list


def doping(coord_GP_list, zlength_oxide, z_translation, doping_source, doping_drain, doping_channel, mark_list):
    """
    set doping concentration of every Gauss Point of every cell
    :param coord_GP_list: coord_GP_list[cell_index] = [GP1[x, y, z], GP2[x, y, z]...]
    :param zlength_oxide: the whole length of the tube along the z axis
    :param z_translation: the start point of gate oxide in z direction
    :param doping_source: N_D - N_A, i.e. N_{D-A} in source region
    :param doping_drain: N_D - N_A, i.e. N_{D-A} in drain region
    :param doping_channel: N_D - N_A, i.e. N_{D-A} in channel region
    :return: doping_GP_list, a 3-level list, doping_GP_list[cell_index] = [doping[GP1], doping[GP2], ...]
    """
    doping_GP_list = [None] * len(coord_GP_list)
    for cell_index in range(len(coord_GP_list)):
        if mark_list[cell_index] != 1:
            doping_GP_list_i = [0.0] * 4
        else:
            doping_GP_list_i = [float] * 4
            for GP_index in range(4):
                x, y, z = coord_GP_list[cell_index][GP_index]
                if z.real < z_translation:
                    doping_GP_list_i[GP_index] = doping_source
                elif z.real < z_translation + zlength_oxide:
                    doping_GP_list_i[GP_index] = doping_channel
                else:
                    doping_GP_list_i[GP_index] = doping_drain
        doping_GP_list[cell_index] = doping_GP_list_i
    return doping_GP_list


def fixedChargeInit(coord_GP_list):
    """
    initialize the list storing the fixed charge on every gauss point of each cell
    :param coord_GP_list: list storing the coordinate of every gauss point of each cell
    :return: fixedCharge_GP_list
    """
    size = len(coord_GP_list)
    fixedCharge_GP_list = [None] * size
    for cell_index in range(size):
        fixedCharge_GP_list_i = [0.0] * 4
        fixedCharge_GP_list[cell_index] = fixedCharge_GP_list_i
    return fixedCharge_GP_list


def addFixedCharge(fixedCharge_GP_list, coord_GP_list, mark_list, scope, density):
    """
    add fixed charge value to the initialized list
    after the initialization, this function can be called more than one time because in some case
    I want to define different density to different area
    :param fixedCharge_GP_list: the initialized list
    :param coord_GP_list: list storing the coordinate of every gauss point of each cell
    :param mark_list: the area number of each cell
    :param scope: define where should be added fixed charge (include the radius scope and the scope in z axis)
    :param density: define the density of fixed charge
    :return: no return value, after this function executed, the fixed charge has been added to specific gauss points
    """
    size = len(fixedCharge_GP_list)
    radius_min, radius_max, z_min, z_max = scope
    for cell_index in range(size):
        if mark_list[cell_index] != 0 and mark_list[cell_index] != 1:
            continue
        else:
            for GP_index in range(4):
                x, y, z = coord_GP_list[cell_index][GP_index]
                radius2 = x.real ** 2 + y.real ** 2
                flag = radius_min ** 2 <= radius2 <= radius_max ** 2 and z_min <= z.real <= z_max
                if flag:
                    fixedCharge_GP_list[cell_index][GP_index] += density


# a = [(0.0034572369685743724+0j), (0.002680160837190706+0j), (0.0019124049899249106+0j), (0.0012428728427696174+0j),
#       (0.0007689273642102201+0j), (0.0006848924676253319+0j), (0.0010295048911840828+0j), (0.0018112301366284985+0j),
#       (0.0028924140519903394+0j), (0.0041230524065758375+0j)]
# b = [0.9956599607932615, 0.9950547696662598, 0.9951972926952809, 0.9867066164938678, 1.0108604301134547,
#      1.2451215515189975, 1.8218731445466385, 1.9943501156287846, 1.9939555592694795, 1.9939538853813874]
# plt.subplot(1, 2, 1)
# plt.title("phi")
# plt.plot(np.arange(len(b)), b, color="green")
# plt.subplot(1, 2, 2)
# plt.title("electron")
# plt.plot(np.arange(len(a)), a, color="red")
# plt.show()
