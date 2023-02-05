# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import math
import numpy as np


def map_tocell(info_mesh, u_vec):
    """
    map the u_vec which is from dof_0 to dof_N to u_cell whose oder is depends on cell
    :param info_mesh: [{14:[x,y,z], 25:[x,y,z], 3:[x,y,z], 6:[x,y,z]},{},{},...]
    :param u_vec: numpy.array, shape is (dof_amount, 1), [[2.1], [3], [1.2],...]
    :return: u_cell: numpy.array, shape is (cell_amount, 4), [[5.2, 6, 4.2, 2.3], [], ...]
    """
    cell_amount = len(info_mesh)
    u_cell = np.empty([cell_amount, 4])
    for cell_index in range(cell_amount):
        cell_i = info_mesh[cell_index]
        dof_number = list(cell_i.keys())
        for i in range(4):
            u_cell[cell_index, i] = u_vec[dof_number[i], 0]
    return u_cell


# this method split the projection to two parts, one is to find which cell the point belongs to,
# the other is to compute the function value

def projection(dict_cell, u_cell, coord, cell_co, num_radius, num_z, r_oxide, z_total):
    coord_ref, cell_index = link_to_cell(dict_cell, coord, cell_co, num_radius, num_z, r_oxide, z_total)
    value = get_func_value(coord_ref, u_cell, cell_index)
    return value


def link_to_cell(dict_cell, coord, cell_co, num_radius, num_z, r_oxide, z_total):
    x, y, z = coord
    if z < 1e-6:
        z = 0.0
    r = math.sqrt(x ** 2 + y ** 2)
    r_index = int(r // (r_oxide / num_radius))
    z_index = int(z // (z_total / num_z))
    tol = 1e-6
    coord_ref = [0, 0, 0]
    index = 0
    for cell_index in dict_cell[(r_index, z_index)]:
        a, b, c, d = cell_co[cell_index][:]
        alpha = a[1] + b[1] * x + c[1] * y + d[1] * z
        beta = a[2] + b[2] * x + c[2] * y + d[2] * z
        gamma = a[3] + b[3] * x + c[3] * y + d[3] * z
        if 0 - tol <= alpha <= 1 + tol and \
                0 - tol <= beta <= 1 + tol and \
                0 - tol <= gamma <= 1 + tol and alpha + beta + gamma <= 1 + tol:
            coord_ref = [alpha, beta, gamma]
            index = cell_index
    return coord_ref, index


def get_func_value(coord_ref, u_cell, cell_index):
    alpha, beta, gamma = coord_ref
    N = np.array([1 - alpha - beta - gamma, alpha, beta, gamma])
    u = u_cell[cell_index]
    value = np.dot(N, u)
    return value

# #  this method loop all the cells directly with the cut process
# def projection(dict_cell, u_cell, coord, cell_co, num_radius, num_z, r_oxide, z_total):
#     x, y, z = coord
#     r = math.sqrt(x ** 2 + y ** 2)
#     r_index = int(r // (r_oxide / num_radius))
#     z_index = int(z // (z_total / num_z))
#     res = 0.0
#     for cell_index in dict_cell[(r_index, z_index)]:
#         a, b, c, d = cell_co[cell_index][:]
#         alpha = a[1] + b[1] * x + c[1] * y + d[1] * z
#         beta = a[2] + b[2] * x + c[2] * y + d[2] * z
#         gamma = a[3] + b[3] * x + c[3] * y + d[3] * z
#         if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and alpha + beta + gamma <= 1:
#             N = np.array([1 - alpha - beta - gamma, alpha, beta, gamma])
#             u = np.array(u_cell[cell_index])
#             res = np.dot(N, u)
#     if res == 0:
#         print(coord)
#     return res


# # this method loop all the cells directly without the cut process
# def projection(dict_cell, u_cell, coord, cell_co, num_radius, num_z, r_oxide, z_total):
#     x, y, z = coord
#     res = 0
#     tol = 1e-5
#     for cell_index in range(len(cell_co)):
#         a, b, c, d = cell_co[cell_index][:]
#         alpha = a[1] + b[1] * x + c[1] * y + d[1] * z
#         beta = a[2] + b[2] * x + c[2] * y + d[2] * z
#         gamma = a[3] + b[3] * x + c[3] * y + d[3] * z
#         if 0 -tol <= alpha <= 1 + tol and \
#                 0 -tol <= beta <= 1+tol and \
#                 0-tol <= gamma <= 1+tol and alpha + beta + gamma <= 1+tol:
#             N = np.array([1 - alpha - beta - gamma, alpha, beta, gamma])
#             u = np.array(u_cell[cell_index])
#             res = np.dot(N, u)
#     if res == 0:
#         print(coord)
#     return res


def cut(r_oxide, z_total, info_mesh, num_radius=3, num_z=3):
    # initialize the dict which stores the cell index belongs to different zones
    dict_cell = {}
    for i in range(num_radius + 1):
        for j in range(num_z + 1):
            dict_cell[(i, j)] = []
    # loop all the cells to classify them to different zones based on the location of the vertex of cell
    for cell_index in range(len(info_mesh)):
        cell_i = info_mesh[cell_index]
        dof_coord = list(cell_i.values())
        for dof_x, dof_y, dof_z in dof_coord:
            r = math.sqrt(dof_x ** 2 + dof_y ** 2)
            r_index = int(r // (r_oxide / num_radius))
            z_index = int(dof_z // (z_total / num_z))
            # if this cell_index has not been added to dict[i, j], it should be added
            if cell_index not in dict_cell[r_index, z_index]:
                dict_cell[r_index, z_index].append(cell_index)
    return dict_cell
