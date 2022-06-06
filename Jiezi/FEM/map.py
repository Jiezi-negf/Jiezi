# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


def map_tocell(info_mesh, u_vec):
    """
    map the u_vec which is from dof_0 to dof_N to u_cell whose oder is depends on cell
    :param info_mesh: [{14:[x,y,z], 25:[x,y,z], 3:[x,y,z], 6:[x,y,z]},{},{},...]
    :param u_vec: [2.1, 3, 1.2,...]
    :return: u_cell: [[5.2, 6, 4.2, 2.3], [], ...]
    """
    cell_amount = len(info_mesh)
    u_cell = [None] * cell_amount
    for cell_index in range(cell_amount):
        u_cell_i = [None] * 4
        cell_i = info_mesh[cell_index]
        dof_number = list(cell_i.keys())
        for i in range(4):
            u_cell_i[i] = u_vec[dof_number[i]]
        u_cell[cell_index] = u_cell_i
    return u_cell



