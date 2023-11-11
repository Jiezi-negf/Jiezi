# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


def create_dof(path):
    """
    according to the output .dat file from the salome, create the data structure which I can use in python code
    to store information of mesh
    :param path: path of the .dat file which stores the mesh information
    :return: cell_vol is a dict which looks like
                \{node_{i,1}:(x,y,z),node_{i,2}:(x,y,z),node_{i,3}:(x,y,z),node_{i,4}:(x,y,z)\}
             nb_dof is the amount of the dof(degree of freedom, which is the grid point including the inter point)
             dof_coord is a list which looks like [[x, y, z], [x, y, z]...]
    """
    with open(path) as file:
        # link every row to one row by order to form a string type list, each element of which is each origin row
        lines = [line.rstrip('\n') for line in file]
    # split the first line which is a string type list by blank space
    first_line = lines[0].split(' ')
    # the first number in the first line is the amount of dofs
    nb_dof = int(first_line[0])
    # the second number in the first is the amount of elements, consist of 1D edge, 2D facet, 3D volume
    nb_cell = int(first_line[1])

    # filling all dofs
    dof_coord = [list] * nb_dof
    # the scope of this sentence "for in range" is [,)
    for i in range(0, nb_dof):
        tmp_line = lines[i + 1].split(' ')
        tmp_coord = [float] * 3
        for j in range(0, 3):
            tmp_coord[j] = round(float(tmp_line[j+1]), 8)
        dof_coord[i] = tmp_coord.copy()

    # filling all 304 cells
    # count the amount of 102 cell and 203 cell
    count_notvol = 0
    for i in range(0, nb_cell):
        tmp_line = lines[i + 1 + nb_dof].split(" ")
        if len(tmp_line) < 7:
            count_notvol = count_notvol + 1
        else:
            break
    cell_vol = [dict] * (nb_cell - count_notvol)
    for i in range(0, len(cell_vol)):
        tmp_line = lines[i + 1 + nb_dof + count_notvol].split(" ")
        tmp_dict = {}
        for j in range(2, 6):
            tmp_dict[int(tmp_line[j])-1] = dof_coord[int(tmp_line[j])-1]
        cell_vol[i] = tmp_dict
    return cell_vol, nb_dof, dof_coord


