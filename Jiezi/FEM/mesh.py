# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


def create_dof(path):
    with open(path) as file:
        lines = [line.rstrip('\n') for line in file]  # 将所有行按顺序合成一行，形成一个字符串数组，每一个元素就是以前的每一行

    first_line = lines[0].split(' ')  # 首行遇见空格就分割，将字符串数组分隔开
    nb_dof = int(first_line[0])  # 首行第一个元素是节点总数，第二个元素是element的总数，包含了1D边edge，2D面facet，3D体volume
    nb_cell = int(first_line[1])

    # filling all dofs
    dof_coord = [list] * nb_dof
    for i in range(0, nb_dof):  # for in range语句的范围是个左闭右开区间
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
    return cell_vol, nb_dof


