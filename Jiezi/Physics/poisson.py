# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
import math
from dolfin import *
import matplotlib.pyplot as plt
import copy
from Jiezi.FEM.constant_parameters import constant_parameters
from Jiezi.FEM.map import map_tocell
from Jiezi.FEM.ef_solver import ef_solver
from Jiezi.FEM.assembly import assembly


def poisson(info_mesh, N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, Dirichlet_list,
            z_length, E_list, TOL_ef, TOL_du, ef_init,
            dos, density_n, u_init):

    # call a mapping function to map the u_init which is defined following the dof order to
    # u_init_cell which is defined following the cell structure
    u_init_cell = map_tocell(info_mesh, u_init)

    # solve the fermi energy level on gauss points of every cell
    ef, dos_GP_list = \
        ef_solver(u_init_cell, N_GP_T, cell_co, dos, density_n, ef_init, mark_list, z_length, E_list, TOL_ef)

    # start the newton iteration now
    u_k = copy.deepcopy(u_init)
    dof_amount = len(u_k)
    while 1:
        u_k_cell = map_tocell(info_mesh, u_k)
        A, b = assembly(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list,
             u_k_cell, dof_amount, ef, dos_GP_list, E_list)
        du = np.linalg.solve(A.get_value(), b.get_value())
        if norm(du) < TOL_du:
            break
        u_k = u_k + du
    return u_k