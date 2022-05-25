# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys

sys.path.append("../../../")

from Jiezi.FEM import mesh
from Jiezi.Physics.PrePoisson import PrePoisson
from Jiezi.Graph import builder
from Jiezi.FEM.dirty import dirty
from Jiezi.FEM.assembly import assembly
from Jiezi.FEM.map import map_tocell
import numpy as np
import random

cnt = builder.CNT(n=5, m=5, Trepeat=6, nonideal=False)
cnt.construct()
path = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/" + 'Mesh_whole.dat'
info_mesh, dof_amount = mesh.create_dof(path)
geo_para, path_xml = PrePoisson(cnt)
print("finish Prepoisson")
Dirichlet_BC = 10.2
N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, Dirichlet_list = dirty(info_mesh, geo_para)
u_k = np.random.rand(dof_amount).tolist()
u_k_cell = map_tocell(info_mesh, u_k)
print("start assembly")
A, b = assembly(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list, Dirichlet_BC,
             u_k_cell, dof_amount)
print(b.get_value())