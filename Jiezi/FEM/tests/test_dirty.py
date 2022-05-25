# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys
sys.path.append("../../../")
from Jiezi.FEM.dirty import dirty
from Jiezi.FEM.mesh import create_dof
from Jiezi.Physics.PrePoisson import PrePoisson
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder


cnt = builder.CNT(n=5, m=5, Trepeat=6, nonideal=False)
cnt.construct()

path = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/" + 'Mesh_whole.dat'

info_mesh, dof_amount = create_dof(path)
geo_para, path_xml = PrePoisson(cnt)
print("finish Prepoisson")
N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, Dirichlet_list = dirty(info_mesh, geo_para)
print("debug")


