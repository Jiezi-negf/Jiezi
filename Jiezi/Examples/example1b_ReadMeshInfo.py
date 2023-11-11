# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.FEM.mesh import create_dof


# set the path for writing or reading files
# path_Files is the path of the shared files among process
path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))


# read mesh information from .dat file -- use PC without salome
path_dat = path_Files + "/Mesh_whole.dat"
#  solve the constant terms and parameters of poisson equation
info_mesh, dof_amount, dof_coord_list = create_dof(path_dat)
print("amount of element:", len(info_mesh))
print("amount of dof:", dof_amount)
print("first cell:", info_mesh[0])