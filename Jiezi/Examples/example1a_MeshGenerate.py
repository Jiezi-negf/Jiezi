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
from Jiezi.Graph import builder
from Jiezi.Physics.common import *
from Jiezi.Physics.PrePoisson import PrePoisson

# construct the cnt object
cnt = builder.CNT(n=8, m=0, Trepeat=60, nonideal=False)
cnt.construct()
num_atom_cell = cnt.get_nn()
num_cell = cnt.get_Trepeat()
num_atom_total = num_atom_cell * num_cell
radius_tube = cnt.get_radius()
length_single_cell = cnt.get_singlecell_length()
z_total = cnt.get_length()
num_supercell = 1
volume_cell = math.pi * radius_tube ** 2 * length_single_cell
print("Amount of atoms in single cell:", num_atom_cell)
print("Total amount of atoms:", num_atom_total)
print("Radius of tube:", radius_tube)
print("Length of single cell:", length_single_cell)
print("Length of whole device:", z_total)
# define the geometry parameters
width_cnt_scale = 1
width_oxide_scale = 3.18
z_length_oxide_scale = 0.167
width_cnt = width_cnt_scale * radius_tube
zlength_oxide = z_length_oxide_scale * z_total
width_oxide = width_oxide_scale * radius_tube
print("Length of gate:", zlength_oxide)
print("Thickness of cnt:", width_cnt)
print("Thickness of oxide:", width_oxide)
r_inter = radius_tube - width_cnt / 2
r_outer = r_inter + width_cnt
r_oxide = r_outer + width_oxide
z_translation = 0.5 * (z_total - zlength_oxide)
z_isolation = 10
# use salome to build the FEM grid -- use PC equipped with software salome
geo_para, path_xml = PrePoisson(cnt, width_cnt_scale, width_oxide_scale, z_length_oxide_scale, z_isolation)
