# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys
import os

sys.path.append("../../../")
from Jiezi.Graph.builder import CNT
from Jiezi.Physics.common import *


def PrePoisson(cnt: CNT):
    # step1: get the geometry parameters from the graph module
    cnt_radius = cnt.get_radius()
    width_cnt = 0.3 * cnt_radius
    r_inter = cnt_radius - width_cnt / 2
    width_oxide = 0.2 * cnt_radius
    z_total = cnt.get_length()
    zlength_oxide = 0.2 * z_total

    mesh_whole_min_size = 0.08 * z_total
    mesh_whole_max_size = 0.15 * z_total
    mesh_cnt_min_size = 0.7 * mesh_whole_min_size
    mesh_cnt_max_size = 0.7 * mesh_whole_max_size

    # step2: determine where the salome script file is and where the Mesh_whole.dat file is
    path_salome_bin = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC"
    path_salome_script = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/cnt_whole_test.py"
    path_salome_dat = path_salome_bin + "/myPro/" + "Mesh_whole.dat"
    path_salome_kill = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/BINARIES-UB18.04/KERNEL/bin/salome/killSalome.py"
    path_salome2fenics = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/salome2fenics.py"

    # step3: change the geometry parameters and the mesh parameters of the salome script
    file_data = ""
    para_geo_name_list = ["r_inter", "width_cnt", "width_oxide", "z_total", "zlength_oxide"]
    para_geo_data_list = [r_inter, width_cnt, width_oxide, z_total, zlength_oxide]
    para_mesh_name_list = ["NETGEN_3D_Parameters_1.SetMaxSize", "NETGEN_3D_Parameters_1.SetMinSize",
                           "NETGEN_3D_Parameters_2.SetMaxSize", "NETGEN_3D_Parameters_2.SetMinSize"]
    para_mesh_data_list = [mesh_whole_max_size, mesh_whole_min_size,
                           mesh_cnt_max_size, mesh_cnt_min_size]
    with open(path_salome_script, "r", encoding="utf-8") as f:
        for line in f:
            for i in range(len(para_geo_name_list)):
                if "notebook.set(\"" + para_geo_name_list[i] + "\"" in line:
                    line = "notebook.set(\"" + para_geo_name_list[i] + "\", " + str(para_geo_data_list[i]) + ")" + "\n"
                    break
            for i in range(len(para_mesh_name_list)):
                if para_mesh_name_list[i] in line:
                    line = para_mesh_name_list[i] + "(" + str(para_mesh_data_list[i]) + ")" + "\n"
                    break
            file_data += line
    with open(path_salome_script, "w", encoding="utf-8") as f:
        f.write(file_data)

    # step 4: run the salome script without GUI and then kill the process
    os.system(path_salome_bin + "/salome -t" + " " + path_salome_script)
    os.system("python3" + " " + path_salome_kill)

    # step 5: run the "salome2fenics.py" to transform .dat to .xml
    os.system("python3" + " " + path_salome2fenics)

    # step 6: compute all the geometric parameters  that poisson solver needs as the return value
    r_outer = r_inter + width_cnt
    r_oxide = r_outer + width_oxide
    z_translation = 0.5 * (z_total - zlength_oxide)
    geo_para = [r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation]
    path_xml = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/"
    return geo_para, path_xml