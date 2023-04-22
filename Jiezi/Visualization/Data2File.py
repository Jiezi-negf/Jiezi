# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from itertools import combinations


def phi2VTK(func_value_vec, dof_coord_list, info_mesh, path_Files):
    # create root element
    root = ET.Element("VTKFile")
    root.set("type", "UnstructuredGrid")
    root.set("version", "0.1")
    # create grid_type element (sub-element of VTKFile)
    grid_type_elem = ET.Element("UnstructuredGrid")
    root.append(grid_type_elem)

    # data structure or hierarchy of real data part
    # 1st: Piece -- UnstructuredGrid->
    #  2nd: Points -- Piece->
    #   3rd: DataArray -- Points->
    #  2nd: Cells -- Piece->
    #   3rd: DataArray(connectivity) -- Cells->
    #   3rd: DataArray(offsets) -- Cells->
    #   3rd: DataArray(types) -- Cells->
    #  2nd: PointData -- Piece->
    #   3rd: DataArray(function name) -- PointData->

    # create Piece element
    piece_elem = ET.Element("Piece")
    num_points = len(dof_coord_list)
    num_cells = len(info_mesh)
    piece_elem.set("NumberOfPoints", str(num_points))
    piece_elem.set("NumberOfCells", str(num_cells))
    # add Piece to UnstructuredGrid
    grid_type_elem.append(piece_elem)

    # create Points -- Piece->
    points_elem = ET.Element("Points")
    # create DataArray -- Points->
    DataArray_Points_elem = ET.Element("DataArray")
    DataArray_Points_elem.set("type", "Float64")
    DataArray_Points_elem.set("NumberOfComponents", "3")
    DataArray_Points_elem.set("format", "ascii")
    # link the coordinates of every point to a string variable
    coordinate_list_1D = sum(dof_coord_list, [])
    coordinate_string = " ".join(list(map(str, coordinate_list_1D)))
    # write text of DataArray element
    DataArray_Points_elem.text = coordinate_string
    # add DataArray to Points
    points_elem.append(DataArray_Points_elem)
    # add Points to Piece
    piece_elem.append(points_elem)

    # create Cells -- Piece->
    cells_elem = ET.Element("Cells")
    # add Cells to Piece
    piece_elem.append(cells_elem)

    # create DataArray(connectivity) -- Cells->
    DataArray_Cells_connectivity_elem = ET.Element("DataArray")
    DataArray_Cells_connectivity_elem.set("type", "UInt32")
    DataArray_Cells_connectivity_elem.set("Name", "connectivity")
    DataArray_Cells_connectivity_elem.set("format", "ascii")
    # link point index inside every cell to a string variable
    connect_index_list = []
    for i in range(num_cells):
        cell_i = list(map(str, info_mesh[i].keys()))
        connect_index_list.append(cell_i)
    connect_index_string = " ".join(sum(connect_index_list, []))
    # write text of DataArray(connectivity)
    DataArray_Cells_connectivity_elem.text = connect_index_string
    # add DataArray(connectivity) to Cells
    cells_elem.append(DataArray_Cells_connectivity_elem)

    # create DataArray(offsets) -- Cells->
    DataArray_Cells_offsets_elem = ET.Element("DataArray")
    DataArray_Cells_offsets_elem.set("type", "UInt32")
    DataArray_Cells_offsets_elem.set("Name", "offsets")
    DataArray_Cells_offsets_elem.set("format", "ascii")
    # offset into the connectivity array for the end of each cell
    # because of all the grid are belong to same type--tetrahedron, so the offsets are all the same
    # every four points form a cell, so the offsets are 4, 8, 12, ...
    # if the cell are tetrahedron, triangle, line, triangle, ..., then the offsets are 4, 7, 9, 12, ...
    offsets_list = list(range(4, 4 * num_cells + 1, 4))
    offsets_string = " ".join(list(map(str, offsets_list)))
    # write text of DataArray(offsets)
    DataArray_Cells_offsets_elem.text = offsets_string
    # add DataArray(offsets) to Cells
    cells_elem.append(DataArray_Cells_offsets_elem)

    # create DataArray(types) -- Cells->
    DataArray_Cells_types_elem = ET.Element("DataArray")
    DataArray_Cells_types_elem.set("type", "UInt8")
    DataArray_Cells_types_elem.set("Name", "types")
    DataArray_Cells_types_elem.set("format", "ascii")
    # because all the grid are tetrahedron, the type of which is 10
    types_list = ["10"] * num_cells
    types_string = " ".join(types_list)
    # write text of DataArray(types)
    DataArray_Cells_types_elem.text = types_string
    # add DataArray(types) to Cells
    cells_elem.append(DataArray_Cells_types_elem)

    # create PointData -- Piece->
    PointData_elem = ET.Element("PointData")
    name_PointData_DataArray = "function"
    PointData_elem.set("Scalars", name_PointData_DataArray)

    # create DataArray(function) -- PointData->
    DataArray_PointData_elem = ET.Element("DataArray")
    DataArray_PointData_elem.set("type", "Float64")
    DataArray_PointData_elem.set("Name", name_PointData_DataArray)
    DataArray_PointData_elem.set("format", "ascii")
    # receive the input parameter "func_value_vec", which is function value of every point
    # transform the numpy 1D array to be a string variable

    # func_value_list = list(func_value_vec)
    # func_value_string = " ".join(func_value_list)
    func_value_string = np.array2string(func_value_vec, formatter={'float_kind':lambda x: "%.2f" % x})[1:-1]
    # write text of DataArray(function)
    DataArray_PointData_elem.text = func_value_string
    # add DataArray(function) to PointData
    PointData_elem.append(DataArray_PointData_elem)
    # add PointData to Piece
    piece_elem.append(PointData_elem)

    # write out the whole xml format data to be a .vtu file
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" ")
    output_file = '/Potential3D.vtu'
    with open(path_Files + output_file, 'w') as f:
        f.write(xmlstr)


def atomPos2XYZ(dict_coordinate, path):
    """
    write the atom position to a file as .xyz format
    :param dict_coordinate: cnt.get_coordinate()
    :param path: choose a path to save the file, maybe 'Jiezi/Jiezi/Files'
    :return: None
    """
    coord_list = list(dict_coordinate.values())
    num_points = len(coord_list)
    lines = [None] * (num_points + 2)
    lines[0] = str(num_points)+'\n'
    lines[1] = "Jiezi output atom position"+'\n'
    for point_index in range(num_points):
        x, y, z = coord_list[point_index]
        line = 'C'+'\t'+str(x)+'\t'+str(y)+'\t'+str(z)+'\n'
        lines[point_index+2] = line
    with open(path+"cnt_atom.xyz", "w") as f:
        f.writelines(lines)


def spectrumXY2Dat(E_list, length_single_cell, num_cell, path, fileName):
    # number of cell(for current, the size is bigger, should be plus 1)
    if fileName == "SpectrumXYForCurrent.dat":
        num_pos = num_cell + 1
    else:
        num_pos = num_cell
    # number of energy
    num_energy = len(E_list)
    lines = [None] * (num_pos * num_energy)
    for ee in range(num_energy):
        for zz in range(num_pos):
            coord_z = (zz + 0.5) * length_single_cell
            line = str(E_list[ee]) + '\t' + str(coord_z) + '\n'
            lines[ee * num_pos + zz] = line
    with open(path + '/' + fileName, 'w') as f:
        f.writelines(lines)


def spectrumZ2Dat(quantity, path, fileName):
    # number of position
    num_pos = len(quantity[0])
    # number of energy
    num_energy = len(quantity)
    lines = [None] * (num_pos * num_energy)
    for ee in range(num_energy):
        for zz in range(num_pos):
            line = str(quantity[ee][zz]) + '\n'
            lines[ee * num_pos + zz] = line
    with open(path + '/' + fileName, 'w') as f:
        f.writelines(lines)








# dof_coord_list = [[2.35752646, 0.0, 42.7608], [2.35752646, 0.0, 27.79452], [2.35752646, 0.0, 14.96628]]
# info_mesh = [{886: [-0.68894701, 2.25461372, 1.51794568], 825: [-0.2416354, 2.34511052, 0.75620287],
#               2146: [0.11525289, 1.81178704, 1.10159407], 898: [0.16256059, 2.35191519, 1.50499037]},
#              {624: [1.42500991, 1.87810484, 37.96560497], 2075: [0.29164866, 1.72144342, 37.82975152],
#               1882: [0.73067054, 1.1839084, 37.38776805], 671: [0.80031953, 2.21752557, 37.57884912]},
#              {1189: [0.5942274, -0.51423894, 42.3711228], 4: [0.78584215, 0.0, 42.7608],
#               1786: [1.34317449, -0.51493974, 42.15214453], 221: [1.02446811, -0.60801637, 42.7608]}]
# func_value_vec = np.array([1, 2, 3])
# trans_VTK_xml(func_value_vec, dof_coord_list, info_mesh)


# a = [[2.35752646, 0.0, 42.7608], [212, 6], [1]]
# c = sum(a, [])
# b = " ".join(list(map(str, c)))
# print(b)

# a = [{12:[1, 1], 5:[2]}, {23:[1, 1], 15:[2]}]
# connect_index_list = []
# for i in range(len(a)):
#     cell_i = list(map(str, a[i].keys()))
#     connect_index_list.append(cell_i)
# connect_index_string = " ".join(sum(connect_index_list, []))
# print(connect_index_string)

# a = list(range(3, 9, 3))
# print(a)

# a = ["5"] * 3
# print(a)
# print(" ".join(a))

# a = "s"
# b = 's'
# print(a+b)

# a = np.ones([3, 1])
# print(a[:, 0])

