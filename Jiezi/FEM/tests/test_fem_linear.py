# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import copy
from Jiezi.Graph.builder import CNT
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.Physics.PrePoisson import PrePoisson
from Jiezi.Physics.common import *
from dolfin import *
from Jiezi.FEM.mesh import create_dof
from Jiezi.FEM.map import *
from Jiezi.FEM.constant_parameters import *
from matplotlib import pyplot as plt


# 111111111111111111111111111111111111111111111111111111111111111111111111111
# solve poisson equation by fenics
# create cnt object
cnt = CNT(n=4, m=0, Trepeat=4, nonideal=False)
cnt.construct()
nn = cnt.get_nn()
layer_amount = cnt.get_Trepeat()
atom_coord = cnt.get_coordinate()

# use salome to build the FEM grid
geo_para, path_xml = PrePoisson(cnt)
r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation = geo_para
file_xml = path_xml + 'Mesh_whole.xml'
# read mesh file to generate mesh
mesh = Mesh(file_xml)
# define the function space
V = FunctionSpace(mesh, 'Lagrange', 1)
# print(geo_para)

# cell = Cell(mesh, 0)
# point = Point(0.68397611, -1.67302272, 18.94516959)
# print(cell.contains(point))


# define the sub_boundary, Dirichlet boundary will be marked 1, the other Neumann boundary is 0
class gate_oxide(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.2
        return ((z_translation - 1e-6) < x[2] < (z_total - z_translation + 1e-6)) and \
                abs(math.sqrt(x[0] ** 2 + x[1] ** 2) - r_oxide) < tol
        # return ((z_translation - 1e-6) < x[2] < (z_total - z_translation + 1e-6)) and \
        #     abs(x[0] ** 2 + x[1] ** 2 - r_oxide**2) < tol and on_boundary


sub_boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_boundaries.set_all(0)
gate_oxide = gate_oxide()
gate_oxide.mark(sub_boundaries, 1)
Dirichlet_value = 2.0
# define Dirichlet BC
bc = DirichletBC(V, Constant(Dirichlet_value), sub_boundaries, 1)

# output the boundary points as .xdmf file which will be opened by "paraview" software
file = File("fenics_sub_boundaries.xml")
file << sub_boundaries
xdmf = XDMFFile("boundaries_gate.xdmf")
xdmf.write(sub_boundaries)
xdmf.close()


# define parameter--epsilon
class EPS_FENICS(UserExpression):
    def eval(self, value, x):
        r = math.sqrt(x[0] ** 2 + x[1] ** 2)
        if r < r_inter:
            value[0] = epsilon_air
        if r_outer > r > r_inter:
            value[0] = epsilon_cnt
        if r > r_outer and z_translation < x[2] < (z_total - z_translation):
            value[0] = epsilon_oxide
        if r > r_outer and (x[2] < z_translation or x[2] > (z_total - z_translation)):
            value[0] = epsilon_air_outer


doping_source = 5e-3
doping_channel = 5e-3
doping_drain = 5e-3


# define source term N_doping
class N_DOPING(UserExpression):
    def eval(self, value, x):
        r = math.sqrt(x[0] ** 2 + x[1] ** 2)
        if r_outer > r > r_inter and x[2] < z_translation:
            value[0] = doping_source
        elif r_outer > r > r_inter and z_translation < x[2] < z_translation + zlength_oxide:
            value[0] = doping_channel
        elif r_outer > r > r_inter and x[2] > z_translation + zlength_oxide:
            value[0] = doping_drain
        else:
            value[0] = 0.0


eps_fenics = EPS_FENICS()
N_doping = N_DOPING()

# Define variational problem for initial guess (f(u)=0)
u = TrialFunction(V)
v = TestFunction(V)
a = eps_fenics * dot(grad(u), grad(v)) * dx
f = N_doping
L = f * v * dx
u_k = Function(V)
solve(a == L, u_k, bc)
U_k = list(u_k.vector()[:])


# 222222222222222222222222222222222222222222222222222222222222222222222222222222222
# solve poisson equation by my code
path = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/" + 'Mesh_whole.dat'
info_mesh, dof_amount, dof_coord_list = create_dof(path)
N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, cnt_cell_list, Dirichlet_list = \
    constant_parameters(info_mesh, geo_para)
coord_GP_list = get_coord_GP_list(cell_co)
Dirichlet_list = Dirichlet_list[1]

doping_GP_list = doping(coord_GP_list, zlength_oxide, z_translation,
                        doping_source, doping_drain, doping_channel, mark_list)
n_GP_list = [[0.0] * 4] * len(coord_GP_list)
p_GP_list = [[0.0] * 4] * len(coord_GP_list)


# rectify function assembly for this test module
def assembly_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                    dof_amount, doping_GP_list, n_GP_list, p_GP_list):
    # create A_total and b_total to store the final matrix and vector in Ax=b
    A_total = matrix_numpy(dof_amount, dof_amount)
    b_total = vector_numpy(dof_amount)

    for cell_index in range(len(info_mesh)):
        # create a variable A_cell to store one part of A in Ax=b
        # create a variable b_cell to store one part of b in Ax=b
        A_cell = matrix_numpy(4, 4)
        b_cell = vector_numpy(4)

        # add the left first term to A_cell
        A_cell = op.addmat(A_cell, cell_long_term[cell_index])

        # compute and add the right second term to b_cell
        for i in range(4):
            f = - n_GP_list[cell_index][i] + p_GP_list[cell_index][i] + doping_GP_list[cell_index][i]
            b_cell = op.addvec(b_cell, op.scamulvec(f, cell_NJ[cell_index][i]))

        # assembly the local matrix A_cell and local vector b_cell to the total matrix A and total vector b
        # cell_dof_index stores the index of dof in the specific cell, such as [23, 677, 89, 2]
        cell_dof_index = list(info_mesh[cell_index].keys())
        for i in range(4):
            for j in range(4):
                value_matA = A_cell.get_value(i, j) + A_total.get_value(cell_dof_index[i], cell_dof_index[j])
                A_total.set_value(cell_dof_index[i], cell_dof_index[j], value_matA)
            value_vecb = b_cell.get_value(i) + b_total.get_value(cell_dof_index[i])
            b_total.set_value((cell_dof_index[i], 0), value_vecb)

    # based on the Dirichlet list and Dirichlet boundary condition to adjust the final matrix and final vector
    for i in range(len(Dirichlet_list)):
        D_index = Dirichlet_list[i]
        vec_zero = vector_numpy(dof_amount)
        # # set the column where the dirichlet point locates in to 0
        # A_total.set_block_value(0, dof_amount, D_index, D_index + 1, vec_zero)
        # set the row where the dirichlet point locates in to 0
        A_total.set_block_value(D_index, D_index + 1, 0, dof_amount, vec_zero.trans())
        # set the element about the Dirichlet point to 1
        A_total.set_value(D_index, D_index, 1.0)
        # set the value of b on Dirichlet point to its value based on the Dirichlet boundary condition
        b_total.set_value((D_index, 0), Dirichlet_value)
    # print("finish the assembly process")
    return A_total, b_total

# dirichlet_cell_index = []
# for i in range(len(Dirichlet_list)):
#     point_index = Dirichlet_list[i]
#     for j in range(len(info_mesh)):
#         point_index_list_j = info_mesh[j].keys()
#         for k in point_index_list_j:
#             if k == point_index:
#                 dirichlet_cell_index.append(j)
# dirichlet_cell_index = list(set(dirichlet_cell_index))
# dirichlet_cell_index.sort()
# print(dirichlet_cell_index)
# print(len(dirichlet_cell_index))


A, b = assembly_linear(info_mesh, cell_long_term, cell_NJ, Dirichlet_list, Dirichlet_value,
                    dof_amount, doping_GP_list, n_GP_list, p_GP_list)
u_my = np.linalg.solve(A.get_value(), b.get_value())


# dirichlet_value_my = []
# dirichlet_value_fenics = []
# diff_dirichlet = []
# for i in range(len(Dirichlet_list)):
#     point_index = Dirichlet_list[i]
#
#     dirichlet_value_my.append(u_my[point_index, 0])
#     dirichlet_value_fenics.append(U_k[point_index])
#     diff_dirichlet.append(abs(dirichlet_value_my[i] - dirichlet_value_fenics[i]))
# x_axis = range(0, len(dirichlet_value_my))
# plt.title("value on dirichlet boundary")
# plt.plot(x_axis, dirichlet_value_fenics, color="green", label="fenics")
# plt.plot(x_axis, diff_dirichlet, color="black", label="diff")
# plt.legend()
# plt.show()
# print(u_k(2.48002198, -1.18930763, 9.30589193))
# print(u_my[2750, 0])


# plot dirichlet points in my code
# .xyz format can be opened by "VESTA" software
path_bc = "/home/zjy/Jiezi/Jiezi/FEM/tests/" + "bc_point_my.xyz"
lines = []
lines.append(["   " + str(385) + "\n"])
for i in range(len(Dirichlet_list)):
    line = []
    point_index = Dirichlet_list[i]
    x, y, z = dof_coord_list[point_index]
    line.append("C" + "          ")
    line.append(str(round(x, 8)) + "       " + str(round(y, 8)) + "       " + str(round(z, 8)) + "\n")
    lines.append(line)
with open(path_bc, "w") as f:
    for i in lines:
        f.writelines(i)


## 333333333333333333333333333333333333333333333333333333333333
path = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/"

vert_dim = 3

with open(path + 'Mesh_whole.dat') as file:
    lines = [line.rstrip('\n') for line in file]

first_line = lines[0].split(' ')
nb_vert = int(first_line[0])


# create sub_element "vertices"
vertices = [list] * nb_vert
# filling all vertices
for i in range(nb_vert):  # for in range语句的范围是个左闭右开区间
    tmp_line = lines[i + 1].split(' ')
    temp_coord = [float] * 3
    temp_coord[0] = round(float(tmp_line[1]), 8)
    temp_coord[1] = round(float(tmp_line[2]), 8)
    temp_coord[2] = round(float(tmp_line[3]), 8)
    vertices[i] = temp_coord

u_fenics = []
diff = []
for i in range(len(u_my)):
    u_fenics.append(u_k(vertices[i][0], vertices[i][1], vertices[i][2]))

for i in range(len(u_my)):
    diff.append(u_fenics[i] - u_my[i, 0])

norm_compare = 0.0
for i in range(len(u_my)):
    norm_compare += diff[i] ** 2
print(norm_compare)
y = range(0, len(u_my))
plt.subplot(2, 1, 1)
plt.title("phi on dof points")
plt.plot(y, u_fenics, color="green", label="fenics")
plt.plot(y, u_my[:, 0], color="black", label="my")
# plt.plot(y, diff, color="black", label="diff")
plt.legend()


u_cell = map_tocell(info_mesh, u_my)
phi_atom_list_my = []
phi_atom_list_fenics = []
cut_radius = 3
cut_z = 3
dict_cell = cut(r_oxide, z_total, info_mesh, cut_radius, cut_z)
incell_count = 0
outcell_count = 0
for layer_number in range(layer_amount):
    for i in range(nn):
        atom_number = i + layer_number * nn + 1
        # atom_coord is a list [x, y, z]
        atom_coord_i = atom_coord[atom_number]
        coord_ref, cell_index_my = link_to_cell(dict_cell, atom_coord_i, cell_co, cut_radius, cut_z, r_oxide, z_total)
        x = atom_coord_i[0]
        y = atom_coord_i[1]
        if atom_coord_i[2] < 0:
            z = 0.0
        else:
            z = atom_coord_i[2]
        point_fenics = Point(x, y, z)
        cell_fenics = Cell(mesh, cell_index_my)
        if cell_fenics.contains(point_fenics):
            incell_count += 1
        else:
            outcell_count += 1
        # print(cell_index_my)
        phi_atom_i = projection(dict_cell, u_cell, atom_coord_i, cell_co, cut_radius, cut_z, r_oxide, z_total)
        phi_atom_list_my.append(phi_atom_i)
        phi_atom_list_fenics.append(u_k(round(atom_coord_i[0], 8), round(atom_coord_i[1], 8),
                                        round(abs(atom_coord_i[2]), 8)))
print("number of point found successfully:", incell_count)
print("number of point found unsuccessfully:", outcell_count)
plt.subplot(2, 1, 2)
x_axis = range(0, nn * layer_amount)
plt.title("phi on atom")
plt.plot(x_axis, phi_atom_list_fenics, color="green", label="fenics")
plt.plot(x_axis, phi_atom_list_my, color="black", label="my")
plt.legend()
plt.show()


