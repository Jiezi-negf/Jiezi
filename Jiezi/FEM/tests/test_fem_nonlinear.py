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
from Jiezi.FEM.map import map_tocell
from Jiezi.FEM.constant_parameters import constant_parameters
from matplotlib import pyplot as plt
# 111111111111111111111111111111111111111111111111111111111111111111111111111
# solve poisson equation by fenics
# create cnt object
cnt = CNT(n=4, m=4, Trepeat=3, a_cc=1.44, nonideal=False)
cnt.construct()

# use salome to build the FEM grid
geo_para, path_xml = PrePoisson(cnt)
r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation = geo_para
file_xml = path_xml + 'Mesh_whole.xml'
# read mesh file to generate mesh
mesh = Mesh(file_xml)
# define the function space
V = FunctionSpace(mesh, 'Lagrange', 1)
# print(geo_para)


# define the sub_boundary, Dirichlet boundary will be marked 1, the other Neumann boundary is 0
class gate_oxide(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.08
        return ((z_translation - tol) < x[2] < (z_total - z_translation + tol)) and \
               abs(math.sqrt(x[0] ** 2 + x[1] ** 2) - r_oxide) < tol


sub_boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_boundaries.set_all(0)
gate_oxide = gate_oxide()
gate_oxide.mark(sub_boundaries, 1)
# define Dirichlet BC
Dirichlet_BC = -10
bc = DirichletBC(V, Constant(Dirichlet_BC), sub_boundaries, 1)
bc_0 = DirichletBC(V, Constant(0), sub_boundaries, 1)

file = File("fenics_sub_boundaries.xml")
file << sub_boundaries


# define parameter--epsilon
class EPS_FENICS(UserExpression):
    def eval(self, value, x):
        tol = 1e-6
        r = math.sqrt(x[0] ** 2 + x[1] ** 2)
        if r < r_inter + tol or (r > r_outer - tol and (x[2] < z_translation or x[2] > (z_total - z_translation))):
            value[0] = epsilon_air
        elif r_outer + tol > r > r_inter - tol:
            value[0] = epsilon_cnt
        elif r > r_outer - tol and z_translation < x[2] < (z_total - z_translation):
            value[0] = epsilon_oxide


eps_fenics = EPS_FENICS()

# Define variational problem for initial guess (f(u)=0)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
f = Constant(0)
L = f * v * dx
u_k = Function(V)
solve(a == L, u_k, bc)
U_k = list(u_k.vector()[:])





# newton iteration
# define the weak form of newton iteration
du = TrialFunction(V)  # u = u_k + omega*du
a = eps_fenics * dot(grad(du), grad(v)) * dx - 2e-3 * u_k * du * v * dx
L = - eps_fenics * dot(grad(u_k), grad(v)) * dx + 1e-3 * u_k ** 2 * v * dx

# start newton iteration
# relaxation parameter
omega = 1
tol_fenics = 1.0e-5
iter_fenics = 0
res_fenics = 1.0
max_iter_fenics = 120
res_fenics_list = []
du = Function(V)
while res_fenics > tol_fenics and iter_fenics < max_iter_fenics:
    iter_fenics += 1
    # A, b = assemble_system(a, L, bcs_du)
    solve(a == L, du, bc_0)
    res_fenics = np.linalg.norm(np.array(du.vector()), ord=2)
    print("Norm:", res_fenics)
    u_k.vector()[:] = u_k.vector() + omega * du.vector()
    res_fenics_list.append(res_fenics)





# 222222222222222222222222222222222222222222222222222222222222222222222222222222222
# solve poisson equation by my code
path = "/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/" + 'Mesh_whole.dat'
info_mesh, dof_amount, dof_coord_list = create_dof(path)
N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, cnt_cell_list, Dirichlet_list = constant_parameters(info_mesh,
                                                                                                     geo_para)


# rectify function assembly for this test module
def assembly_for_test(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list,
             u_k_cell, dof_amount):
    # create A_total and b_total to store the final matrix and vector in Ax=b
    A_total = matrix_numpy(dof_amount, dof_amount)
    b_total = vector_numpy(dof_amount)

    for cell_index in range(len(info_mesh)):
        # print(cell_index)
        # create a variable A_cell to store one part of A in Ax=b
        # create a variable b_cell to store one part of b in Ax=b
        A_cell = matrix_numpy(4, 4)
        b_cell = vector_numpy(4)

        # u_in_f[:] is the variable which will be regard as the input to the f_DFD and f
        # u_in_f[i] = N_GP_T * u_k, row dot column is a number, u_in_f stores four numbers
        u_in_f = [None] * 4
        u_cell_row = vector_numpy(4)
        for i in range(4):
            u_cell_row.copy([u_k_cell[cell_index]])
            u_in_f[i] = op.vecdotvec(N_GP_T[i], u_cell_row.trans())

        # add the left first term to A_cell
        A_cell = op.addmat(A_cell, cell_long_term[cell_index])

        # compute and add the left second term to A_cell, the operator is minus
        for i in range(4):
            A_cell = op.addmat(A_cell, op.scamulmat(
                func_f_DFD_for_test(u_in_f[i]),
                cell_NNTJ[cell_index][i]).nega())

        # compute and add the right first term to b_cell, the operator is minus
        b_cell = op.addvec(b_cell, op.matmulvec(cell_long_term[cell_index], u_cell_row.trans()).nega())

        # compute and add the right second term to b_cell
        for i in range(4):
            b_cell = op.addvec(b_cell, op.scamulvec(
                func_f_for_test(u_in_f[i]),
                cell_NJ[cell_index][i]))

        # assembly the local matrix A_cell and local vector b_cell to the total matrix A and total vector b
        # cell_dof_index stores the index of dof in the specific cell, such as [23, 677, 89, 2]
        cell_dof_index = list(info_mesh[cell_index].keys())
        for i in range(4):
            for j in range(4):
                value_matA = A_cell.get_value(i, j) + A_total.get_value(cell_dof_index[i], cell_dof_index[j])
                A_total.set_value(cell_dof_index[i], cell_dof_index[j], value_matA)
            value_vecb = b_cell.get_value(i) + b_total.get_value(cell_dof_index[i])
            b_total.set_value((cell_dof_index[i], 0), value_vecb)
    print("finish the loop")
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
        b_total.set_value((D_index, 0), 0)
        # print(i)
    return A_total, b_total


def func_f_DFD_for_test(u):
    return 2e-3 * u


def func_f_for_test(u):
    return 1e-3 * u ** 2


# start the loop
u_k_my = np.ones([dof_amount, 1]) * 0.0
for i in Dirichlet_list:
    u_k_my[i, 0] = Dirichlet_BC
res_my_list = []

# A, b = assembly_for_test(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list,
#                 u_k_cell, dof_amount)
# du = np.linalg.solve(A.get_value(), b.get_value())
# res_my = np.linalg.norm(du, ord=2)
# print(res_my)
while 1:
    u_k_cell = map_tocell(info_mesh, u_k_my)
    A, b = assembly_for_test(info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, Dirichlet_list,
                    u_k_cell, dof_amount)
    du = np.linalg.solve(A.get_value(), b.get_value())
    res_my = np.linalg.norm(du, ord=2)
    print(res_my)
    res_my_list.append(res_my)
    if res_my < 1e-5:
        break
    for i in range(len(u_k_my)):
        u_k_my[i] = u_k_my[i] + du[i][0]









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
for i in range(0, nb_vert):
    tmp_line = lines[i + 1].split(' ')
    temp_coord = [float] * 3
    temp_coord[0] = round(float(tmp_line[1]), 8)
    temp_coord[1] = round(float(tmp_line[2]), 8)
    temp_coord[2] = round(float(tmp_line[3]), 8)
    vertices[i] = temp_coord

u_fenics = []
for i in range(len(u_k_my)):
    u_fenics.append(u_k(vertices[i][0], vertices[i][1], vertices[i][2]))

norm_compare = 0.0
for i in range(len(u_k_my)):
    norm_compare += (u_fenics[i] - u_k_my[i]) ** 2
print(norm_compare)
y = range(0, len(u_k_my))
plt.plot(y, u_fenics, color="green", label="fenics")
plt.plot(y, u_k_my, color="black", label="my")
plt.legend()
plt.show()