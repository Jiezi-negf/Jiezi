# Jiezi

## Introduction

`Jiezi` is an open-source software developed based on Python framework for simulating the quantum transport of Nanoscale devices. It's an atomistic level simulator that solves the Schrodinger equation and Poisson equation self-consistently by non-equilibrium Green's function (NEGF) method and finite element method (FEM). The tight-binding (TB) typed Hamiltonian matrix built on orbital basis or Wannier basis are all accepted. Given the geometry, the material composition of the electronic device, and the atomic information of the channel, the electronic properties such as current and carrier concentration can be simulated under user-defined bias. At the same time, the microscopic quantity such as the local density of states can be observed. 

In the current version, most of the examples and modules are designed to be convenient for simulating gate-all-around carbon Nanotube Field-effect transistor (CNTFET), but it's extendable to more general cases with tiny changes.

The feasibility of Jiezi was only verified on Linux system.

## Getting started

### Installation

About the installation of Python (recommended version is 3.6 or higher) environment, please refer to https://www.python.org/.

The source distribution of  `Jiezi` can be downloaded from GitHub:

```bash
git clone https://github.com/Jiezi-negf/Jiezi.git
cd Jiezi
```


### Requirements

The list of required python packages can be found in the file `Jiezi/requirements.txt`. It should be noticed that most functions will not be influenced without package `mayavi` which is only for visualization of the atomic ball-stick model on screen. It can be replaced by the software "Vesta".

```bash
pip install -r requirements.txt
```

In addition to the packages required for Python, the following third-party software assistance is helpful:

#### Salome

"Salome" is an open-source computer-aided engineering (CAE) software, which can design geometry shapes and generate mesh. It's helpful to build the shape of device and generate the mesh required for the Poisson equation's solver. These modules can be directly used by GUI or via Python scripts. When designing the device structure for the first time, it's best to operate with GUI. In this process, some key parameters that are possible to be adjusted in the future should be set as variables. After the generation of mesh, the process of designing the shape and meshing the grid can be exported as a Python script. It's very convenient to adjust the shape of this device automatically without GUI by changing the variable value in the script and then loading it. 

Such a  script called `cnt_whole_test.py` about the mesh generation process of gate-all-around CNTFET is offered in the directory `Jiezi/Jiezi/Physics`. After the installation of Salome, a directory named `myPro` needs to be built under the root directory and store the script there. Then if the example "MeshGenerate" is run, the function `PrePoisson` will do the following set of operations: 

- Rectify the values of geometric variables saved in the script file according to the input of the function.
- Start Salome process without GUI to load the script and operate following the script content automatically.
- Kill the process.

The final file `Mesh_whole.dat` will be exported in the directory `myPro` as the complement of function `PrePoisson` and it should be copied to `Jiezi/Jiezi/Files` for other applications. 

All the information about SALOME can be found on its website: https://www.salome-platform.org/

#### ParaView

ParaView is an open-source post-processing visualization engine. Data with a wide range of formats computed from CAE software can be displayed in 3D view and ParaView supports a lot of processing way such as contour, clip, plot over line, and so on. It's chosen to be the recommended data visualization tool of `Jiezi`.  In module `Jiezi/Visualization/Data2File.py`, the function `phi2VTK` accepts the physical quantity value such as the electrostatic potential value on FEM grids and the information of mesh, then export to the VTK format file which can be opened by ParaView. 

The VTK file organized its content by XML-typed structure which looks like this:

```xml
<?xml version="1.0" ?>
<VTKFile type="UnstructuredGrid" version="0.1">
 <UnstructuredGrid>
  <Piece NumberOfCells="27923" NumberOfPoints="5624">
      
   <Points>
    <DataArray NumberOfComponents="3" format="ascii" type="Float64">
        2.35752646 0.0 1.203 42.7608 2.35752646 0.0 ...</DataArray>
   </Points>
      
   <Cells>
    <DataArray Name="connectivity" format="ascii" type="UInt32">
        886 825 2146 ...</DataArray>
    <DataArray Name="offsets" format="ascii" type="UInt32">
        4 8 12 16 20 ...</DataArray>
    <DataArray Name="types" type="UInt8" format="ascii">
        10 10 10 10 10 10 ...</DataArray>
   </Cells>
      
   <PointData Scalars="function">
    <DataArray Name="function" type="Float64" format="ascii" >
        1.00 0.96 0.95 1.00 0.95 0.97 ...</DataArray>
   </PointData>
      
  </Piece>
 </UnstructuredGrid>
</VTKFile>
```

The operation of ParaView and more information about it can be referred to on its website: https://www.paraview.org/

#### VESTA

VESTA is a 3D visualization program for structural models, volumetric data such as electron/nuclear densities, and crystal morphologies. The atomic structure that contains the information of the atomic coordinate and adjacency can be visualized as ball-stick model or other models. As step 4 shown in the example "CNTstructure", the CNT atomic information is exported as the `.xyz` file which is accepted by VESTA.

The homepage of this software is http://jp-minerals.org/vesta/en/.

### Tests

All unit test files are in the directory `Jiezi/Jiezi/tests` . After the installation is completed, user can run the tests file by the following command:

```bash
 python3 -m unittest -v
```

## Examples

### CNTstructure

Below we describe how to build the carbon nanotube structure.

0. Import necessary module, especially `builder`:

```python
import sys
sys.path.append("../../../")
from Jiezi.Visualization.Data2File import atomPos2XYZ
from Jiezi.Graph import builder
from Jiezi.Visualization.Visualization_Graph import visual
```

1. Define the chirality parameter (n, m)  to form a CNT single cell and define the amount of cell :

```python
cnt80 = builder.CNT(8, 0, 1, a_cc=1.42536)   
cnt80.construct()
```

   the default value of carbon-to-carbon distance is 1.42536.

2. All the information about the CNT structure such as the radius, cell length, cell amount, coordinate of atoms, and the atom neighboring relation can be printed on the screen:

```python
cnt80.data_print()
```

3. If the module `mayavi` is available, the ball-stick style figure of atoms can be shown directly:

```python
visual(cnt80)
```

4. If the module `mayavi` is not available, the other method to visualize the atom structure is exporting this structural information in .xyz format to the directory `Jiezi/Files`, which can be opened by specific software such as `vesta` for visualization:

```python
path_xyz = os.path.abspath(os.path.join(__file__, "../..", "Files"))
atomPos2XYZ(cnt80.get_coordinate(), path_xyz)
```

### MeshGenerate

This example describes how to generate the mesh of the finite element method. In the current version, the following method is only suitable for CNTFET. 

0. Import necessary module:

```python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Graph import builder
from Jiezi.Physics.common import *
from Jiezi.Physics.PrePoisson import PrePoisson
```

1. Construct the CNT structure and export the key geometric parameter for the next mesh generation:

```python
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
```

2. Define the other parameters such as thickness of oxide layer:

```python
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
```

3. Final step is to use the  software `salome` to generate the FEM mesh:

```python
geo_para, path_xml = PrePoisson(cnt, width_cnt_scale, width_oxide_scale, z_length_oxide_scale, z_isolation)
```

After that, the file called `Mesh_whole.dat` storing all the mesh information will be created in the directory where you set in the "getting started" part. 

### ReadMeshInfo

This method is more general and can be used for any geometry. Any geometry can be designed by user using the GUI of  `salome` and export the mesh information to the `Mesh_whole.dat`. This file has to be saved in the directory `Jiezi/Files` to make it read by function `create_dof`.

```python
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
```

All the information in mesh such as the cell composition and grid number, grid coordinate is saved in the variable `info_mesh`. 

### HamiltonianTB

This example shows the way to build the tight-binding Hamiltonian matrix based on the CNT object.

0. Import relevant module and create the CNT object.

```python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder
cnt = builder.CNT(n=8, m=0, Trepeat=3, nonideal=False)
cnt.construct()
```

1. Build Hamiltonian matrix based on connections among atoms, on-site value, and hopping value.

```python
H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
lead_H00_L, lead_H00_R = H.get_lead_H00()
lead_H10_L, lead_H10_R = H.get_lead_H10()
print("Tight binding hamiltonian matrix Hii of (8, 0) CNT is:", Hii[1].get_value())
print("Tight binding hamiltonian matrix Hi1 of (8, 0) CNT is:", Hi1[1].get_value())
```

`Hii`and`Hi1` belong to the channel region. `Hii`is the matrix which contains the atoms inside a single block.  `Hi1` is the matrix that describes the coupling between two nearest blocks. `lead_Hxx_L`and`lead_Hxx_R` belong to the left lead region and right lead region respectively. 

### HamiltonianW90

This example shows how to extract useful information from Hamiltonian matrix from the output file of `wannier90`.

0. Use the command "write_hr = True" in file `wannier90.win` to export the file `wannier90_hr.dat` that contains all the information about Hamiltonian matrix. This file needs to be saved in the directory `Jiezi/Files` to be found by the following process.
1. Import modules.

```python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics.w90_trans import read_hamiltonian, latticeVectorInit, w90_supercell_matrix
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder
```

2. Read matrix information from file 'wannier90_hr.dat' and transform it to the data structure that can be used in Jiezi.

```python
path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
path_hr = path_Files + "/wannier90_hr_new.dat"
r_set, hr_set = read_hamiltonian(path_hr)
```

The next optional step is just for the heterogeneous structure such as CNT+graphene which needs to be split into two parts. If there is no demand to partition the whole matrix into separated parts, user can skip the next several steps and add some codes to use the `hr_set` directly.

3. (optional) Reorder the number of wannier bases manually, because the index of wannier bases not the same as the index of original atoms. Those corresponding out-of-order matrix elements need to be swapped. 

```python
hr_cnt_set = [None] * len(hr_set)
hr_total_set = [None] * len(hr_set)
swap_list = [(3, 55), (13, 56), (15, 41), (35, 43)]
for i in range(len(hr_set)):
    hr_cnt_set[i] = matrix_numpy()
    hr_total_set[i] = matrix_numpy()
    hr_temp = matrix_numpy()
    hr_temp.copy(hr_set[i].get_value())
    for j in range(len(swap_list)):
        hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
    ## swapped matrix
    hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
    hr_total_set[i].copy(hr_temp.get_value(0, 72, 0, 72))
```

4. (optional)  Extract the target matrix from the whole matrix. And considering the long-range influence resulting from the wannier basis spread, single cell may not be enough. In this situation, the super cell has to be built to maintain the nearest tight binding form and keep the whole matrix block tridiagonal.

```python
r_1, r_2, r_3, k_1, k_2, k_3 = latticeVectorInit()
k_coord = [0.33333, 0, 0]
num_supercell = 1
Mii_total, Mi1_total = w90_supercell_matrix(r_set, hr_total_set, r_1, r_2, r_3, k_1, k_2, k_3, k_coord, num_supercell)
Mii_cnt, Mi1_cnt = w90_supercell_matrix(r_set, hr_cnt_set, r_1, r_2, r_3, k_1, k_2, k_3,
                                        k_coord, num_supercell)
```

5. Construct the Hamiltonian matrix which has the appropriate data structure according to the matrix extracted in step 4.

```python
cnt = builder.CNT(n=8, m=0, Trepeat=60, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
H.H_readw90(Mii_total, Mi1_total, Mii_cnt, Mi1_cnt, num_supercell)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
LEAD_H00_L, LEAD_H00_R = H.get_lead_H00()
LEAD_H01_L, LEAD_H01_R = H.get_lead_H10()
print("channel layer hamiltonian matrix Hii from wannier90 is:", Hii[1].get_value())
print("channel coupling hamiltonian matrix Hi1 from wannier90 is:", Hi1[1].get_value())
print("lead layer hamiltonian matrix Hii from wannier90 is:", LEAD_H00_L.get_value())
print("lead coupling hamiltonian matrix Hi1 from wannier90 is:", LEAD_H01_L.get_value())
print("coupling matrix between lead and channel from wannier90 is:", Hi1[0].get_value())
```

The function `read_hamiltonian` is designed for receiving the matrix from wannier90. But the function `H_readw90`can be adapted to a variety of situations with tiny changes, not just for wannier90.

### BandStructure4Chain

The method used in this example is designed for 1D chain covering real 1D atom chain and 1D layered chain such as CNT.

```python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder
import numpy as np
import matplotlib.pyplot as plt

# build CNT and its Hmiltonian matrix
cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
# compute subband and eigenvector on specific K point
E_subband, U = band.subband(H, k=0)
```

Because of quantum confinement in XY plane, the subband can be observed in band structure computation. `E_subband[z]` is a list containing the eigenvalues on given k and z position and `U`is the corresponding eigenvector. Both of them are reordered by the values of eigenvalues from low to high.

For CNT, the minimum conduction band energy value is at the point where k=0. So the energy band gap and the bottom of conduction band can be got by the following code:

```python
Ec, Ev, Eg = get_EcEg(Hii, Hi1)
print("Ec is", Ec)
print("Ev is", Ev)
```

As long as the parameters of the k-point list are set, the band structure can be computed and plotted on the screen:

```python
k_total, energy_band = band.band_structure(H, -np.pi, np.pi, 0.1)
i = 0
for band_k in energy_band:
    k = np.ones(energy_band[0].get_size()) * k_total[i]
    i += 1
    # # normalization of energy value (real value divided by the hopping value 2.97)
    # plt.scatter(k, band_k.get_value() / 2.97)
    plt.scatter(k, band_k.get_value(), s=10, c='r')
plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((-12, 12))
plt.show()
```

### BandStructure4W90

Although the name of this example is "for wannier90", it can be applied to any Hamiltonian matrix to compute and plot the band structure.

0. Import the necessary modules in `Jiezi/Physics/w90_trans.py`.

```python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics.w90_trans import read_hamiltonian, read_kpt, computeEKonKpath, plotBandAlongKpath
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.vector_numpy import vector_numpy
import math
```

1. Extract the set of real space vector coordinate `r_set` and the set of corresponding coupling Hamiltonian matrix `hr_set` from wannier90 output file `wannier90_hr.dat` just like what is shown in example "HamiltonianW90".

```python
path_Files = "/home/zjy/Jiezi/Jiezi/Files/"
path_hr = path_Files + "wannier90_hr_new.dat"
r_set, hr_set = read_hamiltonian(path_hr)
```


2. (Optional) Extract the useful Hamiltonian matrix of the part you want to research.

```python
hr_cnt_set = [None] * len(hr_set)
hr_graphene_set = [None] * len(hr_set)
hr_total_set = [None] * len(hr_set)
for i in range(len(hr_set)):
    hr_cnt_set[i] = matrix_numpy()
    hr_graphene_set[i] = matrix_numpy()
    hr_total_set[i] = matrix_numpy()
    hr_temp = matrix_numpy()
    hr_temp.copy(hr_set[i].get_value())
    # swap for new version
    swap_list = [(3, 55), (13, 56), (15, 41), (35, 43)]
    for j in range(len(swap_list)):
        hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
    ## swapped matrix
    hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
    hr_graphene_set[i].copy(hr_temp.get_value(0, 40, 0, 40))
    hr_total_set[i].copy(hr_temp.get_value(0, 72, 0, 72))
```

3. Define the base vector in real space and k space.

```python
r_1 = vector_numpy(3)
r_2 = vector_numpy(3)
r_3 = vector_numpy(3)
r_1.set_value((0, 0), 24.6881123)
r_2.set_value((1, 0), 40)
r_3.set_value((2, 0), 4.2761526)
k_1 = vector_numpy(3)
k_2 = vector_numpy(3)
k_3 = vector_numpy(3)
k_1.set_value((0, 0), 2 * math.pi / 24.6881123)
k_2.set_value((1, 0), 2 * math.pi / 40)
k_3.set_value((2, 0), 2 * math.pi / 4.2761526)
```

4. (Optional) Read the sample points along the k space path from the wannier90 output file `wannier90_kpt.dat`. Or the user-defined k-points list is also allowed.

```python
kptFilePath = path_Files + "wannier90_band.kpt"
kptList = read_kpt(kptFilePath)
num_kpt = len(kptList)
```

5. (Optional) Considering neighbors to different levels by deleting different indexes in `r_set` and `hr_set`. By comparing the results at different levels, it is possible to determine how many levels of nearest neighbors to use.

```python
threshold_index = 2
deleted_index = []
for i in range(len(r_set)):
    if abs(r_set[i][2]) > threshold_index:
        deleted_index.append(i)
for i in range(len(deleted_index)):
    r_set.pop(deleted_index[len(deleted_index) - i - 1])
    hr_cnt_set.pop(deleted_index[len(deleted_index) - i - 1])
```

6. Compute all the eigenvalues on each k-point in k-path and then plot the band structure.

```python
matrixWholeEK = computeEKonKpath(kptList, hr_cnt_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3)
import warnings
warnings.filterwarnings("ignore")
plotBandAlongKpath(matrixWholeEK)
```

### SingleEnergyGF

This example shows the process of computing the Green's function with specific energy using the RGF method.

0.  Build the Hamiltonian matrix.

```python
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics import hamilton, rgf, band
from Jiezi.Graph import builder
from Jiezi.Physics.common import *
from Jiezi.LA.matrix_numpy import matrix_numpy

cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
mul = 0.0
mur = 0.0
H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)

Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
S00 = H.get_S00()
lead_H00_L, lead_H00_R = H.get_lead_H00()
lead_H10_L, lead_H10_R = H.get_lead_H10()
```

1.  Compute the subband to determine the energy range.

```python
E_subband, U = band.subband(H, k=0)
nz = cnt.get_Trepeat()
nm = Hii[0].get_size()[0]
min_temp = []
max_temp = []
for energy in E_subband:
    min_temp.append(energy.get_value()[0])
    max_temp.append(energy.get_value()[-1])
min_subband = min(min_temp).real
max_subband = max(max_temp).real

E_start = min(mul, mur) - 10 * KT - 0.0005 - 1
E_end = max(mul, mur) + 10 * KT + 1
E_step = 0.001
E_list = np.arange(E_start, E_end, E_step)
print("Energy list:", E_start, "to", E_end, "Step of energy:", E_step)
```

2. Initialize the phonon self-energy matrix as a zero matrix for simplification.

```python
sigma_r_ph = []
sigma_lesser_ph = []
for ee in range(len(E_list)):
    sigma_ph_ee = []
    for j in range(nz):
        sigma_ph_element = matrix_numpy(nm, nm)
        sigma_ph_ee.append(sigma_ph_element)
    sigma_r_ph.append(sigma_ph_ee)
    sigma_lesser_ph.append(sigma_ph_ee)
```

3. Choose the energy and compute the Green's functions by the RGF method.

```python
ee = 10
eta = 5e-6
G_R, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
    Sigma_right_lesser, Sigma_right_greater = \
    rgf.rgf(ee, E_list, eta, mul, mur, Hii, Hi1, Sii, S00,
        lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
        sigma_lesser_ph, sigma_r_ph)
```

### PhysicalQuantity

Example "SingleEnergyGF" shows Green's function computation on one specific energy, which is the basis to get physical quantities such as carrier density and local density of states. This example loops over the interested energy range and obtains the values of the physical quantity at each energy point. 

The first three steps are the same as example "SingleEnergyGF". After that, the SCBA  loop can be started:

```python
iter_SCBA_max = 10
TOL_SCBA = 1e-100
ratio_SCBA = 0.5
eta = 5e-6
Dac = 0
Dop = 0
G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
    Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE = \
    SCBA.SCBA(E_list, iter_SCBA_max, TOL_SCBA, ratio_SCBA, eta, mul, mur, Hii, Hi1, Sii, S00,
         lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
         sigma_lesser_ph, sigma_r_ph, form_factor, Dac, Dop, omega)
```

Although the code about "Self-consistent Born Approximation(SCBA)" has been written, in this version the code of phonon part in SCBA loop is annotated. User can enable this function by replacing `SCBA` with `SCBAopen`.

```python
energyOP, coupleOP, coupleAC = phononPara(cnt)
G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE = \
SCBAopen(E_list, iter_SCBA_max, TOL_SCBA, ratio_SCBA, eta, mul, mur, Hii_new, Hi1_new, Sii_new, S00,lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
sigma_lesser_ph, sigma_r_ph, form_factor, coupleAC, coupleOP, energyOP)
```

The output matrix contains $\Sigma_{R,L}^{>,<}$, the diagonal element of  $G^R,G^<,G^>$, and the off-diagonal element of $G^<$, which can be used to obtain the carrier density and density of states. It should be noticed that although the name of these variable are lesser or greater, they indicates the $\Sigma^{in/out}$ and $G^{n/p}$.

```python
n_spectrum, p_spectrum = quantity.carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell)
dos = quantity.densityOfStates(E_list, G_R_fullE, volume_cell)
```

Because of the lack of information about electrostatic potential, current can not be obtained. It can be computed only when the Poisson equation is involved which will be shown in example "NEGF-Poisson".

### NEGF-Poisson

The NEGF method focuses on the carrier transport and the Poisson equation determines the property of electrostatic field. The diagonal element of Hamiltonian matrix is adjusted by the potential which is the solution of Poisson equation. At the same time, the source term of Poisson equation is determined by the result of NEGF. So the self-consistent iteration is required to couple the two parts together.

User needs to send six parameters to the main function in the console:

```python
cd directory/Jiezi/Jiezi/Examples
python NEGFPoisson.py mul, mur, V_gate, weight_old, tol_loop, process_id
```

`mul,mur` are Fermi energy levels. `V_gate` is the bias on the gate. `weight_old` is the weight of the old solution in a mixture of old and new solution of Poisson equation. `tol_loop` is the absolute error tolerance in the NEGF-Poisson loop. The last parameter `process_id` is the mark of each process when multiple computational jobs run in parallel, where every computational mission is regarded as one process.

The example can be separated into the following steps:

0. Import modules. (omitted to avoid repetition)

1. Set the path for Input and output files. `path_Files` is the directory saving files shared by all processes. The private files produced in each process will be saved in `path_process_Files`.

```python
	path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
    path_process_Files = os.path.abspath(os.path.join(path_Files, "normal", "process" + str(process_id)))
    print(path_process_Files)
    folder = os.path.exists(path_process_Files)
    if not folder:
        os.makedirs(path_process_Files)
```

2. Redirect the I/O stream to both console and file. Every time the `print()` is called, the message inside the brackets will be printed on the console and saved in `log_print.txt` for the convenience of checking software operation.

```python
    class Logger(object):
        def __init__(self, filename):
            self.console = sys.stdout
            self.file = open(filename, 'w+')

        def write(self, message):
            self.console.write(message)
            self.file.write(message)
            self.flush()

        def flush(self):
            self.console.flush()
            self.file.flush()
    sys.stdout = Logger(path_process_Files + '/log_print.txt')
    print(mul, mur, Dirichlet_BC_gate, weight_old, tol_loop, process_id)
```

3. Construct the structure.

```python
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
    geo_para = [r_inter, r_outer, r_oxide, z_total, zlength_oxide, z_translation, z_isolation]
```

4. Read the existing `Mesh_whole.dat` file to get mesh information. This file should be generated following the example "MeshGenetate" in advance.

```python
    path_dat = path_Files + "/Mesh_whole.dat"
    info_mesh, dof_amount, dof_coord_list = create_dof(path_dat)
    print("amount of element:", len(info_mesh))
    print("amount of dof:", dof_amount)
```

5. Compute the invariant quantity in Newton iteration in FEM to avoid computing the shared quantity repeatedly. Those are only related to mesh instead of function.

```python
    N_GP_T, cell_co, cell_long_term, cell_NJ, cell_NNTJ, mark_list, cnt_cell_list, Dirichlet_list = \
        constant_parameters.constant_parameters(info_mesh, geo_para)
    coord_GP_list = constant_parameters.get_coord_GP_list(cell_co)
```

6. Cut region for FEM. Cut the whole area into different groups with simple geometry in order to speed up the process of assigning any point to its corresponding cell.

```python
    cut_radius = 3
    cut_z = 3
    dict_cell = map.cut(r_oxide, z_total, info_mesh, cut_radius, cut_z)
```

7. Set initial guess and boundary conditions of Poisson equation.

```python
    Dirichlet_BC_source = 1.0
    Dirichlet_BC_drain = 1.0
    Dirichlet_BC = [Dirichlet_BC_source, Dirichlet_BC_gate, Dirichlet_BC_drain]
    phi_guess = np.ones([dof_amount, 1]) * 0.5 * (Dirichlet_BC_gate + Dirichlet_BC_source)
    for type_i in range(len(Dirichlet_list)):
        for i in range(len(Dirichlet_list[type_i])):
            phi_guess[Dirichlet_list[type_i][i], 0] = Dirichlet_BC[type_i]
```

8. (Optional) Set doping and fixed charge. 

```python
    # set doping
    doping_source = 0
    doping_channel = 0
    doping_drain = 0
    print("doping source, channel, drain are:", doping_source, doping_channel, doping_drain)
    # set fixed charge parameters in oxide
    fixedChargeDensity = 0
    fixedChargeScope = [r_outer, (r_outer + r_oxide) / 2, 0, z_total]
    # construct doping concentration
    doping_GP_list = constant_parameters.doping(coord_GP_list, zlength_oxide, z_translation,
                                                doping_source, doping_drain, doping_channel, mark_list)
    # construct fixed charge in oxide
    fixedCharge_GP_list = constant_parameters.fixedChargeInit(coord_GP_list)
    constant_parameters.addFixedCharge(fixedCharge_GP_list, coord_GP_list, mark_list,
                                       fixedChargeScope, fixedChargeDensity)
```

9. Compute conduction band bottom and band gap.

```python
    H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
    H.build_H()
    H.build_S(base_overlap=0.018)
    Ec, Ev, Eg = get_EcEg(Hii, Hi1)
```

   Start the NEGF-Poisson loop:

```python
iter_big_max = 25
iter_big = 0
while iter_big < iter_big_max:
    iter_big += 1
    # display which loop is running
    print("\n")
    print("Loop", iter_big)
```

10. Map the data structure of phi from vector organized to cell organized.

```python
        phi_cell = map.map_tocell(info_mesh, phi_guess)
```

11. Build Hamiltonian matrix with the influence of phi.

```python
        H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        layer_phi_list = H.H_add_phi(dict_cell, phi_cell, cell_co, cut_radius, cut_z, r_oxide, z_total, num_supercell)
```

12. Determine the energy points list. (Omitted to avoid repetition)
13. Start SCBA loop to get interested quantity as shown in example "PhysicalQuantity" (Omitted to avoid repetition)
14. Poisson solver starts. Mode 1 is the non-linear solver, mode 2 is the linear solver and mode 3 is the analytical solver which has not been developed. If mode is 1, the quasi-Fermi level solver will be called so the guess value should be initialized. 

```python
        # set the initial value of ef
        ef_init_n = np.ones([len(info_mesh), 4]) * (-1e2)
        ef_init_p = np.ones([len(info_mesh), 4]) * 1e2
        TOL_ef = 1e-4
        TOL_du = 1e-4
        iter_NonLinearPoisson_max = 20
        mode = 1
        phi = poisson(mode, info_mesh, N_GP_T, cell_long_term, cell_NJ, cell_NNTJ, cnt_cell_list,
                      ef_init_n, ef_init_p, mark_list,
                      Dirichlet_list, Dirichlet_BC, E_list_poi, Ec, Eg, TOL_ef, TOL_du, iter_NonLinearPoisson_max,
                      dos_GP_list, n_GP_list, p_GP_list, doping_GP_list, fixedCharge_GP_list, phi_guess, dof_amount)
```

15. Test whether the difference between two adjacent iterations is small enough.

```python
        residual = np.linalg.norm(phi - phi_guess, ord=2) / phi.shape[0]
        if residual < tol_loop:
```

16. Export data as a file to be identified by other software for visualization. The electrostatic potential is saved in VTK format to be opened by the software `Paraview` to get a better 3-D view. Other data of quantities such as carrier spectrum, current spectrum, and dos are saved in .dat file and organized in the order of energy and position.

```python
            phi2VTK(phi[:, 0].real, dof_coord_list, info_mesh, path_process_Files)
            spectrumXY2Dat(E_list, length_single_cell, num_cell, num_supercell,
                           path_process_Files, "SpectrumXYForCurrent.dat")
            spectrumXY2Dat(E_list, length_single_cell, num_cell, num_supercell,
                           path_process_Files, "SpectrumXYForOthers.dat")
            spectrumZ2Dat(J_spectrum, path_process_Files, "currentSpectrum.dat")
            spectrumZ2Dat(dos, path_process_Files, "densityOfState.dat")
            spectrumZ2Dat(n_spectrum, path_process_Files, "electronSpectrum.dat")
            spectrumZ2Dat(p_spectrum, path_process_Files, "holeSpectrum.dat")
```

## Authors

- JunyanZhu (toshihikozhu@gmail.com)
- ChenSong (chen_song@outlook.com)
- JiangCao (jiang.cao@iis.ee.ethz.ch)

The contribution of each author is shown in  [AUTHORS.md](AUTHORS.md) file.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE.md](LICENSE.md) file for details.
