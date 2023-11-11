#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.3.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro')

####################################################
##       Begin of NoteBook variables section      ##
####################################################
notebook.set("r_inter", 1.5716843087568109)
notebook.set("width_cnt", 3.1433686175136217)
notebook.set("r_outer", "r_inter+width_cnt")
notebook.set("width_oxide", 9.995912203693317)
notebook.set("r_oxide", "r_outer+width_oxide")
notebook.set("z_total", 256.5648)
notebook.set("zlength_oxide", 42.8463216)
notebook.set("z_translation", "0.5*(z_total-zlength_oxide)")
####################################################
##        End of NoteBook variables section       ##
####################################################
###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Circle_inter = geompy.MakeCircle(O, OZ, "r_inter")
Circle_cnt = geompy.MakeCircle(O, OZ, "r_outer")
Circle_oxide = geompy.MakeCircle(O, OZ, "r_oxide")
Face_inter = geompy.MakeFaceWires([Circle_inter], 1)
Face_cnt = geompy.MakeFaceWires([Circle_cnt], 1)
Face_oxide = geompy.MakeFaceWires([Circle_oxide], 1)
Extrusion_inter = geompy.MakePrismVecH(Face_inter, OZ, "z_total")
Extrusion_cnt = geompy.MakePrismVecH(Face_cnt, OZ, "z_total")
Extrusion_oxide = geompy.MakePrismVecH(Face_oxide, OZ, "zlength_oxide")
Extrusion_air = geompy.MakePrismVecH(Face_oxide, OZ, "z_total")
geompy.TranslateDXDYDZ(Extrusion_oxide, 0, 0, "z_translation")
Cut_air_and_oxide = geompy.MakeCutList(Extrusion_air, [Extrusion_oxide], True)
air_outer = geompy.MakeCutList(Cut_air_and_oxide, [Extrusion_cnt], True)
oxide = geompy.MakeCutList(Extrusion_oxide, [Extrusion_cnt], True)
cnt = geompy.MakeCutList(Extrusion_cnt, [Extrusion_inter], True)
Partition_whole = geompy.MakePartition([Extrusion_inter, air_outer, oxide, cnt], [], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
[air_outer_1,oxide_1,air_inter,cnt_1,air_outer_2] = geompy.ExtractShapes(Partition_whole, geompy.ShapeType["SOLID"], True)
Group_vol_air_outer = geompy.CreateGroup(Partition_whole, geompy.ShapeType["SOLID"])
geompy.UnionIDs(Group_vol_air_outer, [15, 37])
Group_vol_air_inter = geompy.CreateGroup(Partition_whole, geompy.ShapeType["SOLID"])
geompy.UnionIDs(Group_vol_air_inter, [2])
Group_vol_cnt = geompy.CreateGroup(Partition_whole, geompy.ShapeType["SOLID"])
geompy.UnionIDs(Group_vol_cnt, [67])
Group_vol_oxide = geompy.CreateGroup(Partition_whole, geompy.ShapeType["SOLID"])
geompy.UnionIDs(Group_vol_oxide, [59])
Group_face_natural = geompy.CreateGroup(Partition_whole, geompy.ShapeType["FACE"])
geompy.UnionIDs(Group_face_natural, [11, 69, 24, 13, 72, 51, 39, 17])
Group_face_gate = geompy.CreateGroup(Partition_whole, geompy.ShapeType["FACE"])
geompy.UnionIDs(Group_face_gate, [61])
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Circle_inter, 'Circle_inter' )
geompy.addToStudy( Circle_cnt, 'Circle_cnt' )
geompy.addToStudy( Circle_oxide, 'Circle_oxide' )
geompy.addToStudy( Face_inter, 'Face_inter' )
geompy.addToStudy( Face_cnt, 'Face_cnt' )
geompy.addToStudy( Face_oxide, 'Face_oxide' )
geompy.addToStudy( Extrusion_inter, 'Extrusion_inter' )
geompy.addToStudy( Extrusion_cnt, 'Extrusion_cnt' )
geompy.addToStudy( Extrusion_oxide, 'Extrusion_oxide' )
geompy.addToStudy( Extrusion_air, 'Extrusion_air' )
geompy.addToStudy( Cut_air_and_oxide, 'Cut_air_and_oxide' )
geompy.addToStudy( air_outer, 'air_outer' )
geompy.addToStudy( oxide, 'oxide' )
geompy.addToStudy( cnt, 'cnt' )
geompy.addToStudy( Partition_whole, 'Partition_whole' )
geompy.addToStudyInFather( Partition_whole, air_outer_1, 'air_outer_1' )
geompy.addToStudyInFather( Partition_whole, oxide_1, 'oxide' )
geompy.addToStudyInFather( Partition_whole, air_inter, 'air_inter' )
geompy.addToStudyInFather( Partition_whole, cnt_1, 'cnt' )
geompy.addToStudyInFather( Partition_whole, air_outer_2, 'air_outer_2' )
geompy.addToStudyInFather( Partition_whole, Group_vol_air_outer, 'Group_vol_air_outer' )
geompy.addToStudyInFather( Partition_whole, Group_vol_air_inter, 'Group_vol_air_inter' )
geompy.addToStudyInFather( Partition_whole, Group_vol_cnt, 'Group_vol_cnt' )
geompy.addToStudyInFather( Partition_whole, Group_vol_oxide, 'Group_vol_oxide' )
geompy.addToStudyInFather( Partition_whole, Group_face_natural, 'Group_face_natural' )
geompy.addToStudyInFather( Partition_whole, Group_face_gate, 'Group_face_gate' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

Mesh_whole = smesh.Mesh(Partition_whole)
NETGEN_1D_2D_3D = Mesh_whole.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
NETGEN_3D_Parameters_1 = NETGEN_1D_2D_3D.Parameters()
NETGEN_3D_Parameters_1.SetMaxSize(2.5146948940108977)
NETGEN_3D_Parameters_1.SetMinSize(1.5716843087568109)
NETGEN_3D_Parameters_1.SetSecondOrder( 0 )
NETGEN_3D_Parameters_1.SetOptimize( 1 )
NETGEN_3D_Parameters_1.SetFineness( 2 )
NETGEN_3D_Parameters_1.SetChordalError( -1 )
NETGEN_3D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_3D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_3D_Parameters_1.SetFuseEdges( 1 )
NETGEN_3D_Parameters_1.SetQuadAllowed( 0 )
NETGEN_1D_2D_3D_1 = Mesh_whole.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D,geom=Group_vol_cnt)
NETGEN_3D_Parameters_2 = NETGEN_1D_2D_3D_1.Parameters()
NETGEN_3D_Parameters_2.SetMaxSize(1.7602864258076283)
NETGEN_3D_Parameters_2.SetMinSize(1.1001790161297675)
NETGEN_3D_Parameters_2.SetSecondOrder( 0 )
NETGEN_3D_Parameters_2.SetOptimize( 1 )
NETGEN_3D_Parameters_2.SetFineness( 2 )
NETGEN_3D_Parameters_2.SetChordalError( -1 )
NETGEN_3D_Parameters_2.SetChordalErrorEnabled( 0 )
NETGEN_3D_Parameters_2.SetUseSurfaceCurvature( 1 )
NETGEN_3D_Parameters_2.SetFuseEdges( 1 )
NETGEN_3D_Parameters_2.SetQuadAllowed( 0 )
NETGEN_3D_Parameters_2.SetCheckChartBoundary( 0 )
isDone = Mesh_whole.Compute()
Group_vol_air_outer_1 = Mesh_whole.GroupOnGeom(Group_vol_air_outer,'Group_vol_air_outer',SMESH.VOLUME)
Group_vol_air_inter_1 = Mesh_whole.GroupOnGeom(Group_vol_air_inter,'Group_vol_air_inter',SMESH.VOLUME)
Group_vol_cnt_1 = Mesh_whole.GroupOnGeom(Group_vol_cnt,'Group_vol_cnt',SMESH.VOLUME)
Group_vol_oxide_1 = Mesh_whole.GroupOnGeom(Group_vol_oxide,'Group_vol_oxide',SMESH.VOLUME)
Group_face_natural_1 = Mesh_whole.GroupOnGeom(Group_face_natural,'Group_face_natural',SMESH.FACE)
Group_face_gate_1 = Mesh_whole.GroupOnGeom(Group_face_gate,'Group_face_gate',SMESH.FACE)
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_whole.dat' )
  pass
except:
  print('ExportDAT() failed. Invalid file name?')
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_face_natural.dat', Group_face_natural_1 )
  pass
except:
  print('ExportPartToDAT() failed. Invalid file name?')
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_face_gate.dat', Group_face_gate_1 )
  pass
except:
  print('ExportPartToDAT() failed. Invalid file name?')
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_vol_air_outer.dat', Group_vol_air_outer_1 )
  pass
except:
  print('ExportPartToDAT() failed. Invalid file name?')
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_vol_air_inter.dat', Group_vol_air_inter_1 )
  pass
except:
  print('ExportPartToDAT() failed. Invalid file name?')
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_vol_cnt.dat', Group_vol_cnt_1 )
  pass
except:
  print('ExportPartToDAT() failed. Invalid file name?')
try:
  Mesh_whole.ExportDAT( r'/home/zjy/salome/SALOME-9.3.0-UB18.04-SRC/myPro/Mesh_vol_oxide.dat', Group_vol_oxide_1 )
  pass
except:
  print('ExportPartToDAT() failed. Invalid file name?')
Submesh_cnt = NETGEN_1D_2D_3D_1.GetSubMesh()


## Set names of Mesh objects
smesh.SetName(NETGEN_1D_2D_3D.GetAlgorithm(), 'NETGEN 1D-2D-3D')
smesh.SetName(NETGEN_3D_Parameters_2, 'NETGEN 3D Parameters_2')
smesh.SetName(NETGEN_3D_Parameters_1, 'NETGEN 3D Parameters_1')
smesh.SetName(Group_face_natural_1, 'Group_face_natural')
smesh.SetName(Group_face_gate_1, 'Group_face_gate')
smesh.SetName(Mesh_whole.GetMesh(), 'Mesh_whole')
smesh.SetName(Group_vol_oxide_1, 'Group_vol_oxide')
smesh.SetName(Group_vol_cnt_1, 'Group_vol_cnt')
smesh.SetName(Group_vol_air_inter_1, 'Group_vol_air_inter')
smesh.SetName(Group_vol_air_outer_1, 'Group_vol_air_outer')
smesh.SetName(Submesh_cnt, 'Submesh_cnt')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
