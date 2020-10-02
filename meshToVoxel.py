# Using mesh_to_sdf's mesh_to_voxels function to create a voxellized SDF
#   for a given mesh
# Using example code from documentation on PyPi and GitHub
# Author: Dalton Winans-Pruitt
#

from mesh_to_sdf import mesh_to_voxels

import trimesh
import skimage.measure

import time, os
import numpy as np

voxel_resolution = 512
mesh_names = ['meshes\\crate\\Wooden Crate.obj','meshes\\link\\Toon_Link_Windwaker_Posed.gltf' ]
mesh_id = 0 # 0 = Crate, 1 = Link, ... nothing yet

if mesh_id > 1:
    print("That is not a valid mesh number!"), exit()


print("Loading mesh...", end="\t")
start = time.time()
#filetype = mesh_names[mesh_id].find
mesh = trimesh.load(os.getcwd() + "\\" + mesh_names[mesh_id], file_type='obj', force="mesh")
#mesh = trimesh.load('C:\\Users\\dprui\\OneDrive - Mississippi State University' +
#            '\\4 - Fall 2019\\2 - CSE 6413 - Computer Graphics\\Final Project Stuff\\' +
#            'Graphics Final - Python\\Toon_Link\\ssbb_toon_link\\Toon_Link_Windwaker_Posed.gltf')

print(f"{time.time() - start:.02f}s")

print("mesh_to_voxels...", end="\t")
start = time.time()
voxels = mesh_to_voxels(mesh, voxel_resolution, sign_method='depth', pad=False)
time_to_voxelize = time.time() - start
print(f"{time_to_voxelize:.02f}s")

'''
print(len(voxels))
for i in range(2):
    print(voxels[i])
print("\t...")
for i in range(2):
    print(voxels[len(voxels)-(2-i)])
'''

'''
print("marching cubes...", end="\t")
start = time.time()
vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
print(f"{time.time() - start:.02f}s")

print("Trimesh...", end="\t")
start = time.time()
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
print(f"{time.time() - start:.02f}s")

print("Show mesh...", end="\t")
start = time.time()
mesh.show()
print(f"{time.time() - start:.02f}s")
'''
try:
    os.chdir("SDFs")
except FileNotFoundError:
    os.mkdir("SDFs")
    os.chdir("SDFs")

if mesh_id == 0:
    # Crate
    np.save("crate_voxels_nopad_depthSign_res-"+str(voxel_resolution) +
            "_time-"+f"{round(time_to_voxelize):d}s", voxels)
elif mesh_id == 1:
    # Link
    np.save("ToonLink_voxels_nopad_depthSign_res-"+str(voxel_resolution) +
            "_time-"+f"{round(time_to_voxelize):d}s", voxels)
# TODO: Continue in future


# Old method that required reshaping and saving to text...
'''
print("Before Reshaping: ", end="\t")
print(f"Length = {len(voxels)}", end="\t")
print(f"Shape = {voxels.shape}", end="\t")
'''
#voxels2D = np.reshape(voxels, newshape=((voxel_resolution)**2, voxel_resolution))

'''
print("After Reshaping: ", end="\t")
print(f"Length = {len(voxels2D)}", end="\t")
print(f"Shape = {voxels2D.shape}", end="\t")
print(voxels2D)
exit()
'''
#np.savetxt("Crate_voxels_nopad_depthSign_res-"+str(voxel_resolution)+"_time-"+f"{round(time_to_voxelize):d}s",
#           voxels2D, delimiter=",")

