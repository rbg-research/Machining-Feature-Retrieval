import os
import numpy as np
import open3d as o3d
import torch
from numpy import linalg as LA
import openmesh as om
import MinkowskiEngine as ME

# def down_sample(pcd):
#     down_sample = int(len(pcd.points)/500)
#     if down_sample == 0.0:
#         down_sample = 1
#     pcd = pcd.uniform_down_sample(every_k_points=down_sample)
#     return pcd


def down_sample(pcd):
    if(len(pcd.points)>10000):
        pcd = pcd.voxel_down_sample(voxel_size=0.5)
    elif(len(pcd.points)>1000):
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
    return pcd




def get_features(ply_file, model, device):
    try:
          pcd = o3d.io.read_point_cloud(ply_file)
    except MemoryError :
        print("cannot read file:/t",ply_file)
    
    try:
         if(len(pcd.points)>0):
                #pcd = pcd.voxel_down_sample(voxel_size=0.1)
                pcd = down_sample(pcd)
    except MemoryError :
        print("cannot downsample file:/t",ply_file) 
        
    try:
         temp , feature = extract_features(model, xyz=np.array(pcd.points), voxel_size=0.3, device=device, skip_check=True)
    except MemoryError :
        print("cannot extract features of:/t",ply_file) 
    
    return temp, feature

# def get_pcd(ply_file):
#     pcd = o3d.io.read_point_cloud(ply_file)
#     len_pcd = len(pcd.points)
#     return  pcd,len_pcd

# def get_features(pcd, model, device):
    
#     try:
#          if(len(pcd.points)>0):
#                 pcd = pcd.voxel_down_sample(voxel_size=0.05)
#     except MemoryError :
#         print("cannot downsample file:/t",ply_file) 
        
#     try:
#          temp, feature = extract_features(model, xyz=np.array(pcd.points), voxel_size=0.3, device=device, skip_check=True)
#     except MemoryError :
#         print("cannot extract features of:/t",ply_file) 
    
#     feat =feature.detach().cpu().numpy()
    
#     return feat


def stl2ply_convert(stl_folder,ply_folder):
    file_names = list()
    file_paths = list()
    features = list()
    sub_folders = os.listdir(stl_folder)
    for sub_folder in sub_folders:
            stl_subfolder_path = os.path.join(stl_folder, sub_folder)
            ply_subfolder_path = os.path.join(ply_folder, sub_folder)
            if not os.path.isdir(ply_subfolder_path):
                os.makedirs(ply_subfolder_path)  
            stl_files = os.listdir(stl_subfolder_path)
            for stl_file in stl_files:
                if stl_file.endswith(".STL"):
                    stl_file_path = os.path.join(stl_subfolder_path, stl_file)
                    ply_file_path = os.path.join(ply_subfolder_path, stl_file.split(".stl")[0] + ".ply")
                    mesh = om.read_polymesh(stl_file_path)
                    om.write_mesh(ply_file_path, mesh)


def feature_extract(ply_folder, model, device):
    file_names = list()
    file_paths = list()
    features = list()
    sub_folders = os.listdir(ply_folder)
    idx=0
    for sub_folder in sub_folders:
            sub_folder_path = os.path.join(ply_folder, sub_folder)
            ply_files = os.listdir(sub_folder_path)
            for ply_file in ply_files:
                if ply_file.endswith(".ply"):
                    ply_file_path = os.path.join(sub_folder_path, ply_file)
                    file_names.append(sub_folder)
                    #print("file number:",idx)
                    #print("/tfile name:",ply_file)
                    _, feature = get_features(ply_file_path, model, device)
                    idx+=1
                    features.append(feature.detach().cpu().numpy())
                    file_paths.append(os.path.join(ply_file_path))
    return file_names, file_paths, features


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)

  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

  return return_coords, model(stensor).F