import torch
import numpy as np
import MinkowskiEngine as ME


def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec


#sorting the class-wise similarity in descending order

def sort_list(list1, list2): 
    unique_list = []  
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    zipped_pairs = zip(list2, unique_list) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

# zero padding  2d arrays to make the dimensions equal

def zero_pad(x,y):
    shape_pad = [np.maximum(x.shape[0],y.shape[0]),np.maximum(y.shape[1],y.shape[1])]
    x1 = np.zeros(shape_pad)
    y1 = np.zeros(shape_pad)
    x1[:x.shape[0],:x.shape[1]] = x
    y1[:y.shape[0],:y.shape[1]] = y
    return x1,y1