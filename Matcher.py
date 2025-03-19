import numpy as np
from scipy.spatial import cKDTree
import os
import sys
import random
import math

targetPath = './experiments/customDataset/evaluation/'
targetList = os.listdir(targetPath)
len_targetList = len(targetList)

print('\nMarching')
result = []
scale = 5
for i, target_file in enumerate(targetList):
    target_file_path = os.path.join(targetPath, target_file)
    target_xyz = np.loadtxt(target_file_path)
    target_tree = cKDTree(target_xyz)
    k = 5
    target_distances, target_indices = target_tree.query(target_xyz, k=k)
    delta = np.mean(target_distances, axis=1)
    delta_avg = np.mean(delta)
    x_min, y_min = np.min(target_xyz[:, :2], axis=0)
    x_max, y_max = np.max(target_xyz[:, :2], axis=0)
    boundary_points = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
    distances_to_boundary_points = np.min([np.sqrt(np.sum((target_xyz[:, :2] - bp)**2, axis=1)) for bp in boundary_points], axis=0)
    sampled_xyz_list = []
    for j in range(len(target_xyz)):
        if (delta[j] >= delta_avg or random.randint(0, 100) <= math.exp(1 + (scale - 1) * (delta[j] / delta_avg))) and distances_to_boundary_points[j] >= delta_avg/2:
            sampled_xyz_list.append(target_xyz[j])
    sampled_xyz = np.array(sampled_xyz_list)
    result.append(sampled_xyz)
    sys.stdout.write('\r')
    sys.stdout.write(f"{(i+1)/len_targetList*100:.1f}% 完成")
    sys.stdout.flush()
print('\nresult stacking')
result = np.vstack(result)
print('\nresult saving')
np.savetxt('./outputs/sample_Canonical.txt', result, fmt="%1.9f")
print('\nfinish')
