from scipy.spatial import cKDTree as KDTree
import open3d as o3d
import torch
import numpy as np
import os

def nearest_distances(x, y):
    inner = -2 * torch.matmul(x.transpose(2, 1), y)
    xx = torch.sum( x**2, dim=1, keepdim=True)
    yy = torch.sum( y**2, dim=1, keepdim=True)
    pairwise_distance = xx.transpose(2, 1) + inner + yy
    nearest_distance = torch.sqrt(torch.min(pairwise_distance, dim=2, keepdim=True).values)
    return nearest_distance

def self_nearest_distances(x):
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # x B 3 N
    xx = torch.sum( x**2, dim=1, keepdim=True)
    pairwise_distance = xx.transpose(2, 1) + inner + xx
    pairwise_distance += torch.eye(x.shape[2]).to(pairwise_distance.device) * 2
    nearest_distance = torch.sqrt(torch.min(pairwise_distance, dim=2, keepdim=True).values)
    return nearest_distance

def self_nearest_distances_K(x, k=3):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum( x**2, dim=1, keepdim=True)
    pairwise_distance = xx.transpose(2, 1) + inner + xx
    pairwise_distance += torch.eye(x.shape[2]).to(pairwise_distance.device) * 2
    pairwise_distance *= -1
    k_nearest_distance = pairwise_distance.topk(k=k, dim=2)[0]
    k_nearest_distance *= -1
    k_nearest_distance = torch.clamp(k_nearest_distance, min=0)
    nearest_distance = torch.sqrt(torch.mean(k_nearest_distance, dim=2, keepdim=True))
    return nearest_distance

def write_pc(point_cloud, output_path):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utilit.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(output_path, point_cloud_o3d)



def farthest_point_sampling(xyz, npoint):
    N, _ = xyz.shape
    centroids = []
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids.append(farthest)
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.where(distance == np.max(distance))[0][0])
    return centroids


def split_block_pointcloud(datapath, minPts, delta_xyz, sampling = True):
    pcFileList = os.listdir(datapath)
    sampling = sampling
    for tempFile in pcFileList:
        print(tempFile)
        pc_item_path = os.path.join(datapath, tempFile)
        temp_pc = np.loadtxt(pc_item_path)
        max_x, max_y, max_z = np.max(temp_pc, axis=0)
        min_x, min_y, min_z = np.min(temp_pc, axis=0)
        temp_min_x = min_x
        temp_x_index = 0
        while (temp_min_x < (max_x + delta_xyz)):
            index_x = np.where((temp_pc[:, 0] > temp_min_x) & (temp_pc[:, 0] < temp_min_x + delta_xyz))[0]
            if np.size(index_x) >= minPts:
                temp_min_y = min_y
                temp_y_index = 0
                while (temp_min_y < (max_y + delta_xyz)):
                    index_xy = np.where((temp_pc[:, 0] > temp_min_x) & (temp_pc[:, 0] < temp_min_x + delta_xyz)
                                        & (temp_pc[:, 1] > temp_min_y) & (temp_pc[:, 1] < temp_min_y + delta_xyz))[0]
                    if np.size(index_xy) >= minPts:
                        temp_min_z = min_z
                        temp_z_index = 0
                        while (temp_min_z < (max_z + delta_xyz)):
                            index_xyz = np.where((temp_pc[:, 0] > temp_min_x) & (temp_pc[:, 0] < temp_min_x + delta_xyz)
                                                 & (temp_pc[:, 1] > temp_min_y) & (
                                                             temp_pc[:, 1] < temp_min_y + delta_xyz)
                                                 & (temp_pc[:, 2] > temp_min_z) & (
                                                             temp_pc[:, 2] < temp_min_z + delta_xyz))[0]
                            if sampling:
                                if np.size(index_xyz) >= minPts:
                                    temp_block_pc = temp_pc[index_xyz]
                                    temp_block_pc_index = farthest_point_sampling(temp_block_pc, minPts)
                                    temp_block_pc_output = temp_block_pc[temp_block_pc_index]
                                    return temp_block_pc_output
                            else:
                                if np.size(index_xyz) >= minPts:
                                    temp_block_pc_output = temp_pc[index_xyz]
                                    return temp_block_pc_output
                            temp_min_z = temp_min_z + delta_xyz
                            temp_z_index += 1
                    temp_min_y = temp_min_y + delta_xyz
                    temp_y_index += 1
            temp_min_x = temp_min_x + delta_xyz
            temp_x_index += 1


def normalize_point_cloud(pc):
    min_xyz = np.min(pc, axis=0)
    max_xyz = np.max(pc, axis=0)
    centroid = min_xyz + (max_xyz - min_xyz)/2.0
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m

def voxelized_pointcloud(input_pc, grid_points, kdtree):
    point_cloud, centroid, m = normalize_point_cloud(input_pc)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    _, idx = kdtree.query(point_cloud)
    occupancies[idx] = 1
    compressed_occupancies = np.packbits(occupancies)
    return compressed_occupancies, centroid, m, point_cloud


def init(bb_min, bb_max, input_res):
    global kdtree, grid_points
    grid_points = create_grid_points_from_bounds(bb_min, bb_max, input_res)
    kdtree = KDTree(grid_points)
    return grid_points, kdtree

def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))
    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list