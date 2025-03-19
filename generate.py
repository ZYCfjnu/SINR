import models.local_model as model
from models.generation import Generator
import torch
import configs.config as cfg_loader
import os
import numpy as np
from models.utils import farthest_point_sampling, init, voxelized_pointcloud

def rotate_mat(radian):
    rot_matrix = np.array([[np.cos(radian), 0, -np.sin(radian)],[0, 1, 0],[np.sin(radian), 0, np.cos(radian)]])
    return rot_matrix

def gen_iterator(cfg, datapath, out_path, minPts,  delta_xyz, gen, sampling=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)
    pcFileList = os.listdir(datapath)
    grid_points, kdtree = init(cfg.bb_min, cfg.bb_max, cfg.input_res)
    num = 0
    for tempFile in pcFileList:
        print(tempFile)
        pc_item_path = os.path.join(datapath, tempFile)
        temp_pc = np.loadtxt(pc_item_path)
        max_x, max_y, max_z = np.max(temp_pc, axis=0)
        min_x, min_y, min_z = np.min(temp_pc, axis=0)
        temp_min_x = min_x
        temp_x_index = 0
        while (temp_min_x < (max_x + delta_xyz)):
            print('x index:', temp_x_index)
            index_x = np.where((temp_pc[:, 0] > temp_min_x) & (temp_pc[:, 0] < temp_min_x + delta_xyz))[0]
            if np.size(index_x) >= minPts:
                temp_min_y = min_y
                temp_y_index = 0
                while (temp_min_y < (max_y + delta_xyz)):
                    print('x index:', temp_x_index, 'y index:', temp_y_index)
                    index_xy = np.where((temp_pc[:, 0] > temp_min_x) & (temp_pc[:, 0] < temp_min_x + delta_xyz)
                                        & (temp_pc[:, 1] > temp_min_y) & (temp_pc[:, 1] < temp_min_y + delta_xyz))[0]
                    if np.size(index_xy) >= minPts:
                        temp_min_z = min_z
                        temp_z_index = 0
                        while (temp_min_z < (max_z + delta_xyz)):
                            print(temp_x_index, temp_y_index, temp_z_index)
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
                                    np.savetxt(out_path + 'canonical_point_cloud_ori_{}.txt'.format(num), temp_block_pc_output)
                                    compressed_occupancies, centroid, m = voxelized_pointcloud(temp_block_pc_output, grid_points, kdtree)
                                    canonical_pointcloud, duration = gen.generate_point_cloud(compressed_occupancies, num_steps=5)
                                    canonical_pointcloud = canonical_pointcloud*m + centroid
                                    final_pc = canonical_pointcloud.copy()
                                    final_pc[:, 0], final_pc[:, 2] = canonical_pointcloud[:, 2], canonical_pointcloud[:, 0]
                                    np.savetxt(out_path + 'canonical_point_cloud_{}.txt'.format(num), final_pc/2)
                                    num += 1
                            else:
                                if np.size(index_xyz) >= minPts:
                                    temp_block_pc_output = temp_pc[index_xyz]
                                    np.savetxt(out_path + 'canonical_point_cloud_ori_{}.txt'.format(num), temp_block_pc_output)
                                    compressed_occupancies, centroid, m, nor_pc = voxelized_pointcloud(temp_block_pc_output, grid_points, kdtree)
                                    canonical_pointcloud, duration = gen.generate_point_cloud(compressed_occupancies, num_steps=5)
                                    rot_matrix = rotate_mat(90*np.pi/180)
                                    canonical_pointcloud = np.matmul(canonical_pointcloud, rot_matrix)
                                    canonical_pointcloud = canonical_pointcloud * np.array([1,1,-1])
                                    canonical_pointcloud = canonical_pointcloud * m + centroid
                                    np.savetxt(out_path + 'canonical_point_cloud_{}.txt'.format(num), canonical_pointcloud)
                                    num += 1
                            temp_min_z = temp_min_z + delta_xyz
                            temp_z_index += 1
                    temp_min_y = temp_min_y + delta_xyz
                    temp_y_index += 1
            temp_min_x = temp_min_x + delta_xyz
            temp_x_index += 1

if __name__=="__main__":
    cfg = cfg_loader.get_config()
    device = torch.device("cuda:0")
    net = model.SINR()
    gen = Generator(net, cfg.exp_dir, device=device)
    out_path = './experiments/{}/evaluation/'.format(cfg.exp_name)
    minPts = 1
    delta_xyz = 10
    datapath = './testData/'
    gen_iterator(cfg, datapath, out_path, minPts,  delta_xyz, gen)
