from plyfile import PlyData
import numpy as np
import random
from torch.utils.data import DataLoader,Dataset,random_split
import os
import torchvision.transforms as transforms
import torch
import math
import pandas as pd
import time
from scipy.spatial import distance_matrix

# 八种label对应的RGB值
labels = ((160, 160, 160), (96, 25, 134), (180, 212, 101),(129, 81, 28), (235, 104, 163), (0, 158, 150),
          (125, 0, 34), (244, 152, 0), (255, 0, 255), (164, 0, 91), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
           (212, 22, 26))

def get_data(path=""):
    """
    Input:
        path: path of ply file
    Return:
        points: coordinate of points [N, 3]
        label_points: label of points [N, 1]
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
    """
    row_data = PlyData.read(path)  # 读ply文件
    vertex = row_data.elements[0].data  # 点坐标+RGB+alpha信息
    face = row_data.elements[1].data  # face的点索引+face的RGB标签
    n_point = vertex.shape[0]  # 顶点个数
    n_face = face.shape[0]  # 网格个数
    points = np.zeros([n_point, 6])  # 点云坐标+法向量
    index_face = np.zeros([n_face, 3]).astype('int')  # 组成网格的点的索引
    points_face = np.zeros([n_face, 21]).astype('float32')  # 3个点坐标+3个点法向量+中心点坐标
    label_face = np.zeros([n_face, 1]).astype('int64')  # face标签
    normal_face = np.zeros([n_face,3]).astype('float32')  # face的法向量

    for i, data in enumerate(vertex):
        # get coordinate and normal of points
        points[i][:6] = [data[0], data[1], data[2], data[3], data[4], data[5]]

    for i, data in enumerate(face):
        index_face[i, :] = [data[0][0], data[0][1], data[0][2]]  # get index of points
        # get coordinate of  3 point of face
        points_face[i, :3] = points[data[0][0], :3]
        points_face[i, 3:6] = points[data[0][1], :3]
        points_face[i, 6:9] = points[data[0][2], :3]
        points_face[i, 9:12] = points[data[0][0], 3:]
        points_face[i, 12:15] = points[data[0][1], 3:]
        points_face[i, 15:18] = points[data[0][2], 3:]
        # get center point of face
        points_face[i, 18] = (points[data[0][0], 0] + points[data[0][1], 0] + points[data[0][2], 0]) / 3
        points_face[i, 19] = (points[data[0][0], 1] + points[data[0][1], 1] + points[data[0][2], 1]) / 3
        points_face[i, 20] = (points[data[0][0], 2] + points[data[0][1], 2] + points[data[0][2], 2]) / 3
        # get normal of each face
        x1, y1, z1 = points_face[i, :3]
        x2, y2, z2 = points_face[i, 3:6]
        x3, y3, z3 = points_face[i, 6:9]
        normal_face[i, 0] = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
        normal_face[i, 1] = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
        normal_face[i, 2] = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        # get label of face
        R, G, B = data[1], data[2], data[3]
        for j, label in enumerate(labels):
            if R == label[0] and G == label[1] and B == label[2]:
                label_face[i] = j
                break
    return points, index_face, points_face, label_face, normal_face

def get_data_v2(path=""):
    labels = ((160, 160, 160), (96, 25, 134), (180, 212, 101),(129, 81, 28), (235, 104, 163), (0, 158, 150),
          (125, 0, 34), (244, 152, 0), (255, 0, 255), (164, 0, 91), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
           (212, 22, 26))
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    n_face = faces.shape[0]  # number of faces
    n_points = points.shape[0]
    xyz = points[:, :3]  # coordinate of vertex shape=[N, 3]
    label_point = np.zeros([n_points, 1]).astype('int32')
    label_point_onehot = np.zeros([n_points, 15]).astype(('int32'))
    """ index of faces shape=[N, 3] """
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
    norm = points[:, 3:6]

    points_cn = np.concatenate((xyz, norm), axis=1).astype('float32')

    """ RGB of faces shape=[N, 3] """
    RGB_face = points[:, 6:9]
    """ get label of each face """
    for i, label in enumerate(labels):
        label_point[(RGB_face == label).all(axis=1)] = i
        label_point_onehot[(RGB_face == label).all(axis=1), i] = 1
    return index_face, points_cn, label_point, label_point_onehot, xyz

def generate_plyfile(index_face, point_face, label_face, path= " "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    unique_index = np.unique(index_face.flatten())  # get unique points index
    flag = np.zeros([unique_index.max()+1, 2]).astype('uint64')
    order = 0
    with open(path, "a") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(unique_index.shape[0]) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face " + str(index_face.shape[0]) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for i, index in enumerate(index_face):
            for j, data in enumerate(index):
                if flag[data, 0] == 0:  # if this point has not been wrote
                    xyz = point_face[i, 3*j:3*(j+1)]  # Get coordinate
                    xyz_nor = point_face[i, 3*(j+4):3*(j+5)]
                    f.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + " " + str(xyz_nor[0]) + " "
                            + str(xyz_nor[1]) + " " + str(xyz_nor[2]) + "\n")
                    flag[data, 0] = 1  # this point has been wrote
                    flag[data, 1] = order  # give point a new index
                    order = order + 1  # index add 1 for next point

        for i, data in enumerate(index_face):  # write new point index for every face
            RGB = labels[label_face[i, 0]]  # Get RGB value according to face label
            f.write(str(3) + " " + str(int(flag[data[0], 1])) + " " + str(int(flag[data[1], 1])) + " "
                    + str(int(flag[data[2], 1])) + " " + str(RGB[0]) + " " + str(RGB[1]) + " "
                    + str(RGB[2]) + " " + str(255) + "\n")
        f.close()

class plydataset(Dataset):
    """
    Input:
        path: root path
        downsample: type of downsample
        ratio: down sample scale
    Return:
        sampled_index_face: [N, 3]
        sampled_point_face: [N, 12]
        sampled_label_face: [N, 1]
    """
    def __init__(self, path="data/train", array=np.zeros([1,1]), patch_size=14000):
        self.root_path = path
        self.file_list = list(array)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_cn, label_point, label_point_onehot, points = get_data_v2(path=read_path)
        raw_points_cn = points_cn.copy()

        # centre
        centre = points_cn[:, :3].mean(axis=0)
        points[:, :3] -= centre
        max = points.max()
        points_cn[:, :3] = points_cn[:, :3] / max

        # normalized data
        #maxs = points[:, :3].max(axis=0)
        #mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points_cn[:, 3:].mean(axis=0)
        nstds = points_cn[:, 3:].std(axis=0)

        for i in range(3):
            # normalize coordinate
            points_cn[:, i] = (points_cn[:, i] - means[i]) / stds[i]  # point 1
            # normalize normal vector
            points_cn[:, i + 3] = (points_cn[:, i + 3] - nmeans[i]) / nstds[i]  # normal1

        positive_idx = np.argwhere(label_point > 0)[:, 0]  # tooth idx
        negative_idx = np.argwhere(label_point == 0)[:, 0]  # gingiva idx
        num_positive = len(positive_idx)

        if num_positive > self.patch_size:  # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:  # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive  # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)

        points_cn_selected = points_cn[selected_idx, :]
        label_point_selected = label_point[selected_idx, :]

        return points_cn_selected, label_point_selected, self.file_list[item]


class plydataset_pred(Dataset):
    """
    Input:
        path: root path
        downsample: type of downsample
        ratio: down sample scale
    Return:
        sampled_index_face: [N, 3]
        sampled_point_face: [N, 12]
        sampled_label_face: [N, 1]
    """
    def __init__(self, path="data/train", array=np.zeros([1,1]), patch_size=15999):
        self.root_path = path
        self.file_list = list(array)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_cn, label_point, label_point_onehot, points = get_data_v2(path=read_path)
        raw_points_cn = points_cn.copy()

        # centre
        centre = points_cn[:, :3].mean(axis=0)
        points[:, :3] -= centre
        max = points.max()
        points_cn[:, :3] = points_cn[:, :3] / max

        # normalized data
        #maxs = points[:, :3].max(axis=0)
        #mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points_cn[:, 3:].mean(axis=0)
        nstds = points_cn[:, 3:].std(axis=0)

        for i in range(3):
            # normalize coordinate
            points_cn[:, i] = (points_cn[:, i] - means[i]) / stds[i]  # point 1
            # normalize normal vector
            points_cn[:, i + 3] = (points_cn[:, i + 3] - nmeans[i]) / nstds[i]  # normal1



        return index_face, points_cn, label_point, self.file_list[item], raw_points_cn



if __name__ == "__main__":
    print("ok")







