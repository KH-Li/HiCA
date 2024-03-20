import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
#from torch import linalg as LA
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
    return idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def get_graph_feature1(x, k=10, flag=False):
    batch_size, num_dims, num_points  = x.shape
    x = x.view(batch_size, -1, num_points)
    if flag:
        idx = knn(x[:,9:12,:], k=k)  # (batch_size, num_points, k)
    else:
        idx = knn(x, k=k)

    device = torch.device('cuda')
    index = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    #feature = index_points(x.transpose(2,1), idx)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor, k=10, idx=None):
    batch_size, num_dims, num_points  = coor.shape
    coor = coor.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(coor, k=k)

    index = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()

    coor = coor.transpose(2,1).contiguous()
    nor = nor.transpose(2,1).contiguous()

    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature-coor, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature-nor, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    return coor_feature, nor_feature, index


class GraphAttention(nn.Module):
    def __init__(self,feature_dim,out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.K=K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0,2,3,1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0,3,2,1)
        e = self.conv(delta_f)
        e = e.permute(0,3,2,1)
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature),dim = 2) .permute(0,2,1)
        return graph_feature

class NonLocalBlock(nn.Module):
    def __init__(self,in_channels, out_channels, k):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.g = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,kernel_size=1,
                           stride=1, padding=0)

        self.theta_x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,kernel_size=1,
                                 stride=1, padding=0)
        #self.phi_x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1,
        #                       stride=1, padding=0)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.W = nn.Sequential( nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels,kernel_size=1,
                           stride=1, padding=0),
                                self.bn)
    def forward(self, graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        neighbor_feature = index_points(x, graph_index)#tensor.size[1,16000,16,12]
        centre = x.view(B, N, 1, C)
        centre = centre.permute(0, 3 ,2, 1)
        centre = self.theta_x(centre).permute(0, 3, 2, 1)
        #neighbor_feature = torch.cat((neighbor_feature,centre),dim=2)#拼接自身特征变成17维
        phi_x = neighbor_feature.permute(0, 3, 2, 1)
        phi_x = self.theta_x(phi_x).permute(0, 3, 1,2)

        theta_x = centre
        #print('theta=', theta_x.shape)
        #print('phi=', phi_x.shape)
        multiply_mid = torch.matmul(theta_x, phi_x)
        coefficient = F.softmax(multiply_mid,dim=3)
        #feature = neighbor_feature.permute(0, 2, 3, 1)
        feature = self.g(feature)
        g_x = feature.permute(0, 2, 3, 1)
        output = torch.matmul(coefficient,g_x)
        output = torch.sum(output, dim=2).permute(0, 2, 1)
        z = self.W(output)
        return z

class att_module(nn.Module):
    def __init__(self, in1_ch, in2_ch, out_ch):
        super(att_module, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv1d(in_channels=in1_ch + in2_ch, out_channels=in1_ch + in2_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(in1_ch + in2_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in1_ch + in2_ch, out_channels=in1_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(in1_ch),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in1_ch, out_channels=out_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self, x1, x2):
        y = torch.cat([x1, x2], dim=1)
        att_mask = self.att_conv(y)
        x1 = att_mask * x1
        x1 = self.conv(x1)
        return x1

'''
TSGCNet
'''
class HiCANet(nn.Module):
    def __init__(self, k=16, in_channels=12, output_channels=15, return_feature=False, name=None):
        super(HiCANet, self).__init__()
        self.name = name
        self.k = k
        self.return_feature = return_feature
        ''' coordinate stream '''
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(64)
        self.bn3_c = nn.BatchNorm2d(128)
        self.bn3_c_ms = nn.BatchNorm2d(128)
        self.bn4_c = nn.BatchNorm2d(256)
        self.bn5_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv2_c = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.conv3_c = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_c_ms = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                        self.bn3_c_ms,
                                        nn.LeakyReLU(negative_slope=0.2))
        self.bn3_c_cat = nn.BatchNorm1d(128)
        self.conv3_c_cat = nn.Sequential(nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False),
                                         self.bn3_c_cat,
                                         nn.LeakyReLU(negative_slope=0.2))


        self.conv5_c = nn.Sequential(nn.Conv1d(384, 512, kernel_size=1, bias=False),
                                     self.bn5_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.attention_layer1_c = GraphAttention(feature_dim=3, out_dim=64, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=64, out_dim=64, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=64, out_dim=128, K=32)
        self.attention_layer3_c_ms = GraphAttention(feature_dim=64, out_dim=128, K=32)

        #self.nonlocalblock_layer1_n = NonLocalBlock(in_channels=3, out_channels=64, k=self.k)
        #self.nonlocalblock_layer2_n = NonLocalBlock(in_channels=64, out_channels=64, k=self.k)
        #self.nonlocalblock_layer3_n = NonLocalBlock(in_channels=64, out_channels=128, k=32)

        self.FTM_c1 = STNkd(k=3)

        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(64)
        self.bn3_n = nn.BatchNorm2d(128)
        self.bn3_n_ms = nn.BatchNorm2d(128)
        self.bn4_n = nn.BatchNorm2d(256)
        self.bn5_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(nn.Conv2d((in_channels)*2, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv2_n = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv3_n = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv3_n_ms = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                        self.bn3_n_ms,
                                        nn.LeakyReLU(negative_slope=0.2))
        self.bn3_n_cat = nn.BatchNorm1d(128)
        self.conv3_n_cat = nn.Sequential(nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False),
                                         self.bn3_n_cat,
                                         nn.LeakyReLU(negative_slope=0.2))


        self.conv5_n = nn.Sequential(nn.Conv1d(384, 512, kernel_size=1, bias=False),
                                     self.bn5_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.FTM_n1 = STNkd(k=3)

        self.AM_coor3 = att_module(128, 128, 256)
        self.AM_nor3 = att_module(128, 128, 256)

        ''' feature fusion1 '''
        self.pred1 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)

        '''new block'''
        self.bn_hidden = nn.BatchNorm1d(128)
        self.hidden_block2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                           self.bn_hidden,
                                           nn.LeakyReLU(negative_slope=0.2))
        self.bn_class = nn.BatchNorm1d(15)
        self.class_block2 = nn.Sequential(nn.Conv1d(128, 15, kernel_size=1, bias=False),
                                          self.bn_class,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.bn_c_block2 = nn.BatchNorm2d(512)
        self.bn_n_block2 = nn.BatchNorm2d(512)
        self.bn_c_cat_block2 = nn.BatchNorm1d(512)
        self.bn_n_cat_block2 = nn.BatchNorm1d(512)
        self.conv_block_c_block2 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
                                                 self.bn_c_block2,
                                                 nn.LeakyReLU(negative_slope=0.2))
        self.conv_block_n_block2 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
                                                 self.bn_n_block2,
                                                 nn.LeakyReLU(negative_slope=0.2))
        self.attention_layer_block_c_block2 = GraphAttention(feature_dim=512, out_dim=512, K=16)
        self.nonlocalblock_layer_block_n_block2 = NonLocalBlock(in_channels=512, out_channels=512, k=16)
        self.conv_cat_c_block2 = nn.Sequential(nn.Conv1d(256 * 2, 512, kernel_size=1, bias=False),
                                               self.bn_c_cat_block2,
                                               nn.LeakyReLU(negative_slope=0.2))
        self.conv_cat_n_block2 = nn.Sequential(nn.Conv1d(256 * 2, 512, kernel_size=1, bias=False),
                                               self.bn_n_cat_block2,
                                               nn.LeakyReLU(negative_slope=0.2))
        '''--------'''
        self.pred_fin1_block2 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                              nn.BatchNorm1d(512),
                                              nn.LeakyReLU(negative_slope=0.2))
        self.pred_fin2_block2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                              nn.BatchNorm1d(256),
                                              nn.LeakyReLU(negative_slope=0.2))
        self.pred_fin3_block2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                              nn.BatchNorm1d(128),
                                              nn.LeakyReLU(negative_slope=0.2))
        self.pred_fin4_block2 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        self.dp_fin1_block2 = nn.Dropout(p=0.6)
        self.dp_fin2_block2 = nn.Dropout(p=0.6)
        self.dp_fin3_block2 = nn.Dropout(p=0.6)

    def forward(self, x):
        batch_size = x.size(0)
        coor = x[:, :3, :]
        #coordinate_space
        centre = x[:, :3, :]
        idx = knn(centre, k=self.k)
        idx1 = knn(centre, k=32)

        fea = coor
        nor = x[:, 3:, :]

        # transform
        trans_c = self.FTM_c1(coor)
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        # layer1
        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k, idx=idx)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]
        #nor1 = self.nonlocalblock_layer1_n(index, nor, nor1)

        # layer2
        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k, idx=idx)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]
        #nor2 = self.nonlocalblock_layer2_n(index, nor1, nor2)

        # layer3
        coor3, nor3, index = get_graph_feature(coor2, nor2, k=32, idx=idx1)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]
        #nor3_orig = self.nonlocalblock_layer3_n(index, nor2, nor3)
        # layer3 multiscale
        coor3_ms, nor3_ms, index = get_graph_feature(coor2, nor2, k=32)
        coor3_ms = self.conv3_c_ms(coor3_ms)
        nor3_ms = self.conv3_n_ms(nor3_ms)
        coor3_ms = self.attention_layer3_c_ms(index, coor2, coor3_ms)
        # nor3_ms = self.nonlocalblock_layer3_n_ms(index, nor2, nor3_ms)
        nor3_ms = nor3_ms.max(dim=-1, keepdim=False)[0]
        # concatenation
        coor3_orig = torch.cat((coor3, coor3_ms), dim=1)
        coor3_orig = self.conv3_c_cat(coor3_orig)
        nor3_orig = torch.cat((nor3, nor3_ms), dim=1)
        nor3_orig = self.conv3_n_cat(nor3_orig)

        #AMlayer
        coor3 = self.AM_coor3(coor3_orig, nor3_orig)
        nor3 = self.AM_nor3(nor3_orig, coor3_orig)

        # feature fusion
        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor_cat = self.conv5_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor_cat = self.conv5_n(nor)
        x = torch.cat((coor_cat, nor_cat), dim=1)


        # prediction
        x = self.pred1(x)
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x_score = self.pred3(x)
        self.dp3(x)
        score1 = self.pred4(x_score)
        score = F.log_softmax(score1, dim=1)
        score_output = score.permute(0, 2, 1)

        """----------------------------------------new block-------------------------------------------------------"""

        x_hidden_use = self.hidden_block2(x_score)
        # print(x_hidden_use)
        x_class = self.class_block2(x_score)
        x_hidden_use_row = F.normalize(x_hidden_use, dim=2, p=2)

        x_class = F.normalize(x_class, dim=2, p=2)
        #x_class = score1
        x_hidden = x_hidden_use_row.permute(0, 2, 1)
        prototype = torch.matmul(x_class, x_hidden)
        prototype = F.normalize(prototype, dim=2, p=2)
        x_hidden_use_line = F.normalize(x_hidden_use, dim=1, p=2)
        N_to_class = torch.matmul(prototype, x_hidden_use_line)
        # N_to_class = score1
        N_C = F.log_softmax(N_to_class, dim=1)
        N_to_class = F.normalize(N_to_class, dim=1, p=2)
        # test = LA.norm(N_to_class, dim=1)
        N_C_output = N_C.permute(0, 2, 1)
        N_to_class_T = N_to_class.permute(0, 2, 1)
        N_to_N = torch.matmul(N_to_class_T, N_to_class)
        KNN_idx = N_to_N.topk(k=16 + 1, dim=-1)[1][:, :, 1:]

        coor_block, nor_block, index = get_graph_feature(coor_cat, nor_cat, k=16, idx=KNN_idx)
        coor_block = self.conv_block_c_block2(coor_block)
        # print(coor_block.shape)
        nor_block = self.conv_block_n_block2(nor_block)
        coor_block = self.attention_layer_block_c_block2(index, coor_cat, coor_block)
        #nor_block = self.nonlocalblock_layer_block_n_block2(index, nor_cat, nor_block)
        nor_block = nor_block.max(dim=-1, keepdim=False)[0]

        coor_concat = self.conv_cat_c_block2(coor_block)
        nor_concat = self.conv_cat_n_block2(nor_block)
        z = torch.cat((coor_concat, nor_concat), dim=1)

        z = self.pred_fin1_block2(z)
        self.dp_fin1_block2(x)
        z = self.pred_fin2_block2(z)
        self.dp_fin2_block2(z)
        z = self.pred_fin3_block2(z)
        self.dp_fin3_block2(z)
        z_score = self.pred_fin4_block2(z)
        z_score = F.log_softmax(z_score, dim=1)
        z_score = z_score.permute(0, 2, 1)
        return score_output, x_score, coor_cat, nor_cat, z_score, N_to_N, N_C_output





if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input size: [batch_size, C, N], where C is number of dimension, N is the number of mesh.
    x = torch.rand(1,6,3000)
    x = x.cuda()
    model = HiCANet(in_channels=3, output_channels=15, k=16)
    model = model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    y = model(x)
    print("ok")
