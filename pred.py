# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:20:12 2021

@author: WORK
"""
import torch
from torch.utils.data import DataLoader
from torch.nn.modules import module
#import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from torch.autograd import Variable
from dataloader import *
from dataloader import plydataset
from HiCA import HiCANet
import logging
from pathlib import Path
import scipy.io as scio
filepath = r'/home/data/likehan/code/HiCA_test/experiment/HiCA/checkpoints/coordinate_95_0.948622.pth'
#net = models.squeezenet1_1(pretrained=False)
model = HiCANet(in_channels=3, output_channels=15, k=16)
checkpoints = torch.load(filepath)
model.load_state_dict({k.replace('module.',''): v for k,v in checkpoints.items()})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
#model=torch.load(filepath)
def get_data_v2(path=""):
    labels = ((255, 255, 255), (255, 134, 55), (255, 125, 72),(34, 0, 34), (255, 255, 0), (0, 255, 0),
          (255, 0, 0), (0, 0, 0), (125, 125, 125), (0, 125, 125), (125, 0, 125), (0, 0, 125), (255, 255, 1), (88, 125, 88),
           (125, 0, 0))
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


def compute_cat_dice_new(pred,target,dice_tabel):  # pred [B,N,C] target [B,N]
    dice_list = []
    dice_one = []
    target = target.cpu().data.numpy()
    dice_tabel_one = np.zeros((15, 1))
    for j in range(pred.size(0)):
        batch_pred = pred[j]  # batch_pred [N,C]
        batch_target = target[j]  # batch_target [N]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()  # index of max value  batch_choice [N]
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                dice = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                dice = (I+I) / float(U+I)
            dice_tabel[cat,0] += dice
            #dice_tabel_one[cat, 0] = dice
            dice_tabel[cat,1] += 1
            dice_list.append(dice)
        #dice_one = np.mean(dice_list)
        dice_one.append(np.mean(dice_list))
    return dice_tabel,dice_list, dice_one

def test_semseg_pred_newblock(model, loader, num_classes = 15, gpu=True, generate_ply=False):
    '''
    Input
    :param model:
    :param loader:
    :param num_classes:
    :param pointnet2:
    Output
    metrics: metrics['accuracy']-> overall accuracy
             metrics['iou']-> mean Iou
    hist_acc: history of accuracy
    cat_iou: IoU for o category
    '''
    dice_tabel = np.zeros((num_classes,3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    total_loss = []
    dice_array = []
    for batch_id, (index ,points, label_face, name, raw_points_face) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points_face = raw_points_face[0].numpy()
        index_face = index[0].numpy()
        coordinate = points.transpose(2,1)
        normal = points[:, :, 12:]
        centre = points[:, :, 9:12]
        label_face = label_face[:, :, 0]

        coordinate, label_face, centre = Variable(coordinate.float()), Variable(label_face.long()), Variable(centre.float())
        coordinate, label_face, centre = coordinate.cuda(), label_face.cuda(), centre.cuda()
        #model.eval()
        with torch.no_grad():
               _, _, _, _, fin_pred, _, _ = model(coordinate)

        dice_tabel, dice_list, dice_one = compute_cat_dice_new(fin_pred,label_face,dice_tabel)
        dice_array.append(dice_one)
        fin_pred = fin_pred.contiguous().view(-1, num_classes)
        label_face = label_face.view(-1, 1)[:, 0]
        loss = F.nll_loss(fin_pred, label_face)
        total_loss.append(loss.cpu().data.numpy())
        pred_choice = fin_pred.data.max(1)[1]
        correct = pred_choice.eq(label_face.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
        label_face = pred_choice.cpu().reshape(pred_choice.shape[0], 1)
        if generate_ply:
               #label_face=label_optimization(index_face, label_face)
               generate_plyfile(index_face, points_face, label_face, path=("pred/%s") % name)
    dice_tabel[:,2] = dice_tabel[:,0] /dice_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['dice'] = np.mean(dice_tabel[:, 2])
    #iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    #iou_tabel['Category_IOU'] = ["label%d"%(i) for i in range(num_classes)]
    #cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    dice = np.mean(dice_tabel[:,2])
    output_loss = np.mean(total_loss)

    return metrics, np.mean(dice_array), output_loss


if __name__=='__main__':
    experiment_dir = Path('./prediction_metrics/')
    experiment_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('metrics/')
    log_dir.mkdir(exist_ok=True)
    formatter = logging.Formatter('%(name)s - %(message)s')
    logger = logging.getLogger("all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_dir) + '/metrics.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000'
    data_load_path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000_divide/fold4.mat'
    data_loader = scio.loadmat(data_load_path)
    train_set = data_loader['train']
    test_set = data_loader['test']
    val_set = data_loader['val']
    test_dataset = plydataset_pred(path, test_set)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=64)
    metrics, DICE, outout_loss = test_semseg_pred_newblock(model, test_loader, num_classes=15, generate_ply=False)
    logger.info("------------------pred------------------")
    logger.info("accuracy= %f, mIoU= %f" % (metrics['accuracy'], DICE))
    logger.info(outout_loss)