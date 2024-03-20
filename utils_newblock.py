# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F
from dataloader import *
from torch.utils.data import DataLoader





def test(model, loader):
    mean_correct = []
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    return np.mean(mean_correct)

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
def compute_cat_dice(pred,target,dice_tabel):  # pred [B,N,C] target [B,N]
    dice_list = []
    dice_one = []
    target = target.cpu().data.numpy()
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
            dice_tabel[cat,1] += 1
            dice_list.append(dice)
            dice_one.append(np.mean(dice_list))
    return dice_tabel,dice_list, dice_one


def compute_cat_iou(pred,target,iou_tabel):  # pred [B,N,C] target [B,N]
    iou_list = []
    target = target.cpu().data.numpy()
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
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_semseg_newblock(model, loader, num_classes = 15, gpu=True, generate_ply=False):
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
    for batch_id, (points, label_face, name) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        #points_face = raw_points_face[0].numpy()
        #index_face = index[0].numpy()
        coordinate = points.transpose(2,1)
        normal = points[:, :, 12:]
        centre = points[:, :, 9:12]
        label_face = label_face[:, :, 0]

        coordinate, label_face, centre = Variable(coordinate.float()), Variable(label_face.long()), Variable(centre.float())
        coordinate, label_face, centre = coordinate.cuda(), label_face.cuda(), centre.cuda()
        #model.eval()
        with torch.no_grad():
               _, _, _, _, fin_pred, _, _ = model(coordinate)

        dice_tabel, dice_list, dice_one = compute_cat_dice_new(fin_pred, label_face, dice_tabel)
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
    #dice_tabel[:,2] = dice_tabel[:,0] /dice_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['dice'] = np.mean(dice_tabel[:, 2])
    #dice_tabel = pd.DataFrame(dice_tabel,columns=['dice','count','mean_dice'])
    #dice_tabel['Category_dice'] = ["label%d"%(i) for i in range(num_classes)]
    #cat_dice = dice_tabel.groupby('Category_dice')['mean_dice'].mean()
    #dice = np.mean(dice_tabel[:,2])
    output_loss = np.mean(total_loss)

    return metrics, np.mean(dice_array), output_loss

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












if __name__ == "__main__":
    print("ok")




