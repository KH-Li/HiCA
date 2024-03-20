# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:15:46 2022

@author: WORK
"""
import torch
import os
import numpy as np
from dataloader import plydataset,plydataset_pred
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
#from utils import test_semseg, test_semseg_pred
from utils_newblock import test_semseg_newblock, test_semseg_pred_newblock
import torch.nn as nn
import random
#from TSGCNet import TSGCNet
from HiCA import HiCANet
import random
import scipy.io as scio

def get_k_fold_data(k, i, X):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
 
    X_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part = X[idx] 
        if (i == 0) & (j == k-1):
            X_val = X_part
        elif j == i-1:
            X_val = X_part
        elif j == i:  ###第i折作valid
            X_test = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)  # dim=0增加行数，竖着连接
    return X_train, X_test, X_val



#os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
path = r'/home/ssd/likehan/data_dental/ds_ply_all/alldatanewlabel'
file_list = os.listdir(path)
file_array = np.array(file_list)
random.seed(10)
random.shuffle(file_array)
k_fold = 5
batch_size = 2

k = 16

"""--------------------------- create Folder ----------------------------------"""
experiment_dir = Path('./experiment/')
experiment_dir.mkdir(exist_ok=True)
current_time = str(datetime.datetime.now().strftime('%m-%d_%H-%M'))
file_dir = Path(str(experiment_dir) + '/HiCA')
file_dir.mkdir(exist_ok=True)
log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
log_dir.mkdir(exist_ok=True)
checkpoints.mkdir(exist_ok=True)

formatter = logging.Formatter('%(name)s - %(message)s')
logger = logging.getLogger("all")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
writer = SummaryWriter(file_dir.joinpath('tensorboard'))
train_accuracy = []
train_dice = []
val_accuracy = []
val_dice = []
test_accuracy = []
test_dice = []
for i_k in range(k_fold):
    if i_k == 0:
        data_load_path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000_divide/fold0.mat'
    elif i_k == 1:
        data_load_path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000_divide/fold1.mat'
    elif i_k == 2:
        data_load_path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000_divide/fold2.mat'
    elif i_k == 3:
        data_load_path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000_divide/fold3.mat'
    elif i_k == 4:
        data_load_path = r'/home/data/likehan/data_dental/ds_ply_all/alldata15000_divide/fold4.mat'
    #train_set, test_set, val_set = get_k_fold_data(k_fold, i_k, file_array)
    data_loader = scio.loadmat(data_load_path)
    train_set = data_loader['train']
    test_set = data_loader['test']
    val_set = data_loader['val']
    
    train_dataset = plydataset(path, train_set)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    train_val_dataset = plydataset(path, train_set)
    train_val_loader = DataLoader(train_val_dataset, batch_size=1, shuffle=False, num_workers=64)
    val_dataset = plydataset(path, val_set)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=64)

    """--------------------------- Build Network and optimizer----------------------"""
    model = HiCANet(in_channels=3, output_channels=15, k=k)
    #model = TSGCNet(in_channels=3, output_channels=15, k=k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    """------------------------------------- train --------------------------------"""
    logger.info("------------------train------------------")
    best_acc = 0
    best_dice = 0
    LEARNING_RATE_CLIP = 1e-5
    his_loss = []
    his_smotth = []
    class_weights = torch.ones(15).cuda()
   
    for epoch in range(0, 200):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        optimizer.param_groups[0]['lr'] = lr
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            points_face, label_face, name = data
            coordinate = points_face.transpose(2,1)

            coordinate, label_face = Variable(coordinate.float()), Variable(label_face.long())
            coordinate, label_face = coordinate.cuda(), label_face.cuda()
            optimizer.zero_grad()
            pred, feature_map, coor_feature, nor_feature, fin_pred, N_to_N, N_C_output = model(coordinate)
            #feature_map = feature_map.permute(0, 2, 1).contiguous().view(-1, 128)
            #feature_map = feature_map[torch.randperm(len(feature_map))]
            #U, S_score, V = torch.svd(feature_map[0:6000])
            #loss_lowrank_score = S_score[15]
            '''
            coor_feature = coor_feature.permute(0, 2, 1).contiguous().view(-1, 512)
            coor_feature = coor_feature[torch.randperm(len(coor_feature))]
            U, S_coor, V = torch.svd(coor_feature[0:6000])
            loss_lowrank_coor = S_coor[15]

            nor_feature = nor_feature.permute(0, 2, 1).contiguous().view(-1, 512)
            nor_feature = nor_feature[torch.randperm(len(nor_feature))]
            U, S_nor, V = torch.svd(nor_feature[0:6000])
            loss_lowrank_nor = S_nor[15]
            '''
            label_face = label_face.view(-1, 1)[:, 0]
            pred = pred.contiguous().view(-1, 15)
            constrain_loss = F.nll_loss(pred, label_face)
            #total_loss = loss
            '''-------------N_C_output--------'''
            N_C_output = N_C_output.contiguous().view(-1, 15)
            N_C_loss = F.nll_loss(N_C_output, label_face)

            '''-------------newblock----------'''
            fin_pred = fin_pred.contiguous().view(-1, 15)
            fin_loss = F.nll_loss(fin_pred, label_face)
            total_loss = N_C_loss + fin_loss
            '''-------------newblock----------'''
            #total_loss = loss + loss_lowrank_score*0.008

            total_loss.backward()
            optimizer.step()
            his_loss.append(total_loss.cpu().data.numpy())

        if epoch % 5 == 0:
            print('total_fold =', k_fold)
            print('k_fold = ', i_k)
            print('Learning rate: %f' % (lr))
            print("loss: %f" % (np.mean(his_loss)))
            writer.add_scalar("loss", np.mean(his_loss), epoch)
            metrics, dice, val_loss = test_semseg_newblock(model, val_loader, num_classes=15)
            metrics_train, dice_train, train_loss = test_semseg_newblock(model, train_val_loader, num_classes=15)
            print("Epoch %d, accuracy= %f, dice= %f, val_loss= %f " % (epoch, metrics['accuracy'], dice, val_loss))
            print("train_accuracy= %f, train_dice= %f, train_loss= %f" % (metrics_train['accuracy'], dice_train, train_loss))
            logger.info("Epoch: %d, accuracy= %f, dice= %f loss= %f" % (epoch, metrics['accuracy'], dice, np.mean(his_loss)))
            logger.info("train_accracy= %f, train_dice= %f"% (metrics_train['accuracy'], dice_train))
            writer.add_scalar("accuracy", metrics['accuracy'], epoch)
        
            if (dice > best_dice):
                best_acc = metrics['accuracy']
                best_epoch = epoch
                best_dice = dice
                print("best accuracy: %f best dice :%f" % (best_acc, dice))
                print('this model is better')
                torch.save(model.state_dict(), '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc))
                best_pth = '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc)
                #logger.info(cat_iou)
            his_loss.clear()
            writer.close()

    root_path = r'./experiment/HiCA/checkpoints'
    doc_name = 'coordinate_%d_%f.pth' % (best_epoch, best_acc)
    model_path = os.path.join(root_path, doc_name)
    model = HiCANet(in_channels=3, output_channels=15, k=16)
    checkpoints_model = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoints_model.items()})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_accuracy.append(metrics_train['accuracy'])
    train_dice.append(dice_train)
    val_accuracy.append(best_acc)
    val_dice.append(best_dice)
    test_dataset = plydataset_pred(path, test_set)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=64)
    metrics_test, dice_test, test_loss = test_semseg_pred_newblock(model, test_loader, num_classes=15, generate_ply=False)
    test_accuracy.append(metrics_test['accuracy'])
    test_dice.append(dice_test)
    logger.info('{} fold done, total fold = {}'.format(i_k,k_fold))
    logger.info('train_accuracy={}, train_dice={}, val_accuracy={}, val_dice={}, test_accuracy={}, test_dice={}'.format(metrics_train['accuracy'], dice_train, best_acc, best_dice, metrics_test['accuracy'], dice_test))
    print('{} fold done, total fold = {}'.format(i_k,k_fold))
    print('train_accuracy=', metrics_train['accuracy'])
    print('train_dice=', dice_train)
    print('val_accuracy=', best_acc)
    print('val_dice=', best_dice)
    print('test_accuracy=', metrics_test['accuracy'])
    print('test_dice=', dice_test)
experiment_dir_test = Path('./result/')
experiment_dir_test.mkdir(exist_ok=True)
formatter = logging.Formatter('%(name)s - %(message)s')
logger = logging.getLogger("all")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_dir) + '/result.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


train_accuracy = np.mean(train_accuracy) 
train_dice = np.mean(train_dice)
val_accuracy = np.mean(val_accuracy)
val_dice = np.mean(val_dice)
test_accuracy = np.mean(test_accuracy)
test_dice = np.mean(test_dice)
print('train_accuracy_average={}, train_dice_average={}'.format(train_accuracy, train_dice))
print('val_accuracy_average={}, val_dice={}'.format(val_accuracy, val_dice))
print('test_accuracy_average={}, test_dice={}'.format(test_accuracy, test_dice))

logger.info("------------------result------------------")
logger.info("train_accuracy= %f, train_dice= %f" % (train_accuracy, train_dice))
logger.info("val_accuracy= %f, val_dice= %f" % (val_accuracy, val_dice))
logger.info("test_accuracy= %f, test_dice= %f" % (test_accuracy, test_dice))
#logger.info(cat_iou)