#!/usr/bin/python3
# -*- coding:utf-8 -*-
# project:
# user:ubuntu
# Author: Sean
# createtime: 11/15/20 21:18 AM
import os
import torch
import h5py as h5
from autoEncoder.data import TrainDataset
from autoEncoder.data.getData import getData
from autoEncoder.model import AutoEncoderMlp
from autoEncoder.model.getLoss import getOptimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# ***************************初始化一些函数********************************
# torch.cuda.set_device(gpu_id)#使用GPU


# *************************************数据集的设置****************************************************************************
use_gpu = False
# hyperparameter
modelName = 'autoEncoder_L1'
dataset = 'transe_FB15K237'
orgin_dim = 128
aim_dim = 64
opt_method = 'Adam'
learn_rate = 0.001
epochs  = 1000
batch_size = 3000
modelInfoName = modelName+\
                '_oDim'+str(orgin_dim)+'_aDim'+str(aim_dim)+\
                '_opt'+ opt_method +'_alpha'+ str(learn_rate)+ '_e'+ str(epochs)+'_dataset'+dataset
loss_fuc = nn.L1Loss(reduction='sum')


# dataloader for training
train_dataset = TrainDataset('transe_FB15K237','ent')
print('size of trainset',len(train_dataset))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,num_workers=4)

# dataloader for test

# !!!!!!!!!!!test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
autoEncoderMlp = AutoEncoderMlp(orgin_dim, aim_dim)

# print the nums of parameters
total_params = sum(p.numel() for p in autoEncoderMlp.parameters())
print('{:,} total parameters.'.format(total_params))
total_trainable_params = sum(
    p.numel() for p in autoEncoderMlp.parameters() if p.requires_grad)
print('{:,} training parameters.'.format(total_trainable_params))

#define optimizer
# optimizer = getOptimizer(autoEncoderMlp,opt_method,learn_rate)
optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, autoEncoderMlp.parameters()),
            lr=learn_rate,
            # weight_decay=weight_decay,
        )
# train the model
for epoch in range(epochs):
    trainloss = 0
    for step,data in enumerate(train_dataloader):
        optimizer.zero_grad()  # clear gradients for this training step
        encoded, decoded = autoEncoderMlp(data)
        loss = loss_fuc(decoded, data)  # mean square error
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        trainloss += loss.data.numpy()
        # if step  == 5:
        #     print('__Epoch: ', 5, '| train loss: %.4f' % loss.data.numpy())
    if epoch % 5 == 0:
        print('Epoch: ', epoch, '| train loss: %.4f' % trainloss)

# define the loss function

autoEncoderMlp.save_parameters(os.getcwd()+'/autoEncoder/savedparameters/'+modelInfoName+'.json')


#__________________________________test__________________________________________________
autoEncoderMlp.load_parameters(os.getcwd()+'/autoEncoder/savedparameters/'+modelInfoName+'.json')
autoEncoderMlp.eval()
data = getData('/home/george_sun/code/AE_extraFeature/autoEncoder/data/transe_FB15K237', 'ent')
# !! 对数据集进行归一化处理(可以尝试不同的归一化方法）
min = data.min()
max = data.max()
data = (data - min) / (max - min)
# # definded the tester
# tester = Tester(model = transConvencode_cvT2d_3_simplify, data_loader = test_dataloader, use_gpu = use_gpu)
i = 12
fea, x_ = autoEncoderMlp(data[i])
print(data[i])
print(x_)
print(loss_fuc(x_, data[i]))

# # preocess loss info
# print('loss_history')
# print(loss_history)
#
# transConvencode_cvT2d_3_simplify.save_checkpoint('./checkpoint/'+ modelInfoName+'.ckpt')
# with h5.File('result/historyLoss/'+modelInfoName + 'hisitoryLoss' + '.h5', 'w') as h5f:
# 	h5f.create_dataset('hisitoryLoss', data=loss_history)
#
#
# lFig.figLoss('result/historyLoss/'+modelInfoName + 'hisitoryLoss' + '.h5',modelInfoName,showImg = False)
#
# # test the model
# transConvencode_cvT2d_3_simplify.load_checkpoint('./checkpoint/'+ modelInfoName+'.ckpt')
