#!/usr/bin/python3
# -*- coding:utf-8 -*-
# project:
# user:ubuntu
# Author: Sean
# createtime: 11/15/20 21:53 AM
import torch.optim as optim

def getOptimizer(model, opt_method, alpha):
    optimizer = None
    if opt_method == "Adagrad" or opt_method == "adagrad":
        optimizer = optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=alpha,
            # lr_decay=lr_decay,
            # weight_decay=weight_decay,
        )
    elif opt_method == "Adadelta" or opt_method == "adadelta":
        optimizer = optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=alpha,
            # weight_decay=weight_decay,
        )
    elif opt_method == "Adam" or opt_method == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=alpha,
            # weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=alpha,
            # weight_decay=weight_decay,
        )
    return optimizer