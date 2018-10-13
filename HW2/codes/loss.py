from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Using your codes in Homework 1'''
        return np.sum(np.power(np.linalg.norm((input - target), axis=1, keepdims=True), 2) / 2, axis=0) / len(input)

    def backward(self, input, target):
        '''Using your codes in Homework 1'''
        return input - target


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        loss = -np.sum(np.multiply(target, np.log(input)))
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        return input - target

