from __future__ import division

import numpy as np
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

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

    def forward(self, input, target,epsilon=1e-12):
        '''Your codes here'''
        (n, _) = input.shape
        for i in range(n):
            input[i]=softmax(input[i])
        self.input = np.clip(input, epsilon, 1. - epsilon)
        loss = -np.mean(np.sum(target * np.log(self.input + 1e-9),axis=1))
        return loss

    def backward(self, input, target):
        '''Your codes here'''

        return (self.input - target)/input.size

