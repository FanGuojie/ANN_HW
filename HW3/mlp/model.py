# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.h_units=300
        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train=True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)




    def forward(self, is_train=True, reuse=tf.AUTO_REUSE):

        with tf.variable_scope("model", reuse=reuse) as scope:
            # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer
            # Your BN Layer: use batch_normalization_layer function
            # Your Relu Layer
            # Your Dropout Layer: use dropout_layer function
            # Your Linear Layer

            W1 = tf.get_variable(name="W1",initializer=tf.truncated_normal([784, 300], stddev=0.1))
            b1 = tf.get_variable(initializer=tf.zeros([300]), name="b1")
            W2 = tf.get_variable(initializer=tf.truncated_normal([300, 10], stddev=0.1), name="W2")
            b2 = tf.get_variable(initializer=tf.zeros([10]), name="b2")
            # scope.reuse_variables()
            u1=tf.matmul(self.x_, W1) + b1
            # mu1,sigma1=batch_normalization_layer(u1,is_train=is_train)
            # scale1=tf.get_variable(initializer=tf.ones([1]),name="scale1")
            # shift1=tf.get_variable(initializer=tf.zeros([1]),name="shift1")
            # epsilon=0.001
            o1=batch_norm_layer(u1,is_training=is_train)
            # o1=tf.nn.batch_normalization(u1,mu1,sigma1,shift1,scale1,epsilon)
            # o1=tf.layers.batch_normalization(u1,training=is_train)
            # is_training=tf.constant(is_train, dtype=tf.bool)
            # o1=tf.cond(is_training,
            #         lambda: batch_norm(u1, is_training=True,
            #                            center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope="model"),
            #         lambda: batch_norm(u1, is_training=False,
            #                            center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
            #                            scope="model", reuse=True))
            hidden1 = tf.nn.relu(o1)
            h1_drop = dropout_layer(hidden1, self.keep_prob, is_train=is_train)
            logits = tf.nn.softmax(tf.matmul(h1_drop, W2) + b2)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_norm_layer(value, is_training=False, name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果

    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training is True:
        # 训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value, decay=0.9, updates_collections=None, is_training=True)
    else:
        # 测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value, decay=0.9, updates_collections=None, is_training=False)

def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    mu, sigma = tf.nn.moments(incoming, axes=[0])
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([mu, sigma])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mu), tf.identity(sigma)

    mu, sigma = tf.cond(tf.constant(is_train, dtype=tf.bool), mean_var_with_update,
                        lambda: (ema.average(mu), ema.average(sigma)))

    return mu, sigma
    
def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    if is_train:
        return tf.nn.dropout(incoming, drop_rate)
    else:
        return incoming


