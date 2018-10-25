# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.nn import  conv2d as conv
from tensorflow.nn import  max_pool as max_pool
FLAGS = tf.app.flags.FLAGS

def w_variable(shape,name):
    ini=tf.truncated_normal(shape,stddev=0.1)
    return tf.get_variable(initializer=ini,name=name)

def b_variable(shape,name):
    ini=tf.constant(0.1,shape=shape)
    return tf.get_variable(initializer=ini,name=name)

def conv2d(x,W):
    return conv(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

class Model:
    def __init__(self,
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train=True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=tf.AUTO_REUSE):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Conv Layer
            W_conv1=w_variable([5,5,1,32],"W_conv1")
            b_conv1=b_variable([32],"b_conv1")
            h_conv1=conv2d(self.x_,W_conv1)+b_conv1
            h_relu1=tf.nn.relu(h_conv1)
            h_pool1=max_pool_2x2(h_relu1)

            W_conv2=w_variable([5,5,32,64],"W_conv2")
            b_conv2=b_variable([64],"b_conv2")
            h_conv2=conv2d(h_pool1,W_conv2)+b_conv2
            h_relu2=tf.nn.relu(h_conv2)
            h_pool2=max_pool_2x2(h_relu2)
            h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
            W_fc=w_variable([7*7*64,10],name="W_fc")
            b_fc=b_variable([10],name="b_fc")
            logits=tf.nn.softmax(tf.matmul(h_pool2_flat,W_fc)+b_fc)

            # Your BN Layer: use batch_normalization_layer function
            # Your Relu Layer
            # Your Dropout Layer: use dropout_layer function
            # Your MaxPool
            # Your Conv Layer
            # Your BN Layer: use batch_normalization_layer function
            # Your Relu Layer
            # Your Dropout Layer: use dropout_layer function
            # Your MaxPool
            # Your Linear Layer




        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    pass
    
def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    pass
