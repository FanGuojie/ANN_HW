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
        self.times = tf.placeholder(tf.int32)
        self.epoch = tf.placeholder(tf.int32)
        self.iter = self.epoch * FLAGS.batch_size + self.times
        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train=True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()


        # TODO:  maybe you need to update the parameter of batch_normalization?
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=tf.AUTO_REUSE):

        with tf.variable_scope("model", reuse=reuse):
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            K = 24  # first convolutional layer output depth
            L = 48  # second convolutional layer output depth
            M = 64  # third convolutional layer
            N = 200  # fully connected layer
            tst=tf.constant(not is_train,dtype=tf.bool)
            W1 = tf.get_variable(name="W1",
                initializer=tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
            B1 = tf.get_variable(name="b1",initializer=tf.constant(0.1, tf.float32, [K]))
            W2 = tf.get_variable(name="W2",initializer=tf.truncated_normal([5, 5, K, L], stddev=0.1))
            B2 = tf.get_variable(name="b2",initializer=tf.constant(0.1, tf.float32, [L]))
            W3 = tf.get_variable(name="W3",initializer=tf.truncated_normal([4, 4, L, M], stddev=0.1))
            B3 = tf.get_variable(name="b3",initializer=tf.constant(0.1, tf.float32, [M]))

            W4 = tf.get_variable(name="W4",initializer=tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
            B4 = tf.get_variable(name="b4",initializer=tf.constant(0.1, tf.float32, [N]))
            W5 = tf.get_variable(name="W5",initializer=tf.truncated_normal([N, 10], stddev=0.1))
            B5 = tf.get_variable(name="b5",initializer=tf.constant(0.1, tf.float32, [10]))

            # The model
            # batch norm scaling is not useful with relus
            # batch norm offsets are used instead of biases
            stride = 1  # output is 28x28
            Y1l = tf.nn.conv2d(self.x_, W1, strides=[1, stride, stride, 1], padding='SAME')
            Y1bn, update_ema1 = batchnorm(Y1l, tst,  B1,self.iter, convolutional=True)
            Y1r = tf.nn.relu(Y1bn)
            Y1 = tf.nn.dropout(Y1r, FLAGS.keep_prob, compatible_convolutional_noise_shape(Y1r))
            stride = 2  # output is 14x14
            Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
            Y2bn, update_ema2 = batchnorm(Y2l, tst,  B2,self.iter, convolutional=True)
            Y2r = tf.nn.relu(Y2bn)
            Y2 = tf.nn.dropout(Y2r, FLAGS.keep_prob, compatible_convolutional_noise_shape(Y2r))
            stride = 2  # output is 7x7
            Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
            Y3bn, update_ema3 = batchnorm(Y3l, tst,  B3, self.iter,convolutional=True)
            Y3r = tf.nn.relu(Y3bn)
            Y3 = tf.nn.dropout(Y3r, FLAGS.keep_prob, compatible_convolutional_noise_shape(Y3r))

            # reshape the output from the third convolution for the fully connected layer
            YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

            Y4l = tf.matmul(YY, W4)
            Y4bn, update_ema4 = batchnorm(Y4l, tst,  B4,self.iter)
            Y4r = tf.nn.relu(Y4bn)
            Y4 = tf.nn.dropout(Y4r,FLAGS.keep_prob)
            Ylogits = tf.matmul(Y4, W5) + B5
            logits = tf.nn.softmax(Ylogits)

            update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)





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

def batchnorm(Ylogits, is_test,  offset,iter, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999,iter) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    pass
    
def dropout_layer(incoming, keep_prob, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    if is_train:
        return tf.nn.dropout(x=incoming, keep_prob=keep_prob)
    else:
        return incoming
