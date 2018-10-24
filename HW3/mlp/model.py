# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self,
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28],name="x")
        self.y_ = tf.placeholder(tf.int32, [None],name="y_")
        self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")
        self.h_units=300
        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train=True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
                                    
    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer
            # Your BN Layer: use batch_normalization_layer function
            # Your Relu Layer
            # Your Dropout Layer: use dropout_layer function
            # Your Linear Layer
            W1=tf.Variable(tf.truncated_normal([784,self.h_units],stddev=0.1),name="W1")
            b1=tf.Variable(tf.zeros([self.h_units]),name="b1")
            W2 = tf.Variable(tf.truncated_normal([self.h_units,10], stddev=0.1), name="W2")
            b2=tf.Variable(tf.zeros([10]),name="b2")

            hidden1 = tf.nn.relu(tf.matmul(self.x_, W1) + b1)
            hidden1_drop = tf.nn.dropout(hidden1, self.keep_prob)
            logits = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)



            # u1=tf.matmul(self.x_,W1)+b1
            # mu1,sigma1=batch_normalization_layer(u1,is_train=is_train)
            # scale1=tf.Variable(tf.ones([1]),name="scale1")
            # shift1=tf.Variable(tf.zeros([1]),name="shift1")
            # epsilon=0.001
            # u1=tf.nn.batch_normalization(u1,mu1,sigma1,shift1,scale1,epsilon)
            # h1=tf.nn.relu(u1)
            # h1_drop=dropout_layer(h1,self.keep_prob,is_train=is_train)
            # u2 = tf.matmul(h1_drop, W2) + b2
            # logits = tf.nn.softmax(u2)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    mu,sigma=tf.nn.moments(incoming,axes=[0])
    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
        ema_apply_op = ema.apply([mu, sigma])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mu), tf.identity(sigma)

    mu,sigma=tf.cond(tf.constant(is_train,dtype=tf.bool),mean_var_with_update,lambda :(ema.average(mu),ema.average(sigma)))

    return mu,sigma
    
def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    if is_train:
        return tf.nn.dropout(incoming, drop_rate)
    else:
        return incoming
    # return tf.cond(is_train,lambda :tf.nn.dropout(incoming, drop_rate),lambda :incoming)

