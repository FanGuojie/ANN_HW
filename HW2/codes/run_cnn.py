from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
from plot import show
from solve_net import show4category
train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 8, 3, 1, 0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 8 x 7 x 7
model.add(Reshape('flatten', (-1, 392)))
model.add(Linear('fc3', 392, 10, 0.01))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0,
    'momentum': 0.7,
    'batch_size': 100,
    'max_epoch': 300,
    'disp_freq': 5,
    'test_epoch': 2
}
lo_list=[]
ac_list=[]
categories=[1,2,3,4]
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        lo,ac=test_net(model, loss, test_data, test_label, config['batch_size'])
        lo_list.append(lo)
        ac_list.append(ac)


show4category(model, test_data, test_label, categories)
show(lo_list,ac_list,config['max_epoch'],config['test_epoch'])
