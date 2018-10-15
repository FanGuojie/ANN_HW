import  numpy as np
# def softmax(x):
#     x = x - np.max(x)
#     exp_x = np.exp(x)
#     return exp_x / np.sum(exp_x)

class Network(object):
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1

    def forward(self, input):
        output = input
        for i in range(self.num_layers):
            output = self.layer_list[i].forward(output)
        # (n,_)=output.shape
        # for i in range(n):
        #     output[i]=softmax(output[i])
        return output

    def backward(self, grad_output):
        grad_input = grad_output
        for i in range(self.num_layers - 1, -1, -1):
            grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)
