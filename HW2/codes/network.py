from sklearn.utils.extmath import softmax


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
        # print("forwand")
        for i in range(self.num_layers):
            # print(self.layer_list[i].name)
            # print(output.shape)
            output = self.layer_list[i].forward(output)
        output=softmax(output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output
        # print("backward")
        for i in range(self.num_layers - 1, -1, -1):
            # print("b ",self.layer_list[i].name)
            grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)
