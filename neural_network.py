import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, GaussianLayer, TanhLayer
from pybrain import FullConnection, SoftmaxLayer, BiasUnit

class NeuralNetwork:
    def __init__(self, inp_cnt, out_cnt, hid_cnt=6, max_epochs=20, con_epochs=5,
        bias=True, learningrate=0.01, momentum=0.1, batchlearning=False, verbose=False):
        """
        This function builds an Artificial Neural Network with a specified number of hidden layers.
        Args:
          inp_cnt (int): Number of input neurons
          out_cnt (int): Number of output neurons
          hid_cnt (int): Number of neuron in hidden layer
          max_epochs (int): Maximum number of epochs algorithm will run
          con_epochs (int): Number of epochs algorithm will continue if there is no error gain

        Returns:
          self (MultiClassClassificationNeuralNetwork): the function returns
          an instance of its class, with the neural network initialized.
        """
        self.hid_cnt = hid_cnt
        self.out_cnt = out_cnt
        self.inp_cnt = inp_cnt
        self.max_epochs = max_epochs
        self.con_epochs = con_epochs
        self.bias = bias
        self.learningrate = learningrate
        self.momentum = momentum
        self.batchlearning = batchlearning
        self.verbose = verbose
        self.network = self.__build_default_network()

    def apply_custom_network(self, hidden_counts):
        """
        Changes a network to a new one with possibly multiple layers with various hidden neurons count
        :param hidden_counts: an array of numbers of hidden nodes in every hidden layer. For example:
        [3, 4, 5] means a NN with 3 hidden layers with 3 hidden neurons on 1st layer and so on...
        :return: self
        """
        network = FeedForwardNetwork()

        bias = BiasUnit(name='bias')
        if self.bias:
            network.addModule(bias)

        in_layer = LinearLayer(self.inp_cnt, 'in')
        network.addInputModule(in_layer)

        out_layer = LinearLayer(self.out_cnt, 'out')
        network.addOutputModule(out_layer)

        hidden_layer = SigmoidLayer(hidden_counts[0], 'hidden0')
        network.addModule(hidden_layer)

        in_to_hidden = FullConnection(in_layer, hidden_layer)
        network.addConnection(in_to_hidden)

        if self.bias:
            bias_to_hiden = FullConnection(bias, hidden_layer)
            network.addConnection(bias_to_hiden)

        for i in range(1, len(hidden_counts)):
            last_hidden_layer = hidden_layer
            hidden_layer = SigmoidLayer(hidden_counts[i])
            network.addModule(hidden_layer)
            hidden_to_hidden = FullConnection(last_hidden_layer, hidden_layer)
            network.addConnection(hidden_to_hidden)
            if self.bias:
                bias_to_hiden = FullConnection(bias, hidden_layer)
                network.addConnection(bias_to_hiden)

        hidden_to_out = FullConnection(hidden_layer, out_layer)
        network.addConnection(hidden_to_out)
        if self.bias:
            bias_to_out = FullConnection(bias, out_layer)
            network.addConnection(bias_to_out)

        network.sortModules()
        self.network = network
        return self

    def __build_default_network(self):
        return buildNetwork(self.inp_cnt, self.hid_cnt, self.out_cnt, outclass=LinearLayer, bias=self.bias)
