# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2015

from csv_data import RegressionData
from neural_network import NeuralNetwork

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from sklearn import cross_validation
from pybrain.tools.neuralnets import NNregression
from pybrain.tools.neuralnets import NNclassifier

from sklearn.metrics import mean_squared_error

import numpy as np

class RegressionNeuralNetwork(NeuralNetwork):
    def run(self, data):
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
        :param ds_train (TweetClassificationDatasetFactory): the training dataset the neural network is trained with.
        :param ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        :returns error (float): the percent error of the test dataset, tested on the neural network.
        """
        trainer = BackpropTrainer(
            self.network,
            dataset=data.DS,
            learningrate=self.learningrate,
            momentum=self.momentum,
            verbose=self.verbose,
            batchlearning=self.batchlearning)

        trainer.trainUntilConvergence(
            dataset=data.DS,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)

        error = self.test(data)
        return error

    def test(self, data):
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
        :param ds_test (TweetRegressionDatasetFactory): the test dataset evaluated.
        :returns error (float): the percent error of the test dataset, tested on the network.
        """
        result = self.network.activateOnDataset(data.DS)
        error = mean_squared_error(data.DS['target'], result)
        return error

    def run_with_crossvalidation(self, data, iterations=5):
        """
        This function estimates the performance of the neural network using crossvalidation using a specified dataset.
        Args:
        :param ds (TweetRegressionDatasetFactory): the dataset used to crossvalidate the network.
        :param iterations (int, optional): number of iterations for the crossvalidation.
        :returns error (float): the average percent error of the dataset, tested on the network using crossvalidation.
        """
        x = ds.DS['input']
        y = ds.DS['target']
        n, m = x.shape
        errors = np.zeros(iterations)
        cv = cross_validation.KFold(n, iterations, shuffle=True)

        i = 0
        for train_index, test_index in cv:
            x_train = x[train_index, :]
            y_train = y[train_index, :]
            x_test = x[test_index, :]
            y_test = y[test_index, :]

            ds_train = RegressionData.conv2DS(x_train, y_train)
            ds_test = RegressionData.conv2DS(x_test, y_test)

            trainer = BackpropTrainer(
                self.network,
                dataset=ds_train,
                learningrate=self.learningrate,
                momentum=self.momentum,
                verbose=self.verbose,
                batchlearning=self.batchlearning)

            trainer.trainUntilConvergence(
                dataset=ds_train,
                maxEpochs=self.max_epochs,
                continueEpochs=self.con_epochs)

            tstresult = self.test(ds_test)
            errors[i] = tstresult[0]
            i += 1

        print "Simple Regression Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def getNetworkOutput(self, input):
        return self.network.activate(input)

