# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2015

from csv_data import ClassificationData
from neural_network import NeuralNetwork

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from sklearn import cross_validation
from pybrain.tools.neuralnets import NNclassifier

import numpy as np

class ClassificationNeuralNetwork(NeuralNetwork):
    def run(self, data):
        """
        This function trains the ANN
        Args:
        :param ds_train (TweetClassificationDatasetFactory): the training dataset the neural network is trained with.
        :returns: trainer (BackpropTrainer): trained ANN
        """

        trainer = BackpropTrainer(
            self.network,
            dataset=ds_train,
            learningrate=self.learningrate,
            momentum=self.momentum,
            verbose=self.verbose,
            batchlearning=self.batchlearning)

        trainer.trainUntilConvergence(
            dataset=data.DS,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)

        return trainer

    # noinspection PyProtectedMember
    def run_with_crossvalidation(self, data, iterations=5):
        """
        This function estimates the performance of the neural network using crossvalidation using a specified dataset.
        Args:
        :param ds (ClassificationData): the dataset used to crossvalidate the neural network.
        iterations (int, optional): number of iterations for the crossvalidation.
        :returns: error (float): the average percent error of the dataset, tested on the neural network using crossvalidation.
        """
        x = data.X
        y = data.y
        n, m = x.shape
        errors = np.zeros(iterations)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        i = 0
        for train_index, test_index in cv:
            x_train = x[train_index, :]
            y_train = y[train_index]
            x_test = x[test_index, :]
            y_test = y[test_index]

            ds_train = ClassificationData.conv2DS(x_train, y_train)
            ds_test = ClassificationData.conv2DS(x_test, y_test)

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

            errors[i] = percentError(
                trainer.testOnClassData(dataset=ds_test),
                ds_test['class'])
            i += 1

        print "Multi class NN cross-validation test errors: " % errors
        return np.average(errors)

    def test(self, data):
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
        :param ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
        :returns (float): the percent error of the test dataset, tested on the neural network.
        """
        out = self.network.activateOnDataset(data.DS)
        result = np.ravel(np.argmax(out, 1))
        error = percentError(result, data.DS['class'])

        return error

    def getNetworkOutput(self, input):
        res = self.network.activate(input)
        maxValue = max(res)
        class_index = max(xrange(len(res)), key=res.__getitem__)
        return float(class_index) + maxValue
        # return class_index
