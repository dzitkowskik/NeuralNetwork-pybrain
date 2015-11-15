from pylab import *
import numpy as np
import random
from scipy.io import loadmat
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from sklearn import preprocessing
import string

class ClassificationData(object):
    def __init__(self, path):
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
        noObs, nCols = self.data.shape

        self.y = self.data[:,nCols-1]
        self.X = self.data[:,0:-1]

        print "Number of observations: ", noObs

        # create ClassificationData
        self.N, self.M = self.X.shape
        self.DS = ClassificationData.conv2DS(self.X, self.y.T)

    @staticmethod
    def conv2DS(Xv, yv=None, labels=string.ascii_uppercase):
        N, M = Xv.shape
        if yv is None:
            yv = np.asmatrix(np.ones((Xv.shape[0], 1)))
            for j in range(len(classNames)):
                yv[j] = j

        le = preprocessing.LabelEncoder()
        y_asnumbers = le.fit_transform(np.ravel(yv))

        C = len(np.unique(np.ravel(yv)))
        DS = ClassificationDataSet(
            M,
            1,
            nb_classes=C,
            class_labels=labels)

        for i in range(Xv.shape[0]):
            DS.appendLinked(Xv[i, :], y_asnumbers[i])

        DS._convertToOneOfMany()
        return DS

    def plotResult(self, nn):
        cmask = np.where(self.y==1);
        plot(self.X[cmask,0], self.X[cmask,1], 'or', markersize=4)
        cmask = np.where(self.y==2);
        plot(self.X[cmask,0], self.X[cmask,1], 'ob', markersize=4)
        cmask = np.where(self.y==3);
        plot(self.X[cmask,0], self.X[cmask,1], 'og', markersize=4)

        minX = min(self.X[:,0])
        minY = min(self.X[:,1])
        maxX = max(self.X[:,0])
        maxY = max(self.X[:,1])

        grid_range = [minX, maxX, minY, maxY];
        delta = 0.05; levels = 100
        a = arange(grid_range[0],grid_range[1],delta)
        b = arange(grid_range[2],grid_range[3],delta)
        A, B = meshgrid(a, b)
        values = np.zeros(A.shape)

        for i in range(len(a)):
            for j in range(len(b)):
                values[j,i] = nn.getNetworkOutput( [ a[i], b[j] ] )
        contour(A, B, values, levels=[1], colors=['k'], linestyles='dashed')
        contourf(A, B, values, levels=linspace(values.min(),values.max(),levels), cmap=cm.RdBu)

class RegressionData(object):
    def __init__(self, path):
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
        noObs, nCols = self.data.shape

        self.y = self.data[:,nCols-1]
        self.X = self.data[:,0:-1]

        print "Number of observations: ", noObs

        # create ClassificationData
        self.N, self.M = self.X.shape
        self.DS = RegressionData.conv2DS(self.X, self.y.T)

    @staticmethod
    def conv2DS(x, y):
        """
        This method converts a matrices x and y of input and target values to a regression dataset
        :param x: input values as a matrix NxM where N is a number of records and M a number of features
        :param y: a target values corresponding to true labels of the records
        :return: an object of regression dataset containing passed data
        """
        ds = SupervisedDataSet(x.shape[1], 1)
        for i in range(0, len(y)):
            ds.addSample(x[i], y[i])
        return ds

    def plotResult(self, nn):
        plot(self.X, self.y, 'or', markersize=4)
        minX = min(self.X)
        maxX = max(self.X)

        delta = 0.025
        a = arange(minX,maxX,delta)
        values = np.zeros(len(a))

        for i in range(len(a)):
            values[i] = nn.getNetworkOutput([a[i]])
        plot(a, values)

def main():
    d = ClassificationData('data/class_1_train.csv')
    print d.DS


if __name__ == '__main__':
    main()
