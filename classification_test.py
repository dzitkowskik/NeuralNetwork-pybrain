# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014

from pylab import *
from csv_data import ClassificationData
from classification import ClassificationNeuralNetwork
from pybrain.utilities import percentError

d_train = ClassificationData('data/class_2_train.csv')
nClasses = d_train.DS.nClasses
nn = ClassificationNeuralNetwork(2, nClasses, 6)

# print nn.network

nn.apply_custom_network([6,6])

# print nn.network

#RUN NETWORK FOR CLASSIFICATION
t = nn.run_with_crossvalidation(d_train)


print 'train error', nn.test(d_train)

d_test = ClassificationData('data/class_2_tst.csv')
print 'test error', nn.test(d_test)

# get error
# result = t.testOnClassData(dataset=d_train.DS)
# error = percentError(result, d_train.DS['class'])
# print 'error =', error
figure(1)
d_train.plotResult(nn)
figure(2)
d_test.plotResult(nn)

show()
