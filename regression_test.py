# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014

from pylab import *
from csv_data import RegressionData
from regression import RegressionNeuralNetwork
from pybrain.utilities import percentError

d_train = RegressionData('data/regress_1_train.csv')
nn = RegressionNeuralNetwork(1, 1, 3, verbose=True, learningrate=0.025)

nn.apply_custom_network([3,3])

#RUN NETWORK FOR CLASSIFICATION
t = nn.run(d_train)
print 'train error', nn.test(d_train)

d_test = RegressionData('data/regress_1_tst.csv')
print 'test error', nn.test(d_test)

# get error
# result = t.testOnClassData(dataset=d_train.DS)
# error = percentError(result, d_train.DS['class'])
# print 'error =', error

# PLOT RESULTS
figure(1)
d_train.plotResult(nn)
figure(2)
d_test.plotResult(nn)

show()
