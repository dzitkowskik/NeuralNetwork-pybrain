# Regression (example in regression_test.py)

To use regression first create data from csv file as follows:

```python
d_train = RegressionData('data/regress_1_train.csv') // train data
d_test = RegressionData('data/regress_1_tst.csv') // test data
```
Next create regression neural network:

```python
nn = RegressionNeuralNetwork(1, 1, 3, verbose=True, learningrate=0.025)
nn.apply_custom_network([3,3]) // shape network to have two layers with 3 nodes each
```
Run network for classification:

```python
t = nn.run(d_train)
train_error = nn.test(d_train)
test_error = nn.test(d_test)
```
To plot the results we can use plotResult method od data class

```python
figure(1)
d_train.plotResult(nn)
figure(2)
d_test.plotResult(nn)
show()
```

# Classification (example in classification_test.py)

To use classification first create classification data from csv file:

```python
d_train = ClassificationData('data/class_2_train.csv')
d_test = ClassificationData('data/class_2_tst.csv')
nClasses = d_train.DS.nClasses // get how many classes are in the dataset
```
Create classification neural network with nClasses number of output neurons:

```python
nn = ClassificationNeuralNetwork(2, nClasses, 6)
nn.apply_custom_network([6,6])
```

We can run neural network like before or run it with crossvalidation:

```python
t = nn.run_with_crossvalidation(d_train)
train_error = nn.test(d_train)
test error = nn.test(d_test)
```

We can also print the results as nice plots:

```python
figure(1)
d_train.plotResult(nn)
figure(2)
d_test.plotResult(nn)
show()
```
