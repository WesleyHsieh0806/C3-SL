import numpy as np
import pandas as pd
import os

''' 
*   Sort the data by labels and save them as csv files(for Non-IID case)
'''
if not os.path.isdir('../MNIST/sorted_data_csv'):
    os.makedirs('../MNIST/sorted_data_csv')

# the path of each csv file
train_data_path = '../MNIST/data_csv/train_data.csv'
train_label_path = '../MNIST/data_csv/train_label.csv'
test_data_path = '../MNIST/data_csv/test_data.csv'
test_label_path = '../MNIST/data_csv/test_label.csv'
# the path to be saved
sort_train_data_path = '../MNIST/sorted_data_csv/train_data.csv'
sort_train_label_path = '../MNIST/sorted_data_csv/train_label.csv'
sort_test_data_path = '../MNIST/sorted_data_csv/test_data.csv'
sort_test_label_path = '../MNIST/sorted_data_csv/test_label.csv'
# Load the data
train_data = pd.read_csv(train_data_path)
train_data = np.asarray(train_data.iloc[:, 1:])

train_label = pd.read_csv(train_label_path)
train_label = np.asarray(train_label.iloc[:, 1:])

test_data = pd.read_csv(test_data_path)
test_data = np.asarray(test_data.iloc[:, 1:])

test_label = pd.read_csv(test_label_path)
test_label = np.asarray(test_label.iloc[:, 1:])

# Check the size of data
print("Size of training data:{}".format(train_data.shape))
print("Size of training labels:{}".format(train_label.shape))
print("Size of testing data:{}".format(test_data.shape))
print("Size of testing labels:{}".format(test_label.shape))

# sort the data by label
train_index = train_label.argsort(kind='mergesort', axis=0).squeeze(axis=1)
sort_train_data = train_data[train_index]
sort_train_label = train_label[train_index]

test_index = test_label.argsort(kind='mergesort', axis=0).squeeze(axis=1)
sort_test_data = test_data[test_index]
sort_test_label = test_label[test_index]

# Save them as csv files
sort_train_data = pd.DataFrame(sort_train_data, columns=[
                               'Feature'+str(i) for i in range(1, len(test_data[0])+1)])
sort_train_data.to_csv(sort_train_data_path)

sort_train_label = pd.DataFrame(sort_train_label, columns=['Label'])
sort_train_label.to_csv(sort_train_label_path)

sort_test_data = pd.DataFrame(sort_test_data, columns=[
                              'Feature'+str(i) for i in range(1, len(test_data[0])+1)])
sort_test_data.to_csv(sort_test_data_path)

sort_test_label = pd.DataFrame(sort_test_label, columns=['Label'])
sort_test_label.to_csv(sort_test_label_path)

for i in range(10):
    print("Size of class {}:{} in train data".format(
        i, sort_train_label[sort_train_label == i].count(axis=0)))
for i in range(10):
    print("Size of class {}:{} in test data".format(
        i, sort_test_label[sort_test_label == i].count(axis=0)))
