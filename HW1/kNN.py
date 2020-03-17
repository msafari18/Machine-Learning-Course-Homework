import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle


class kNN():
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def calculate_dist(self,train_row,test_row):
        d = list(np.array(train_row) - np.array(test_row))
        d2 = [abs(i) for i in d]
        return sum(d2)

    def fit(self, X_train, Y_train):

        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predict = []
        for i in range(len(X_test)) :
            X_test_row = X_test.loc[i][:]
            distance_vector = []
            for row_index in range(0,len(self.X_train)):
                dist = self.calculate_dist(list(self.X_train.loc[row_index][:]), X_test_row)
                distance_vector.append(dist)

            distance_sorted = [int(i[0]) for i in sorted(enumerate(distance_vector), key=lambda x: x[1])]
            k_distance_sorted = distance_sorted[:self.k]
            k_first_class = [self.Y_train[i] for i in k_distance_sorted]
            class_0 = k_first_class.count(0.0)
            class_1 = k_first_class.count(1.0)
            if class_1 >= class_0: predict.append(1.0)
            else: predict.append(0.0)

        return predict

