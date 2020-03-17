import pandas as pd
import numpy as np
from DecisionTree import DecisionTree
import matplotlib.pyplot as plt
from kNN import kNN
from scipy import stats

class Utils():
    def __init__(self):
        pass

    def read_data(self, file_name="data\heart.csv", y_col="target"):
        data = pd.read_csv(file_name, header=None)
        data.columns = list(data.loc[0])
        data = data.loc[1:]
        data = data.astype('float')
        Y = data.loc[:][y_col]
        Y.columns = ["target"]
        X = data.drop(y_col, axis=1)
        return X, Y

    def shuffle_data(self, X, Y):
        data = X
        data["target"] = Y
        shuffle_data = data.sample(frac=1).reset_index(drop=True)
        shuffle_Y = shuffle_data["target"]
        shuffle_Y.columns = ["target"]
        shuffle_X = shuffle_data.drop("target", axis=1)

        return shuffle_X, shuffle_Y

    def split_data(self, data, train_fraction):
        length_of_data = len(data)
        data = data.loc[1:]
        train_size = int(train_fraction * length_of_data / 100)
        train = data.loc[:train_size][:]
        train = train.reset_index(drop=True)
        test = data.loc[train_size + 1:][:]
        test = test.reset_index(drop=True)

        return train, test

    def calculate_accuracy(self, predicted_Y, real_Y):
        acc = 0
        for i, j in zip(predicted_Y, real_Y):
            if i == j:
                acc += 1

        return (acc / len(predicted_Y)) * 100

    def calculate_confusion_matrix(self, predicted_Y, real_Y):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for i, j in zip(predicted_Y, real_Y):
            if i == 1 and j == 1:
                TP += 1
            elif i == 0 and j == 0:
                TN += 1
            elif i == 1 and j == 0:
                FP += 1
            elif i == 0 and j == 1:
                FN += 1

        return TP, TN, FP, FN

    def calculate_classification_reports(self, predicted_Y, real_Y):
        TP, TN, FP, FN = self.calculate_confusion_matrix(predicted_Y, real_Y)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / precision + recall
        acc = self.calculate_accuracy(predicted_Y, real_Y)

        return specificity, precision, recall, F1, acc

    def t_test(self,kNN_vector,dtree_vector):
        # size
        N = len(kNN_vector)
        kNN_np_vector = np.array(kNN_vector)
        dtree_np_vector = np.array(dtree_vector)
        mean_dtree = dtree_np_vector.mean()
        mean_knn = kNN_np_vector.mean()

        # based on lesson silde2 formula
        dif = [(i - j)**2 for i,j in zip(kNN_np_vector,dtree_np_vector)]
        t = abs(mean_knn - mean_dtree) * np.sqrt((N * (N - 1)) / sum(dif))

        degree_freedom = N - 1
        p = 1 - stats.t.cdf(t , df = degree_freedom)

        return t, p

    def plot(self, depth_or_k, acc, t, name, mode):
        if mode == 0:
            x = depth_or_k
            y = acc
            plt.plot(x, y)
            plt.xlabel('depth')
            plt.ylabel(name + ' accuracy ')
            plt.title(name + ' accuracy for \n decision tree threshold =' + str(t))
            plt.show()
        elif mode == 1:
            x = depth_or_k
            y = acc
            plt.plot(x, y)
            plt.xlabel('k')
            plt.ylabel(name + ' accuracy ')
            plt.title(name + ' accuracy for kNN' )
            plt.show()

    def k_fold(self,data,k,iteration):
        length = len(data)
        frac = int(length/k)
        if (iteration+1)*frac <= length :
            test = data.loc[iteration*frac:(iteration+1)*frac-1][:]
            if iteration == 0 : train_1 = data.loc[1:iteration*frac]
            else: train_1 = data.loc[0:iteration*frac]
            train_2 = data.loc[(iteration+1)* frac:]
            frames = [train_1,train_2]
            train = pd.concat(frames)
        else :
            test = data.loc[iteration*frac:][:]
            train = data.loc[:iteration*frac][:]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        return train, test
