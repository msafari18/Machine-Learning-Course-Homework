import numpy as np
import pandas as pd
import copy
import math
from sklearn.utils import shuffle


class DecisionTree():
    def __init__(self, max_depth, threshold):
        self.max_depth = max_depth
        self.threshold = threshold
        self.features_values = {}
        self.target_values = [0, 1]
        self.cont_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    def entropy_for_one_class(self, c, n):
        return -(c * 1.0 / n) * math.log(c * 1.0 / n, 2)

    def calculate_entropy(self, data):
        c0 = len(data.query("target == 0.0"))
        c1 = len(data.query("target == 1.0"))

        if c0 == 0 or c1 == 0:
            return 0
        return self.entropy_for_one_class(c1, c1 + c0) + self.entropy_for_one_class(c0, c1 + c0)

    def calculate_info_gain(self, data, not_used_features):
        current_node_entropy = self.calculate_entropy(data)
        all_info_gain = {}
        thresh = {}
        for f in not_used_features:
            feature_entropy = []
            if f in self.cont_features:
                mean = (sum(self.features_values[f]) / len(self.features_values[f]))
                d = data[data[f] <= mean]
                feature_entropy.append(self.calculate_entropy(d) * len(d) / len(data))
                d = data[data[f] > mean]
                feature_entropy.append(self.calculate_entropy(d) * len(d) / len(data))
                gain = current_node_entropy - sum(feature_entropy)
                thresh[f] = mean
                all_info_gain[f] = gain

            else:
                for value in self.features_values[f]:
                    d = data[data[f] == value]
                    feature_entropy.append(self.calculate_entropy(d) * len(d) / len(data))
                gain = current_node_entropy - sum(feature_entropy)
                all_info_gain[f] = gain
        return all_info_gain, thresh

    def label_leaf(self, data, depth):
        c1 = len(data.query("target == 1.0"))
        c0 = len(data.query("target == 0.0"))
        total = len(data)
        if depth >= self.max_depth:
            return True, 1 if c1 > c0 else 0
        elif (c0 / total) >= self.threshold:
            return True, 0
        elif (c1 / total) >= self.threshold:
            return True, 1
        else:
            return False, None

    def make_tree(self, data, not_used_features, depth, node):
        labeled, class_type = self.label_leaf(data, depth)
        if labeled:
            node["class"] = class_type
            node["type"] = "leaf"
            return node
        else:
            features_info_gain, t = self.calculate_info_gain(data, not_used_features)
            selected_features = max(features_info_gain, key=features_info_gain.get)
            node["feature"] = selected_features
            if selected_features in self.cont_features:
                node["feature_type"] = "cont"
                node["thresh"] = t[selected_features]
                nuf = copy.copy(not_used_features)
                child_node_l = data[data[selected_features] <= t[selected_features]]
                child_node_r = data[data[selected_features] > t[selected_features]]
                not_used_features.remove(selected_features)
                new_node = {"feature": None, "feature_type": None, "parent_value": t[selected_features], "type": "node", "childs": []}
                child = self.make_tree(child_node_l, not_used_features, depth + 1, new_node)
                node["childs"].append(child)
                not_used_features = nuf

                not_used_features.remove(selected_features)
                new_node = {"feature": None, "feature_type": "cont", "parent_value": t[selected_features], "type": "node",
                            "childs": []}
                child = self.make_tree(child_node_r, not_used_features, depth + 1, new_node)
                node["childs"].append(child)
                # not_used_features = nuf
            else:
                node["feature_type"] = "cat"
                for value in self.features_values[selected_features]:
                    nuf = copy.copy(not_used_features)
                    child_node = data[data[selected_features] == value]
                    if len(child_node) != 0:
                        not_used_features.remove(selected_features)
                        new_node = {"feature": None, "parent_value": value, "type": "node", "childs": []}
                        child = self.make_tree(child_node, not_used_features, depth + 1, new_node)
                        node["childs"].append(child)
                        not_used_features = nuf
                    else:
                        continue

        return node

    def fit(self,X_train,Y_train):

        features = list(X_train.columns)
        data = copy.copy(X_train)
        data["target"] = Y_train
        for i in features:
            self.features_values[i] = list(data[i].unique())

        tree = self.make_tree(data, features, 0, {"feature": None, "feature_type": None, "type": "root", "childs": []})
        return tree

    def predict(self,X,tree):

        predicted = []
        for i in range(0, len(X)):
            self.predict_row(X.loc[i], tree, predicted)
        return predicted

    def predict_row(self, row, tree, pred):


        feature = tree["feature"]
        if tree["feature_type"] == "cat":
            for child in tree["childs"]:
                if row[feature] == child["parent_value"]:
                    if child["type"] == "leaf":
                        pred.append(child["class"])
                        return child["class"]
                    else:
                        self.predict_row(row, child, pred)
        else:

            if row[feature] <= tree["thresh"]:
                ch = tree["childs"][0]
                if ch["type"] == "leaf":
                    pred.append(ch["class"])
                    return ch["class"]
                else:
                    self.predict_row(row, ch, pred)
            elif row[feature] > tree["thresh"]:

                ch = tree["childs"][1]
                if ch["type"] == "leaf":
                    pred.append(ch["class"])
                    return ch["class"]
                else:
                    self.predict_row(row, ch, pred)


        return -1