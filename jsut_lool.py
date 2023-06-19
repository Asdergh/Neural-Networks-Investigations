import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random as rd
import math as mt
import json as js



def decision_graph(param_x, labels, classifier):
    markers = ["x", "o", "s", "v", "^"]
    colors = ["green", "red", "blue", "gray"]
    cmaps = mcolors.ListedColormap(colors[:len(np.unique(labels))])

    xx1_param_grid, xx2_param_grid = np.meshgrid(np.linspace(param_x[:, 0].min() - 1, param_x[:, 1].max() + 1),
                                                 np.linspace(param_x[:, 1].min() - 1, param_x[:, 1].max() + 1))
    result_param_grid = classifier.predict(np.array([xx1_param_grid.ravel(), xx2_param_grid.ravel()]).T)
    result_param_grid = result_param_grid.reshape(xx1_param_grid.shape)

    plt.contourf(xx1_param_grid, xx2_param_grid, result_param_grid, cmap=cmaps, alpha=0.3)

    for (index, cls) in enumerate(np.unique(labels)):
        plt.scatter(param_x[labels==cls, 0], param_x[labels==cls, 1], color=colors[index],
                    marker=markers[index], label=f"cls{cls}")
    
    plt.legend(loc="upper left")
    plt.show()


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])



data = load_iris()
x_data = np.asarray(data.data[:, [2, 3]])
labels = np.asarray(data.target)
snn = StandardScaler()
x_data_train, x_data_test, label_train, label_test = train_test_split(x_data, labels, test_size=0.3, random_state=0)
snn.fit(x_data_train)
x_data_train_std, x_data_test_std = snn.transform(x_data_train), snn.transform(x_data_test)


tree_net = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
tree_net.fit(x_data_train, label_train)
decision_graph(param_x=x_data, labels=labels, classifier=tree_net)
