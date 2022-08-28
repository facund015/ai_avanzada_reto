import numpy as np
import pandas as pd
import math as m
import metricas as mt
from sklearn.model_selection import train_test_split
import os


def predict_all_classes(X_, thetas):
    p = []
    for k in range(n_wine_class):
        p.append(
            h(X_.x, thetas[k], X_.x1, X_.x2, X_.x3, X_.x4, X_.x5, X_.x6, X_.x7, X_.x8, X_.x9, X_.x10, X_.x11, X_.x12))
    return p.index(max(p)) + 1


def predict_single_class(x, k, thetas):
    return h(x, thetas[k])


def h(x, theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return 1 / (1 + m.exp(-(
                theta[0] + theta[1] * x + theta[2] * x1 + theta[3] * x2 + theta[4] * x3 + theta[5] * x4 + theta[
            6] * x5 + theta[7] * x6 + theta[8] * x7 + theta[9] * x8 + theta[10] * x9 + theta[11] * x10 + theta[
                    12] * x11 + theta[13] * x12)))


cols = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
        "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv("wine.data", header=None, names=cols)
wine_classes = df["Class"].unique().tolist()
n_wine_class = len(wine_classes)

X_ = df.drop(columns=["Class"], axis=1)
X_.columns = ["x", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
y_ = df.Class.values

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X_, y_)

alpha = 0.00001
iters = 200
thetas = np.full((n_wine_class, 14), 0.00001)

n_train = len(y_train)

print("\nTraining...")
for k in range(n_wine_class):
    for idx in range(iters):
        acumDelta = {"x_": [], "x": [],
                     "x1": [], "x2": [],
                     "x3": [], "x4": [],
                     "x5": [], "x6": [],
                     "x7": [], "x8": [],
                     "x9": [], "x10": [],
                     "x11": [], "x12": []}
        for (i_row, X), y in zip(X_train.iterrows(), y_train):
            if y != (k + 1):
                # print(k+1, y, 'replaced')
                y = 0
            else:
                # print(k+1, y)
                y = 1

            acumDelta['x_'].append(
                h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10, X.x11, X.x12) - y)
            acumDelta['x'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10, X.x11,
                                     X.x12) - y) * X.x)
            acumDelta['x1'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x1)
            acumDelta['x2'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x2)
            acumDelta['x3'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x3)
            acumDelta['x4'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x4)
            acumDelta['x5'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x5)
            acumDelta['x6'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x6)
            acumDelta['x7'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x7)
            acumDelta['x8'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x8)
            acumDelta['x9'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                      X.x11, X.x12) - y) * X.x9)
            acumDelta['x10'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                       X.x11, X.x12) - y) * X.x10)
            acumDelta['x11'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                       X.x11, X.x12) - y) * X.x11)
            acumDelta['x12'].append((h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10,
                                       X.x11, X.x12) - y) * X.x12)

        sJt_ = sum(acumDelta['x_'])
        sJt0 = sum(acumDelta['x'])
        sJt1 = sum(acumDelta['x1'])
        sJt2 = sum(acumDelta['x2'])
        sJt3 = sum(acumDelta['x3'])
        sJt4 = sum(acumDelta['x4'])
        sJt5 = sum(acumDelta['x5'])
        sJt6 = sum(acumDelta['x6'])
        sJt7 = sum(acumDelta['x7'])
        sJt8 = sum(acumDelta['x8'])
        sJt9 = sum(acumDelta['x9'])
        sJt10 = sum(acumDelta['x10'])
        sJt11 = sum(acumDelta['x11'])
        sJt12 = sum(acumDelta['x12'])

        thetas[k][0] = thetas[k][0] - alpha / n_train * sJt_
        thetas[k][1] = thetas[k][1] - alpha / n_train * sJt0
        thetas[k][2] = thetas[k][2] - alpha / n_train * sJt1
        thetas[k][3] = thetas[k][3] - alpha / n_train * sJt2
        thetas[k][4] = thetas[k][4] - alpha / n_train * sJt3
        thetas[k][5] = thetas[k][5] - alpha / n_train * sJt4
        thetas[k][6] = thetas[k][6] - alpha / n_train * sJt5
        thetas[k][7] = thetas[k][7] - alpha / n_train * sJt6
        thetas[k][8] = thetas[k][8] - alpha / n_train * sJt7
        thetas[k][9] = thetas[k][9] - alpha / n_train * sJt8
        thetas[k][10] = thetas[k][10] - alpha / n_train * sJt9
        thetas[k][11] = thetas[k][11] - alpha / n_train * sJt10
        thetas[k][12] = thetas[k][12] - alpha / n_train * sJt11
        thetas[k][13] = thetas[k][13] - alpha / n_train * sJt12

print(thetas)

predicts = []
for idx, value in X_.iterrows():
    predicts.append(predict_all_classes(value, thetas))

print(y_)
print(predicts)
acc, hits, misses = mt.accuracy_simple(predicts, y_)
print('Accuracy:', acc, '%')
print('Hits:', hits)
print('Misses:', misses)

os.system('pause')
