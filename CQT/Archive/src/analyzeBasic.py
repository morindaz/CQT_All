#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:25:23 2017

@author: ansir
"""

import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from tsne import tsne
from matplotlib import pyplot as plt
import matplotlib.font_manager


def read_data():
    # 选取特征对视频进行表示，维度太高会让聚类更加困难，适当选取最有效的一些特征
    selected = np.linspace(0, 20, 20).astype(int)
    # selected = [65,2,4,69,70,71,9,10,75,66,14,15,16,23,68,27,29,77,37,38,48,49,51,53,54,56,58,60,74,63]

    # 准绳问题
    path_c = 'E:\pingan\Archive\\features\\answerC.csv'
    df_c = pd.read_csv(path_c, sep=',', header=0)
    data_c = df_c.values
    data_c = data_c[:, selected]
    # 无关问题
    path_i = 'E:\pingan\Archive\\features\\answerI.csv'
    df_i = pd.read_csv(path_i, sep=',', header=0)
    data_i = df_i.values
    data_i = data_i[:, selected]
    #相关问题
    path_r = 'E:\pingan\Archive\\features\\answerR.csv'
    df_r = pd.read_csv(path_r, sep=',', header=0)
    data_r = df_r.values
    data_r = data_r[:, selected]
    # label=0:准绳问题
    # label=1:无关问题
    # label=2:相关问题
    data = np.concatenate((data_c, data_i, data_r), axis=0)
    label = np.array([0] * len(data_c) + [1] * len(data_i) + [2] * len(data_r))
    return data, label

def dimension_reduction(data, method='tsne', label=None, plot=False):
    n_components = 2
    # 所有降维方法都是基于距离的，需要保证特征距离标准化
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    # 大多数情况下，用tsne 将高纬度数据用二维方式展示出来。不同方法采用不同的特征映射方法计算出
    # 不同的X，用fittransform方法进行标准化，这里是最小-最大规范化
    if method == 'tsne':
        model = TSNE(n_components=n_components, perplexity=20, early_exaggeration=20, method='exact',
                     learning_rate=100, n_iter=1000, random_state=250, verbose=2)
        X = model.fit_transform(data)  # X是两列数据，经过了聚类+规范化
    if method == 'isomap':
        model = Isomap(n_components=n_components, n_neighbors=20)
        X = model.fit_transform(data)
    if method == 'MDS':
        model = MDS(n_components=n_components, verbose=2, n_init=1, max_iter=500)
        X = model.fit_transform(data)
    if method == 'tsne_v2':
        X = tsne(data, 2, 44, 50.0)

    data_len = len(X)  # 统计X长度
    print(data_len)  # data_len = 1653
    print(X)  # 二维数组，（1653L,2L）
    if plot:
        fig, ax = plt.subplots()  # 说明有几个子图，数量未定
        ax.scatter(X[label == 0, 0], X[label == 0, 1], c='darkblue', alpha=0.25, marker='^')
        ax.scatter(X[label == 1, 0], X[label == 1, 1], c='darkred', alpha=0.75, marker='x')
        ax.scatter(X[label == 2, 0], X[label == 2, 1], c='green', alpha=0.25, marker='o')
        ax.set_xlim([np.min(X[label == 0, 0]), np.max(X[label == 0, 0])])
        ax.set_ylim([np.min(X[label == 0, 1]), np.max(X[label == 0, 1])])
        plt.show()
    return X


def outlier_detection(X, label, plot=True):
    # here we use the irrelevent questions to train an outlier detector
    X_train = X[label == 1, :]
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.02)
    clf.fit(X_train)

    # apply the detector to relevent questions to see which answer is suspitous
    X_test = X[label == 2, :]
    y_pred = clf.predict(X_test)

    if plot:
        xx, yy = np.meshgrid(np.linspace(np.min(X[label == 2, 0]), np.max(X[label == 2, 0]), 500),
                             np.linspace(np.min(X[label == 2, 1]), np.max(X[label == 2, 1]), 500))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min() - 7, Z.max(), 10), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

        b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, marker='x', alpha=0.8)
        # b2 = plt.scatter(X_test[y_pred==1, 0], X_test[y_pred==1, 1], c='white', s=20, marker='o',alpha=0.5)
        b3 = plt.scatter(X_test[y_pred == -1, 0], X_test[y_pred == -1, 1], c='green', s=20, marker='o', alpha=0.85)

        plt.xlim([np.min(X[label == 2, 0]), np.max(X[label == 2, 0])])
        plt.ylim([np.min(X[label == 2, 1]), np.max(X[label == 2, 1])])

        plt.legend([b1, b3],
                   ["irrelevent questions",
                    "detected outlier for relevent questions"],
                   loc="upper left",
                   prop=matplotlib.font_manager.FontProperties(size=11))
        plt.show()
    return y_pred


if __name__ == "__main__":
    data, label = read_data()
    X = dimension_reduction(data, method='tsne', label=label, plot=True)



