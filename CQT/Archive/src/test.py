# import numpy as np
#
# # data1 = mat(random.rand(10,10))
# # list1 = linspace(0,2,3).astype(int)
# # L = list(list1)
# # L.extend([233,222,111])
# # b = linspace(145,190,46).astype(int)
# # # data2 = data1[0,L]
# # print (L)
# # # print data2
# # print data1
# # print b
#
# # print data1[0,[0,1]]
# # selected = np.linspace(0, 45, 45).astype(int)
# # b = np.linspace(145, 190, 46).astype(int)
# # selected = list(selected)
# # b = list(b)
# # selected.extend(b)
# # print selected
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import Isomap, MDS, TSNE
#
#
# a = [65, 2, 4, 69, 70, 71, 9, 10, 75, 66, 14, 15, 16, 23, 68, 27, 29, 77, 37, 38, 48, 49, 51, 53, 54, 56, 58, 60, 74, 63]
# print a
# iris=load_iris()
# X,y=iris.data,iris.target
# m1 = X[0,:-1]
# m2 = X[0,-1]
# ii = X[0,:]
# print m1
# print m2
# print ii
# # scaler = StandardScaler().fit(X)
# # X = scaler.transform(X)
# # model = TSNE(n_components=2, perplexity=50, early_exaggeration=100, method='exact',
# #              learning_rate=100, n_iter=1000, random_state=250, verbose=2)
# # X = model.fit_transform(X)
# X=SelectKBest(chi2,k=2).fit_transform(X,y)
# # print X_new.shape
# # print type(X_new)
# # print y
# fig, ax = plt.subplots()
# ax.scatter(X[y == 0, 0], X[y == 0, 1], c='darkblue', alpha=0.25, marker='^')
# ax.scatter(X[y == 1, 0], X[y == 1, 1], c='darkred', alpha=0.75, marker='x')
# ax.scatter(X[y == 2, 0], X[y == 2, 1], c='green', alpha=0.25, marker='o')
# plt.show()
# def hello(x):
#     if(x>100):
#         mm=1+x
#         nn=3+x
#         return mm,nn
#     else:
#         print "you are stupid"
#
# mm=hello(50)
# print mm
# hello(250)
y = [x*x for x in range(1,11)]
print y