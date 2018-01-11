#coding=utf-8

import scipy.io as sio
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.image as mpimg
import random
#===================================加载数据============================#
data=sio.loadmat('C:\Users\DELL\Desktop\\data1.mat')
data=data['X']

#===================================零均值化============================#
def featureNormalize(X):
	m,n=X.shape
	u=zeros([1,n])
	for i in range(n):
		u[0,i]=mean(X[:,i])
	for i in range(m):
		X[i]=X[i]-u
	return X,u
#====================================主成分分析==========================#
def pca(X,u,k):
	m,n=X.shape
	C=1.0/(m-1)*X.T.dot(X)                #计算协方差矩阵
	Vals,Vects = linalg.eig(mat(C))       #计算特征值和特征向量
	print Vals
	print Vects
	val_idx=argsort(Vals)                 #对特征值进行排序
	print val_idx
	val_idx=val_idx[:-(k+1):-1]
	print val_idx
	newVects=Vects[:,val_idx]             #从大到小取出对应的k个特征值
	print newVects
	newX=X.dot(newVects)                  #计算n维数据到k为空间的投影（在新基下的坐标）
	print newX.shape
	recon=(newX.dot(newVects.T))+u        #重构
	print recon.shape
	return newX,recon
#====================================Test=================================#	
data,u=featureNormalize(data)
k=1
newX,recon=pca(data,u,1)
plt.figure()
plt.scatter(data[:,0],data[:,1],marker='x',s=70)
plt.scatter(array(recon[:,0]),array(recon[:,1]),marker='o',s=50,c='red')
plt.show()