import math
import numpy as np
import time
class NNUnit():
	def __init__(self,n_classes,n_pu):
		np.random.seed(int(time.time()))
		self.nc = n_classes
		self.pu = n_pu
		self.b = np.zeros((n_classes,1))
		self.w = np.random.randn(n_classes,n_pu)
		self.oldb = None
		self.oldw = None
		self.a = np.zeros(1,n_classes)
		self.prev_x=None
	def calcActivation(self,x):
		return 1/(1+np.exp(-x))
	def CalcOut(self,x):
		self.prev_x = x
		for i in range(self.nc):
			myIn = np.matmul(self.w[i],np.transpose(np.transpose(x)[i]))+self.b[i]
			self.a[i] = self.calcActivation(myIn)
	def UpdateWB(self,dw,db,lr,regH,b_size):
		self.oldb = np.array(self.b)
		self.oldw = np.array(self.w)
		for i in range(self.nc):
			self.b[i][0] = self.b[i][0] - lr*db[i][0]
		for i in range(self.nc):
			for j in range(self.pu):
				self.w[i][j] = (1-(regH*lr/b_size))*self.w[i][j] - (lr*dw[i][j])



