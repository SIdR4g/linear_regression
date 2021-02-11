import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Linear_Regression:
	def __init__(self,train_data,train_label):
		self.train_data=train_data
		self.train_label=train_label
		self.weight=0
		self.epochs=2850
		self.bias=0
		self.lr=0.0071
	
	def predict(self,data):
		return np.multiply(data,self.weight)+self.bias
	def train(self,data,label):

		dldb=0
		dldw=0
		print(self.test(data,label))
		for e in range(self.epochs):
			pred=self.predict(data)
			for i in range(pred.shape[0]):
				dldw+=(pred[i]-label[i])*data[i]
				dldb+=(pred[i]-label[i])
			self.bias-=self.lr*(1/data.shape[0])*dldb
			self.weight-=self.lr*(1/data.shape[0])*dldw
			if e%50==0:
				print(self.test(data,label))
		print(self.bias)
		print(self.weight)
		return self.weight,self.bias

	def test(self,data,label):
		pred=self.predict(data)
		error=0
		for i in range(pred.shape[0]):
			error+=(1/2*data.shape[0])*((pred[i]-label[i])**2)
		return error

if __name__ == '__main__':
	z=pd.read_csv('ex1data1.txt')
	z=np.array(z.T)
	train_data=np.reshape(z[0],[z.shape[1],1])
	train_label=np.reshape(z[1],[z.shape[1],1])
	lg=Linear_Regression(train_data,train_label)
	w,b=lg.train(train_data,train_label)
	fig = plt.figure()
	ax = fig.add_subplot()	
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.scatter(train_data,train_label, c='g', label="ground truth")
	plt.plot(train_data, train_data*w + b,linestyle='-')	
	ax.legend()
	plt.show()

	
