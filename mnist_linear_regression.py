import numpy as np
import tensorflow as tf

class Linear_regression:
	def __init__(self,train_data,train_label,test_data,test_label,weight,bias):
		self.train_data=train_data
		self.train_label=train_label
		self.test_data=test_data
		self.test_label=test_label

		self.weight=tf.Variable(weight,dtype=tf.float64)
		self.bias=tf.Variable(bias,dtype=tf.float64)

		self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.5)

	def loss(self,data,label):
		return tf.reduce_mean((self.model(data)-label)**2)
	def model(self,data):
		data=tf.reshape(data/255,shape=[-1,data.shape[1]**2])
		val=tf.linalg.matmul(data,self.weight)+self.bias
		return val
	
	@tf.function
	def update_parameters(self,data,label):
		loss=lambda:tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.model(data),labels=tf.one_hot(label,10)))
		self.optimizer.minimize(loss,[self.weight,self.bias])


	def train(self,epochs=5000):
		while epochs:
			data=self.train_data
			label=self.train_label
			self.update_parameters(data,label)
			if epochs%50==0:
				acc=self.test(self.test_data,self.test_label)
				if acc>=90:
					print("\r acc {} at epoch{}".format(acc,epochs))
					

			epochs-=1

	def test(self,data,label):
		#print(data.shape[0],data.shape[1])
		#print(self.weight.shape[0],self.weight.shape[1])
		predicted=self.model(data)
		label=tf.one_hot(label,10)
		# acc=100*(1-(abs(predicted-label)/label))
		# for i in range(data.shape[0]):
		# 	print("predicted {} label {} acc {}".format(predicted[i],label[i],acc[i]))
		c=0
		for i in range(data.shape[0]):
		 	if tf.argmax(label[i])==tf.argmax(predicted[i]):
		 		c+=1
		accuracy=c/data.shape[0]*100
		return accuracy


def initialize_param(data):
	#weights=tf.zeros([data.shape[1],1],dtype=tf.float64) 
	weights=tf.random.uniform(shape=[data.shape[1]**2,10],minval=0,maxval=0.9,dtype=tf.float64)
	bias=0.5
	return (weights,bias)


if __name__ == '__main__':
	epochs=200
	#(train_data,train_label),(test_data,test_label)=dataset_sid.load_data('Linear_regression')
	#(train_data,train_label),(test_data,test_label)=tf.keras.datasets.boston_housing.load_data()
	(train_data,train_label),(test_data,test_label)=tf.keras.datasets.mnist.load_data()
	train_data=tf.constant(train_data,dtype=tf.float64)
	test_data=tf.constant(test_data,dtype=tf.float64)
	(weights,bias)=initialize_param(train_data)
	
	print(weights.shape[0],weights.shape[1])

	# train_label=tf.reshape(train_label,shape=[-1,1])
	# test_label=tf.reshape(test_label,shape=[-1,1])
	lg=Linear_regression(train_data,train_label,test_data,test_label,weights,bias)
	lg.train(epochs)
	lg.test(test_data,test_label)



