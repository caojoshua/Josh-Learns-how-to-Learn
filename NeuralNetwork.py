
import Layer
import LossFunction
import numpy as np
from sklearn import metrics

class NeuralNetwork:
	def __init__(self, input_length = 0, loss=LossFunction.MeanSquaredError(), lr=0.01):
		self.layers = [Layer.InputLayer(input_length)]
		self.loss = loss
		self.lr = lr
		
	def add_layer(self, Layer, *args, **kwargs):
		self.layers.append(Layer(input_length = self.layers[-1].get_output_length(), *args, **kwargs))
		
	def train(self, Ytr, Xtr, Xva, Yva, batch_size=32, epochs=1):
		for epoch in range(epochs):
			print("epoch #", epoch, ":")
			random_indices = np.arange(len(Xtr))
			np.random.shuffle(random_indices)
			Xtr, Ytr = Xtr[random_indices], Ytr[random_indices]
			batch_num = 0
			while batch_num * batch_size < len(Xtr):
				x_batch, y_batch = Xtr[batch_num : batch_num + batch_size], Ytr[batch_num : batch_num + batch_size]
				prediction, inputs = self.forward_propagate(x_batch)
				self.backward_propagate(y_batch, prediction, inputs)
				batch_num += 1
			Y_hat_tr, _ = self.predict(Xtr)
			Y_hat_va, _ = self.predict(Xva)
			print("\tTraining Accuracy: ", self.compute_acc(Ytr, Y_hat_tr))
			print("\tValidation Accuracy: ", self.compute_acc(Yva, Y_hat_va))
			print("\tTraining loss: ", self.compute_loss(Ytr, Y_hat_tr))
			print("\tvalidation loss: ", self.compute_loss(Yva, Y_hat_va))
	
	def predict(self, predict_set):
		return self.forward_propagate(predict_set)
		
	def forward_propagate(self, propagate):
		inputs = []
		for layer in self.layers:
			inputs.append(propagate)
			propagate = layer.forward_propagate(propagate)
		return (propagate, inputs)
		
	def backward_propagate(self, labels, prediction, inputs):
		propagate = self.loss.gradient(labels, prediction)
		batch_size = labels.shape[0]
		# store inputs for each layer in dict where {key = reference to layer : value = inputs}
		delta_x = dict()
		for i in range(len(self.layers) - 1, -1, -1):
			layer = self.layers[i]
			if isinstance(layer, Layer.WeightedLayer):
				delta_x[layer] = -layer.compute_gradient_wrt_weight(inputs[i]).T.dot(propagate) * self.lr / batch_size
				propagate = propagate.dot(layer.gradient(inputs[i]).T) / propagate.shape[-1]
			else:
				propagate = propagate * layer.gradient(inputs[i])

		for weighted_layer in delta_x:
			weighted_layer.increment_weights(delta_x[weighted_layer])
				
	def compute_acc(self, y_true, y_hat):
		return metrics.accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_hat, axis=1))
		
	def compute_loss(self, y_true, y_hat):
		return np.mean(self.loss.loss(y_true, y_hat))
	
