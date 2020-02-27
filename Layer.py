
import math
import numpy as np

class Layer:
	def __init__(self, input_length):
		self.input_length = input_length
		
	def forward_propagate(self, x):
		return x
		
	def gradient(self):
		return 1
	
	def get_input_length(self):
		return self.input_length
	
	def get_output_length(self):
		return self.input_length
		
	def __str__(self):
		return "Abstract Layer"
		
class InputLayer(Layer):
	def __init__(self, input_length):
		super().__init__(input_length)
		
	def forward_propagate(self, x):
		return x
	
	def gradient(self, x):
		return 1
		
	def __str__(self):
		return "Input Layer"
		
		
class ActivationLayer(Layer):
	def __init__(self, input_length):
		super().__init__(input_length)
		
	def forward_propagate(self, x):
		return self.activation(x)
	
	def activation(self, x):
		return x
		
class Relu(ActivationLayer):
	def __init__(self, input_length):
		super().__init__(input_length)
		
	def activation(self, x):
		x = x.copy()
		x[x<0] = 0
		return x
		
	def gradient(self, x):
		x = x.copy()
		x[x<0] = 0
		x[x>0] = 1
		return x
		
	def __str__(self):
		return "Relu Layer"
		
class Softmax(ActivationLayer):
	def __init__(self, input_length):
		super().__init__(input_length)
		
	def activation(self, x):
		# subtracting by the max prevents overflow
		# this will create negative exponents, which will approach 0
		# apparently python will create NaNs when approaching infinite, but not 0
		e = np.exp(x - np.expand_dims(np.max(x, axis=-1), -1))
		return e / np.expand_dims(np.sum(e, axis=-1), axis=-1)
		
	# TODO: make this vectorizable by numpy (is this possible?)
	def gradient(self, x):
		p = self.activation(x)
		return p*(1-p)
		
	def __str__(self):
		return "Softmax Layer"
		
# TODO: convolutional layer when implemented will be a subclass of this
# gradient is stored as weight_gradient, assume bias gradient is 1
class WeightedLayer(Layer):
	def __init__(self, input_length):
		super().__init__(input_length)
		self.weights = np.array([])
		self.biases = np.array([])
		
	def get_weights(self):
		return self.weights
		
	def get_biases(self):
		return self.biases
		
	def increment_weights(self, offset):
		self.weights += offset
		
	def increment_biases(self, offset):
		self.biases += offset
		
	def compute_gradient_wrt_weight(self):
		return 1
		
	def compute_gradient_wrt_bias(self):
		return 1
		
	def forward_propagate(self, inputs):
		return super().forward_propagate(inputs)
		
class FullyConnectedLayer(WeightedLayer):
	def __init__(self, input_length, num_neurons=0):
		super().__init__(input_length)
		# glorot_uniform initializer
		# uniform random distribution from [-limit, limit]
		# where limit is sqrt(6 / (fan_int + fan_out)
		self.weights = np.random.random(size=(input_length, num_neurons))
		glorot_lim = math.sqrt(6 / (input_length + num_neurons))
		self.weights = self.weights * 2 * glorot_lim - glorot_lim
		self.biases = np.zeros(shape=(num_neurons))
		
	def forward_propagate(self, x):
		super().forward_propagate(x)
		x = x.dot(self.weights) + self.biases
		return x
		
	def gradient(self, x):
		return self.weights
		
	def compute_gradient_wrt_weight(self, x):
		return x
	
	def get_output_length(self):
		return self.weights.shape[1]
		
	def __str__(self):
		return "Fully Connected Layer"
		
		
		