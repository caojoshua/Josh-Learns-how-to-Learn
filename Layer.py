
import math
import numpy as np
import utils

class Layer:
	def __init__(self, input_shape):
		self.input_shape = input_shape
		
	def forward_propagate(self, x):
		return x
		
	def gradient(self, x):
		return 1
	
	def get_input_shape(self):
		return self.input_shape
	
	def get_output_shape(self):
		return self.input_shape
		
	def __str__(self):
		return "Abstract Layer"
		
class InputLayer(Layer):
	def __init__(self, input_shape):
		super().__init__(input_shape)
		
	def forward_propagate(self, x):
		return x
	
	def gradient(self, x):
		return 1
		
	def __str__(self):
		return "Input Layer"
		
		
class ActivationLayer(Layer):
	def __init__(self, input_shape):
		super().__init__(input_shape)
		
	def forward_propagate(self, x):
		return self.activation(x)
	
	def activation(self, x):
		return x
		
class Relu(ActivationLayer):
	def __init__(self, input_shape):
		super().__init__(input_shape)
		
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
	def __init__(self, input_shape):
		super().__init__(input_shape)
		
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
	def __init__(self, input_shape):
		super().__init__(input_shape)
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
	def __init__(self, input_shape, num_neurons=0):
		super().__init__(input_shape)
		# glorot_uniform initializer
		# uniform random distribution from [-limit, limit]
		# where limit is sqrt(6 / (fan_int + fan_out)
		self.weights = np.random.random(size=(input_shape, num_neurons))
		glorot_lim = math.sqrt(6 / (input_shape + num_neurons))
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
	
	def compute_gradient_wrt_bias(self):
		return np.ones(self.biases.shape)[np.newaxis]
	
	def get_output_shape(self):
		return self.weights.shape[1]
		
	def __str__(self):
		return "Fully Connected Layer"
	
class ConvolutionalLayer(WeightedLayer):
	def __init__(self, input_shape, num_filters=1, filter_size=(2,2), stride=1):
		super().__init__(input_shape)
		self.x_filter = filter_size[0]
		self.y_filter = filter_size[1]
		self.filters = np.random.random((num_filters, self.y_filter, self.x_filter, input_shape[-1])) * 2 - 1
# 		self.filters = np.ones((num_filters, self.y_filter, self.x_filter, input_shape[-1]))
		self.stride = stride
		
	def forward_propagate(self, x):
		out = np.ones((x.shape[0],) + self.get_output_shape())
		curr_y, out_y = 0, 0
		filters = utils.flatten_axis(self.filters, axis=1).T
		while curr_y <= x.shape[1] - self.y_filter:
			curr_x, out_x = 0, 0
			while curr_x <= x.shape[2] - self.x_filter:
				sample = x[:, curr_y:curr_y + self.y_filter, 
							curr_x:curr_x + self.x_filter]
				sample = utils.flatten_axis(sample, axis=1)
				out[:,out_y,out_x,:] = sample.dot(filters)
				curr_x += self.stride
				out_x += 1
			curr_y += self.stride
			out_y += 1
		return out
	
	
	def gradient(self, x):
		out = np.zeros((x.shape[0],) + self.input_shape)
		return out
		curr_y = 0
		filter_sum = np.sum(self.filters, axis = 0)
		while curr_y <= x.shape[2] - self.y_filter:
			curr_x = 0
			while curr_x <= x.shape[1] - self.x_filter:
				out[:, curr_x:curr_x + self.x_filter, 
					curr_y:curr_y + self.y_filter] += filter_sum
				curr_x += self.stride
			curr_y += self.stride
		return out
	
	def compute_gradient_wrt_weight(self, x, propagate):
		out = np.zeros((x.shape[0],) + self.filters.shape)
		prop = np.reshape(propagate, (propagate.shape[0], propagate.shape[1] * propagate.shape[2], propagate.shape[3]))
		prop = np.swapaxes(prop, 1, 2)
		curr_y, out_y = 0, 0
		while curr_y < out.shape[2]:
			curr_x, out_x = 0, 0
			while curr_x < out.shape[3]:
				sample = x[:, curr_y:curr_y + propagate.shape[1], 
					curr_x:curr_x + propagate.shape[2]]
				sample = np.sum(sample, axis = 0)
				sample = utils.flatten_axis(sample, -1, reverse = True)
				out[:,:,out_y,out_x] += prop.dot(sample)
				curr_x += self.stride
				out_x += 1
			curr_y += self.stride
			out_y += 1
		return out / x.shape[0]
	
	def compute_gradient_wrt_weight_(self, x):
		out = np.zeros((x.shape[0],) + self.filters.shape)
		x_reshaped = x[:,np.newaxis]
		print(self.filters.shape)
		print(out.shape)
		print(x_reshaped.shape)
		curr_y, out_y = 0, 0
		while out_y < out.shape[2]:
			curr_x, out_x = 0, 0
			while out_x < out.shape[3]:
# 				print(x_reshaped[:, :, out_y:out_y + self.y_filter, out_x:out_x + self.x_filter].shape)
				out += x_reshaped[:, :, out_y:out_y + self.y_filter, out_x:out_x + self.x_filter]
				out_x += 1
				curr_x += self.stride
			out_y += 1
			curr_y += self.stride
		return out
		
	
	def increment_weights(self, offset):
		self.filters
		
	
	def get_output_shape(self):
		return tuple((self.input_shape[0] - self.y_filter + 1, 
					self.input_shape[1] - self.x_filter + 1, 
					len(self.filters)))
		
	def __str__(self):
		return "Convolutional Layer"
	
		
		
		
		
class FlattenLayer(Layer):
	def __init__(self, input_shape):
		super().__init__(input_shape)
		
	def forward_propagate(self, x):
		return x.reshape(x.shape[0], self.get_output_shape())
		
	def gradient(self, x):
# 		return np.ones((len(x),) + self.input_shape)
		return x.reshape((len(x),) + self.input_shape)
	
	def get_output_shape(self):
		return (np.prod(self.input_shape))
	
	def __str__(self):
		return "Flatten Layer"
		
		