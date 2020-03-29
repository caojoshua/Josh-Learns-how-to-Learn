
import unittest
import JoshLearnsHowToLearn as jll
from .. import DerivativeApproximation
import numpy as np
	
ALL_ERROR_THRESHOLD = 1e-1
MEAN_ERROR_THRESHOLD = 1e-3
	
NUM_SAMPLES = 1000
NUM_VARIABLES = 100
RANDOM_RANGE_MAX = 10
	
def get_x():
	x = np.random.random(size=(NUM_SAMPLES, NUM_VARIABLES)) * 2 * RANDOM_RANGE_MAX - RANDOM_RANGE_MAX
	return x

class TestRelu(unittest.TestCase):
	def setUp(self):
		self.relu = jll.Layer.Relu(NUM_VARIABLES)
		
	def test_grad(self):
		x = get_x()
		# avoid values close to 0 since ReLU is not countinuous at x=0
		x[x>0] += 0.5
		x[x<0] -= 0.5
		grad_hat = self.relu.gradient(x)
		grad_true = DerivativeApproximation.derivative_approx(self.relu.activation, x)
		diff = np.abs(grad_hat - grad_true)
		
		self.assertTrue(np.all(diff < ALL_ERROR_THRESHOLD))
		self.assertTrue(np.mean(diff) < MEAN_ERROR_THRESHOLD)
	
class TestSoftmax(unittest.TestCase):
	
	def gradient_approx(self, x, h = 1e-5):
		"""
		must compute specific grad_approx for softmax, because its an activation that 
		requires values of other values in the same layer
		
		code uses forward finite difference and is similar to DerivativeApproximation.gradient_approx
		1. for each sample vector (N), duplicate it to a sample matrix (N, N)
		2. add h to each diagonal, which will allow us to compute the gradient wrt each variable
		3. compute finite difference gradient estimate against each row of the sample matrices. 
			Only keep the values in the diagonal because the diagonal is the gradient vector wrt each variable.
		
		for other activations, we simply add h to all values and directly compute the gradient.
		we need this more expensive computation specifically for softmax because it divides by the
		summation of logs across the vector. although h is small, the gradient becomes further from the
		truth as the number of variables increase
		"""
		x = np.expand_dims(x, axis=-2)
		x_forward = x.copy()
		x_forward = x_forward.repeat(repeats=x_forward.shape[-1], axis=-2)
		diag_index = np.arange(x_forward.shape[-1])
		x_forward[:,diag_index, diag_index] += h
		grad_true = self.softmax.activation(x_forward) - self.softmax.activation(x)
		return grad_true[:,diag_index,diag_index] / h
	
	def setUp(self):
		self.softmax = jll.Layer.Softmax(NUM_VARIABLES)
		
	def test_grad(self):
		x = get_x()
		
		grad_hat = self.softmax.gradient(x)
		grad_true = self.gradient_approx(x)
		diff = np.abs(grad_hat - grad_true)
		
		self.assertTrue(np.all(diff < ALL_ERROR_THRESHOLD))
		self.assertTrue(np.mean(diff) < MEAN_ERROR_THRESHOLD)
		
		
if __name__ == '__main__':
	unittest.main()
	