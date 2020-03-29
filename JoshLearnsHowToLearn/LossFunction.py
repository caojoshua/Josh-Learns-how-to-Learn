
import numpy as np
from . import DerivativeApproximation

class LossFunction:
	def __init__(self):
		pass
		
	def loss(self, y_true, y_hat):
		return -1
		
	def gradient(self, y_true, y_hat):
		return 0
		
class MeanSquaredError(LossFunction):
	def __init__(self):
		super().__init__
		
	def loss(self, y_true, y_hat):
		return np.sum((y_true - y_hat)**2, axis=-1) / y_true.shape[-1]
		
	def gradient(self, y_true, y_hat):
		return -2 * (y_true - y_hat) / len(y_true)
		
class CrossEntropy(LossFunction):
	EPSILON = 1e-7
	def __init__(self):
		super().__init__
		
	def loss(self, y_true, y_hat):
		y_hat = np.clip(y_hat, CrossEntropy.EPSILON, 1-CrossEntropy.EPSILON)
		y_hat /= np.expand_dims(y_hat.sum(axis=-1), axis=-1)
		return -np.sum(y_true*np.log(y_hat), axis=-1)
		
	def gradient(self, y_true, y_hat):
		# TODO: find out why symbolic gradient is not performing well
		return DerivativeApproximation.loss_gradient_approx(self.loss, y_true, y_hat)
# 		y_hat /= np.expand_dims(y_hat.sum(axis=-1), axis=-1)
# 		y_hat = np.clip(y_hat, CrossEntropy.EPSILON, 1-CrossEntropy.EPSILON)
# 		return -y_true/y_hat
		
		