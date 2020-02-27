
import unittest
import LossFunction
from sklearn.metrics import log_loss, mean_squared_error
import numpy as np
import DerivativeApproximation
	
ALL_ERROR_THRESHOLD = 1e-1
MEAN_ERROR_THRESHOLD = 1e-3
	
NUM_SAMPLES = 1000
NUM_VARIABLES = 100
	
	
def get_y_practicle():
	y_true = np.zeros(shape=(NUM_SAMPLES, NUM_VARIABLES))
	y_true[np.arange(NUM_SAMPLES), np.random.randint(low=0, high=NUM_VARIABLES, size=NUM_SAMPLES)] = 1
	y_hat = np.random.rand(NUM_SAMPLES, NUM_VARIABLES)
	y_hat /= np.expand_dims(np.sum(y_hat, axis=-1), axis=-1)
	return y_true, y_hat
	
class TestMeanSquaredError(unittest.TestCase):
	
	def setUp(self):
		self.mse = LossFunction.MeanSquaredError()
	
	def test_loss_practical(self):
		y_true, y_hat = get_y_practicle()
		
		loss_hat = self.mse.loss(y_true, y_hat)
		# sklearn's mse does opposite dimensions
		loss_true = mean_squared_error(y_true.T, y_hat.T, multioutput='raw_values')
		diff = np.abs(loss_hat - loss_true)
		
		self.assertTrue(np.all(diff < ALL_ERROR_THRESHOLD))
		self.assertTrue(np.mean(diff) < MEAN_ERROR_THRESHOLD)
		
	def test_grad_practical(self):
		y_true, y_hat = get_y_practicle()
		
		grad_hat = self.mse.gradient(y_true, y_hat)
		grad_true = DerivativeApproximation.loss_gradient_approx(self.mse.loss, y_true, y_hat)
		diff = np.abs(grad_hat - grad_true)
		
		self.assertTrue(np.all(diff < ALL_ERROR_THRESHOLD))
		self.assertTrue(np.mean(diff) < MEAN_ERROR_THRESHOLD)
		
		
class TestCrossEntropy(unittest.TestCase):
	
	def setUp(self):
		self.cross_entropy = LossFunction.CrossEntropy()
	
	def test_loss_practical(self):
		y_true, y_hat = get_y_practicle()
		
		loss_hat = np.mean(self.cross_entropy.loss(y_true, y_hat))
		loss_true = log_loss(y_true, y_hat)
		diff = np.abs(loss_hat - loss_true)
		
		self.assertTrue(diff < MEAN_ERROR_THRESHOLD)
		
	def test_grad_practical(self):
		y_true, y_hat = get_y_practicle()
		
		grad_hat = self.cross_entropy.gradient(y_true, y_hat)
		grad_true = DerivativeApproximation.loss_gradient_approx(self.cross_entropy.loss, y_true, y_hat)
		diff = np.abs(grad_hat - grad_true)
		
		self.assertTrue(np.mean(diff) < MEAN_ERROR_THRESHOLD)
		
		
if __name__ == '__main__':
	unittest.main()
	