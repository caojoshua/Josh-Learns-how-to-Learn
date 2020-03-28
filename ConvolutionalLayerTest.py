
import unittest
import Layer
import numpy as np
import DerivativeApproximation
    
ALL_ERROR_THRESHOLD = 1e-1
MEAN_ERROR_THRESHOLD = 1e-3
    
NUM_SAMPLES = 100
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 30
NUM_CHANNELS = 3
NUM_FILTERS = 8
FILTER_SIZE= (2,4)

h = 1e-5
    
class TestConvolutionalLayer(unittest.TestCase):
    
    def setUp(self):
        self.conv = Layer.ConvolutionalLayer((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), 
                                             num_filters = NUM_FILTERS,
                                             filter_size = FILTER_SIZE)
        self.test_input = np.random.random(size = (NUM_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
    
    def test_gradient_wrt_weight(self):
        grad_hat = self.conv.compute_gradient_wrt_weight(self.test_input)
        
        grad_true = np.zeros(NUM_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FILTERS)
        for h in IMAGE_HEIGHT:
            for w in IMAGE_WIDTH:
                for channel in NUM_CHANNELS:
                    self.test_input[:,h,w,channel] += h
                    g = self.conv.forward_propagate(self.test_input)
                    self.test_input[:,h,w,channel] -= h
                     
    
#         grad_true = DerivativeApproximation.gradient_approx(lambda x : self.conv.forward_propagate(x), self.test_input)
        diff = np.abs(grad_hat - grad_true)
        
        
        
if __name__ == '__main__':
    unittest.main()
    