
import numpy as np

def derivative_approx(f, x, difference='', h=1e-7):
	"""
	Approximates the derivative of a single variable function using the finite difference method.
	
	Parameters:
		f
			callable function to approximate the gradient of. Must accept
			y_true and x parameters
		x
			an np.array of any size
		difference
			'backward' : backward finite difference
			'forward' : forward finite difference
			other : central finite difference on unrecognized option
		h
			small value h used in the finite difference formula
	Returns:
		An np.array A with same size as x where each element is the derivative of f(x)
	"""
	forward_f = f(x + h)
	central_f = f(x)
	backward_f = f(x-h)
	if difference == 'backward':
		return (central_f - backward_f) / h
	elif difference == 'forward':
		return (forward_f - central_f) / h
	else:
		return (forward_f - backward_f) / (2 * h)
	
def gradient_approx(f, x, difference='', h=1e-7):
	"""
	Approximates gradient of a multivariable function using the finite difference method. 
	
	Parameters:
		f
			callable function to approximate the gradient of. Must accept
			y_true and x parameters
		x
			an np.array of size (num_samples x num_variables). The gradient vectors will be
			computed wrt each variable in the vector of (num_variables)
		difference
			'backward' : backward finite difference
			'forward' : forward finite difference
			other : central finite difference on unrecognized option
		h
			small value h used in the finite difference formula
			
	Returns:
		A (num_samples x num_variables) where each row is the gradient vector
		for the nth sample and each column is the gradient wrt the mth variable
		
	Control Flow:
		1. for each sample vector (M), duplicate it to a sample matrix (N, N)
		2. add h to each diagonal, which will allow us to compute the gradient wrt each variable
		3. compute the finite difference gradient estimate against each row of the sample matrices,
			shrinking the (N,N) matrix to (N)
	"""

	# setup up proper repeats/dimensions
	x = np.expand_dims(x, axis=-2)
	x_forward = x.copy()
	x_forward = x_forward.repeat(repeats=x_forward.shape[-1], axis=-2)
	diag_index = np.arange(x_forward.shape[-1])
	
	# compute forward and backward f
	# could optimize this in if-else block, but this code is cleaner
	x_backward = x_forward.copy()
	x_forward[:,diag_index, diag_index] += h
	x_backward[:,diag_index, diag_index] -= h
	forward_f = f(x_forward)
	central_f = f(x)
	backward_f = f(x_backward)
	# compute gradient esimate and return
	if difference == 'backward':
		return (central_f - backward_f) / h
	elif difference == 'forward':
		return (forward_f - central_f) / h
	else:
		return (forward_f - backward_f) / (2 * h)
	

def loss_gradient_approx(f, y_true, y_hat, **kwargs):
	return gradient_approx(lambda x : f(np.expand_dims(y_true, -2), x), y_hat, **kwargs)
