
import numpy as np

def flatten_axis(arr, axis=1, reverse=False):
    """
    flatten arr across axis 1:N
    or N:arr.ndim exclusive if reverse = True
    """
    if reverse:
        return np.reshape(arr, (np.prod(arr.shape[:axis]),) + arr.shape[axis:])
    return np.reshape(arr, arr.shape[0:axis] + (np.prod(arr.shape[axis:]),))
