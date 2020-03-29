# Josh-Learns-how-to-Learn (JLL)
A library to quickly build Deep Learning models inspired by Tensorflow and other libraries. Intended to help Josh better understand Deep Learning through implementation.

## Implementation details
The machine learning algorithms are all implemented by hand to help Josh deepen his understanding. Numpy is used for easy and efficient tensor operations, and sklearn is to verify function implementations (eg. cross entropy) through unit tests.

## Prerequisites
* python3
* python packages:
  * numpy
  * sklearn
  * nose (if running unit tests)

## Setup
Install the package from the root directory
```
pip install .
```
To run unit tests
```
python setup.py test
```

## Example usage
```
import JoshLearnsHowToLearn as jll

# Xtrain is np.array of (num_samples, image_dimensions) representing the inputs
# Ytrain is np.array of (num_samples) representing labels for corresponding inputs
# Xvalid and Yvalid are np.array validation sets with same shape as their training counterparts

model = jll.NeuralNetwork.NeuralNetwork(input_shape = Xtr.shape[1:], lr=0.01, loss=jll.LossFunction.CrossEntropy())
model.add_layer(jll.Layer.ConvolutionalLayer, filter_size = (2,2), num_filters = 8)
model.add_layer(jll.Layer.Relu)
model.add_layer(jll.Layer.ConvolutionalLayer, filter_size = (2,2), num_filters = 4)
model.add_layer(jll.Layer.Relu)
model.add_layer(jll.Layer.FlattenLayer)
model.add_layer(jll.Layer.FullyConnectedLayer, num_neurons = 16)
model.add_layer(jll.Layer.Relu)
model.add_layer(jll.Layer.FullyConnectedLayer, num_neurons = 10)
model.add_layer(jll.Layer.Softmax)

model.train(Ytrain, Xtrain, Xva, Yva, epochs=100, batch_size=32)
```
The `examples/` directory includes some examples. 
