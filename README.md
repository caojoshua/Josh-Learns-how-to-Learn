# Josh-Learns-how-to-Learn (JLL)
A library to quickly build Deep Learning models inspired by Tensorflow and other libraries. Intended to help Josh better understand Deep Learning through implementation.

## Implementation details
The machine learning algorithms are all implemented by hand to help Josh deepen his understanding. Numpy is used for easy and efficient tensor operations, and sklearn is to verify function implementations (eg. cross entropy) through unit tests.

## Prerequisites
* python
* numpy
* sklearn

## Example usage
```
import NeuralNetwork as nn

# Xtrain is np.array of (num_samples, num_inputs) representing the inputs
# Ytrain is np.array of (num_samples) representing labels for corresponding inputs
# Xvalid and Yvalid are np.array validation sets with same shape as their training counterparts

model = nn.NeuralNetwork(input_length = Xtrain.shape[1], lr=0.01, cost=nn.LossFunction.CrossEntropy())
model.add_layer(nn.Layer.FullyConnectedLayer, num_neurons = 32)
model.add_layer(nn.Layer.Relu)
model.add_layer(nn.Layer.FullyConnectedLayer, num_neurons = 10)
model.add_layer(nn.Layer.Softmax)

model.train(Ytrain, Xtrain, Xva, Yva, epochs=100, batch_size=32)
```
The python notebooks contain more detailed examples.
