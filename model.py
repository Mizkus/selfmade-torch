import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, padding=1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolingLayer(4, 4)

        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, padding=1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolingLayer(4, 4)

        flatten_size = (input_shape[0] // 16) * (input_shape[1] // 16) * conv2_channels
        self.flatten = Flattener()
        self.fc = FullyConnectedLayer(flatten_size, n_output_classes)
        self.softmax = softmax_with_cross_entropy

    def compute_loss_and_gradients(self, X, y):
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        loss, grad = self.softmax(out, y)

        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

        return loss

    def predict(self, X):
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        return np.argmax(out, axis=1)

    def params(self):
        result = {}
        layers = [self.conv1, self.conv2, self.fc]
        for layer in layers:
            for name, param in layer.params().items():
                result[name] = param
        return result
