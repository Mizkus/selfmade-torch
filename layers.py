import numpy as np

def im2col(X, filter_size, padding=0):
    batch_size, height, width, channels = X.shape

    out_height = height - filter_size + 2 * padding + 1
    out_width = width - filter_size + 2 * padding + 1

    X_padded = np.pad(X, [(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='constant')
    
    col = np.zeros((batch_size, out_height, out_width, filter_size, filter_size, channels))

    for y in range(filter_size):
        for x in range(filter_size):
            col[:, :, :, y, x, :] = X_padded[:, y:y + out_height, x:x + out_width, :]

    col = col.transpose(0, 3, 4, 1, 2, 5).reshape(batch_size * out_height * out_width, -1)
    return col
    

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    shifted_predictions = predictions - np.max(predictions, axis=1, keepdims=True)
    
    exp_preds = np.exp(shifted_predictions)
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    batch_size = predictions.shape[0]
    correct_logprobs = -np.log(probs[np.arange(batch_size), target_index])
    loss = np.sum(correct_logprobs) / batch_size
    
    dprediction = probs.copy()
    dprediction[np.arange(batch_size), target_index] -= 1
    dprediction /= batch_size
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        d_result = d_out * (self.X > 0).astype(float)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        d_input = d_out.dot(self.W.value.T)
        
        self.W.grad = self.X.T.dot(d_out)
        
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        
        self.X = X
        
        batch_size, height, width, channels = X.shape
        assert channels == self.in_channels

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        padded_X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for h in range(out_height):
            for w in range(out_width):
                window = padded_X[:, h:h+self.filter_size, w:w+self.filter_size, :]
                out[:, h, w, :] = np.sum(window[:, :, :, :, None] * self.W.value[None, :, :, :, :], axis=(1, 2, 3))
        
        out += self.B.value

        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        padded_X = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        padded_dX = np.zeros_like(padded_X)

        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

        for h in range(out_height):
            for w in range(out_width):
                window = padded_X[:, h:h+self.filter_size, w:w+self.filter_size, :]
                for c in range(self.out_channels):
                    self.W.grad[:, :, :, c] += np.sum(window * d_out[:, h, w, c][:, None, None, None], axis=0)
                for n in range(batch_size):
                    for c in range(self.out_channels):
                        padded_dX[n, h:h+self.filter_size, w:w+self.filter_size, :] += self.W.value[:, :, :, c] * d_out[n, h, w, c]
        
        self.B.grad = np.sum(d_out, axis=(0, 1, 2))

        if self.padding > 0:
            dX = padded_dX[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = padded_dX

        return dX

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size

        out = np.zeros((batch_size, out_height, out_width, channels))
        out_X = np.zeros_like(X)

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size
                    w_start = w * self.pool_size
                    w_end = w_start + self.pool_size
                    window = X[b, h_start:h_end, w_start:w_end, :]
                    max_vals = np.max(window, axis=(0, 1), keepdims=True)
                    mask = (window == max_vals)
                    out[b, h, w, :] = max_vals.squeeze()
                    out_X[b, h_start:h_end, w_start:w_end, :] = mask

        self.X = X
        self.out_X = out_X

        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape 

        dX = np.zeros_like(self.X)

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.pool_size
                    h_end = h_start + self.pool_size
                    w_start = w * self.pool_size
                    w_end = w_start + self.pool_size
                    mask = self.out_X[b, h_start:h_end, w_start:w_end, :]
                    dX[b, h_start:h_end, w_start:w_end, :] += mask * d_out[b, h, w, :]

        return dX

    def params(self):
        return {}

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape
        
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
