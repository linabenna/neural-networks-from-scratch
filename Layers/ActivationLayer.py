import numpy as np  # import numpy library

class SigmoidLayer:
    """
    This file implements activation layers
    inline with a computational graph model
    Args:
        shape: shape of input to the layer
    Methods:
        forward(Z)
        backward(upstream_grad)
    """

    def __init__(self, shape):
        """
        The consturctor of the sigmoid/logistic activation layer takes in the following arguments
        Args:
            shape: shape of input to the layer
        """
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        """
        This function performs the forwards propagation step through the activation function
        Args:
            Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        """
        This function performs the  back propagation step through the activation function
        Local gradient => derivative of sigmoid => A*(1-A)
        Args:
            upstream_grad: gradient coming into this layer from the layer above
        """
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        self.dZ = upstream_grad * self.A*(1-self.A)

class SoftmaxLayer:
    def __init__(self, Z_shape):
        self.Z = np.zeros(Z_shape)
        self.A = np.zeros(Z_shape)
        self.dZ = np.zeros(Z_shape)
    
    def forward(self, Z):
        self.Z = Z
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stability
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def backward(self, dA):
        # Softmax derivative is handled directly in the cross-entropy backward
        self.dZ = dA

class ReLULayer:
    def __init__(self, shape):
        self.Z = np.zeros(shape)
        self.A = np.zeros(shape)
        self.dZ = np.zeros(shape)

    def forward(self, Z):
        self.Z = Z
        self.A = np.maximum(0, Z)

    def backward(self, dA):
        self.dZ = dA * (self.Z > 0).astype(float)
