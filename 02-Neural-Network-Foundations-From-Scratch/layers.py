# You are not allowed to import any other libraries or modules.
import torch
import torch.nn as nn

""" Fully Connected Layer """
    
class FCLayer(nn.Module):
    def __init__(self, num_input, num_output):
        """
        Initialize the Fully Connected (Linear) Layer.
      
        Args:
            num_input: Number of input features.
            num_output: Number of output features.
        """
        super(FCLayer, self).__init__()
        self.num_input = num_input
        self.num_output = num_output

        # Xavier initialization for weights
        self.W = nn.Parameter(torch.randn(num_input, num_output) * (2 / (num_input + num_output))**0.5)
        self.b = nn.Parameter(torch.zeros(1, num_output))
        self.X = None  # Store input for backward pass

    def forward(self, X):
        """
        Perform the forward pass.
        
        Args:
            X: Tensor of shape (batch_size, num_input), the input features.
        Returns:
            Tensor of shape (batch_size, num_output), the output after applying the linear transformation.
        """
        self.X = X  # Store input for backward pass
        return X @ self.W + self.b  # Linear transformation

    def backward(self, delta):
        """
        Perform the backward pass.
        
        Args:
            delta: Tensor of shape (batch_size, num_output), the gradient from the next layer.
        Returns:
            delta_next: Tensor of shape (batch_size, num_input), the gradient to pass to the previous layer.
        """
        # Gradient with respect to the input
        delta_next = delta @ self.W.t()
        # Gradient with respect to weights and bias
        self.W.grad = self.X.t() @ delta  # Gradient for weights
        self.b.grad = delta.sum(0, keepdim=True)  # Gradient for bias
        return delta_next


""" Sigmoid Layer """

class SigmoidLayer(nn.Module):
    def __init__(self):
        """
        Initialize the Sigmoid activation layer.
        """
        super(SigmoidLayer, self).__init__()
        self.Z = None  # Store output of the sigmoid for backward pass

    def forward(self, X):
        """
        Perform the forward pass using the Sigmoid function.
        
        Args:
            X: Tensor of shape (batch_size, num_features), the input features.
        Returns:
            Tensor of shape (batch_size, num_features), the output after applying the Sigmoid function.
        """
        self.Z = 1 / (1 + torch.exp(-X))  # Sigmoid function
        return self.Z

    def backward(self, delta):
        """
        Perform the backward pass.
        
        Args:
            delta: Tensor of shape (batch_size, num_features), the gradient from the next layer.
        Returns:
            delta_next: Tensor of shape (batch_size, num_features), the gradient to pass to the previous layer.
        """
        return delta * self.Z * (1 - self.Z)  # Element-wise product for gradient


""" ReLU Layer """

class ReLULayer(nn.Module):
    def __init__(self):
        """
        Initialize the ReLU activation layer.
        """
        super(ReLULayer, self).__init__()
        self.X = None  # Store input for backward pass

    def forward(self, X):
        """
        Perform the forward pass using the ReLU function.
        
        Args:
            X: Tensor of shape (batch_size, num_features), the input features.
        Returns:
            Tensor of shape (batch_size, num_features), the output after applying ReLU (max(0, x)).
        """
        self.X = X  # Store input for backward pass
        return X.clamp(min=0)  # Apply ReLU

    def backward(self, delta):
        """
        Perform the backward pass.
        
        Args:
            delta: Tensor of shape (batch_size, num_features), the gradient from the next layer.
        Returns:
            delta_next: Tensor of shape (batch_size, num_features), the gradient to pass to the previous layer.
        """
        return delta * (self.X > 0).float()  # Gradient is delta where X > 0


""" Dropout Layer """

class DropoutLayer(nn.Module):
    def __init__(self, dropout_rate):
        """
        Initialize the Dropout layer.
        
        Args:
            dropout_rate: The probability of dropping a neuron.
        """
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs):
        """
        Apply Dropout during training.
        Automatically disabled during evaluation.
        
        Args:
            inputs: Tensor of any shape, the input activations.
        Returns:
            out: Tensor of the same shape as inputs, with dropout applied in training mode.
        """
        if self.training:
            # Keep neurons with probability (1 - dropout_rate)
            self.mask = (torch.rand(*inputs.shape) > self.dropout_rate).float()
            out = inputs * self.mask / (1 - self.dropout_rate)  # Scale the outputs
        else:
            out = inputs  # No dropout during evaluation
        return out

    def backward(self, dout):
        """
        Perform the backward pass for (inverted) dropout.
        
        Args:
            dout: Upstream gradients of any shape.
        Returns:
            dout_next: Gradient with respect to the input x.
        """
        if self.training and self.mask is not None:
            return dout * self.mask / (1 - self.dropout_rate)  # Scale the upstream gradients
        else:
            return dout  # No scaling during evaluation