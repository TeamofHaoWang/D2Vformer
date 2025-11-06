import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        RevIN: Reversible Instance Normalization

        :param num_features: Number of features or channels in the input tensor
        :param eps: A small value added for numerical stability in variance calculation
        :param affine: If True, RevIN includes learnable affine parameters
        :param subtract_last: If True, the last timestep is subtracted for normalization
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        # Initialize learnable affine parameters if affine is True
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Forward pass of the RevIN normalization.

        :param x: Input tensor to normalize or denormalize
        :param mode: The operation mode ('norm' for normalization, 'denorm' for denormalization)
        :return: Normalized or denormalized tensor
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError("Mode not implemented")
        return x

    def _init_params(self):
        """Initialize the learnable affine parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Compute mean and standard deviation for normalization.
        This function computes statistics along all dimensions except the feature dimension.
        """
        dim2reduce = tuple(range(1, x.ndim-1))  # Reduce all dimensions except for the feature dimension
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)  # Store the last timestep for subtraction if needed
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # Calculate mean across dimensions
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()  # Calculate stdev

    def _normalize(self, x):
        """Normalize the input tensor."""
        if self.subtract_last:
            x = x - self.last  # Subtract the last timestep if specified
        else:
            x = x - self.mean  # Subtract the mean across dimensions
        x = x / self.stdev  # Normalize by the standard deviation
        if self.affine:
            x = x * self.affine_weight  # Apply the learnable scaling factor
            x = x + self.affine_bias  # Apply the learnable bias
        return x

    def _denormalize(self, x):
        """Denormalize the input tensor."""
        if self.affine:
            x = x - self.affine_bias  # Subtract the learnable bias
            x = x / (self.affine_weight + self.eps * self.eps)  # Rescale using the affine weight
        x = x * self.stdev  # Rescale by the standard deviation
        if self.subtract_last:
            x = x + self.last  # Add the last timestep back if specified
        else:
            x = x + self.mean  # Add the mean back if needed
        return x
