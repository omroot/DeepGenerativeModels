from functools import partial
from einops import reduce
from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F

class WeightStandardizedConv2d(nn.Conv2d):
    """
    A class that extends nn.Conv2d to include weight standardization.

    Weight standardization is a technique that can work synergistically with group normalization.
    For more details, refer to the paper: https://arxiv.org/abs/1903.10520

    Methods
    -------
    forward(x: Tensor) -> Tensor:
        Performs the forward pass of the convolution operation with weight standardization.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the convolution operation with weight standardization.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The output tensor after applying the convolution operation with weight standardization.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )