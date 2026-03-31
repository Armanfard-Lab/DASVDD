"""PIMA autoencoder used by DASVDD."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PimaAutoencoder(nn.Module):
    """Autoencoder architecture used for the PIMA dataset."""

    code_size = 4

    def __init__(self, input_shape: int):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=10)
        self.encoder_middle_layer = nn.Linear(in_features=10, out_features=4)
        self.encoder_output_layer = nn.Linear(in_features=4, out_features=self.code_size)
        self.decoder_hidden_layer = nn.Linear(in_features=self.code_size, out_features=4)
        self.decoder_middle_layer = nn.Linear(in_features=4, out_features=10)
        self.decoder_output_layer = nn.Linear(in_features=10, out_features=input_shape)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs and reconstruct them."""

        activation = F.leaky_relu(self.encoder_hidden_layer(features))
        activation = F.leaky_relu(self.encoder_middle_layer(activation))
        code = F.leaky_relu(self.encoder_output_layer(activation))
        activation = F.leaky_relu(self.decoder_hidden_layer(code))
        activation = F.leaky_relu(self.decoder_middle_layer(activation))
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed, code


AE_PIMA = PimaAutoencoder
