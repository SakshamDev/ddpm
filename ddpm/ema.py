"""
Exponential Moving Average (EMA) for model weights.

In diffusion models, sampling from an EMA of the training weights produces
significantly better, smoother images than sampling from the raw, fluctuating
training weights.
"""

import copy
import torch
import torch.nn as nn

class EMA:
    def __init__(self, beta: float = 0.9999):
        """
        Args:
            beta: The decay rate. A value of 0.9999 means the EMA weight is composed of
                  99.99% of the old EMA weight and 0.01% of the new model weight.
        """
        self.beta = beta

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module):
        """
        Updates the parameters of ema_model towards current_model.
        """
        with torch.no_grad():
            for ema_param, current_param in zip(ema_model.parameters(), current_model.parameters()):
                # If parameter doesn't require grad (like buffers), just copy it
                if not current_param.requires_grad:
                    ema_param.data.copy_(current_param.data)
                else:
                    ema_param.data.mul_(self.beta).add_(current_param.data, alpha=1.0 - self.beta)
                    
    def create_ema_model(self, model: nn.Module) -> nn.Module:
        """
        Creates a deepcopy of the model to be used as the EMA model.
        It detaches all parameters and sets them to not require gradients.
        """
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model
