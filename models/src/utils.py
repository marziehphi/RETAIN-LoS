from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    To expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log().where(mask, torch.zeros_like(y))
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log().where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


# Mean Squared Error (MSE) loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the predictions corresponding to no data should be set to 0
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        # then we set the labels that correspond to no data to be 0 as well
        y = y.where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


class BatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # hack to work around model.eval() issue
        if not self.training:
            self.eval_momentum = 0  # set the momentum to zero when the model is validating

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum if self.training else self.eval_momentum

        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training=True, momentum=exponential_average_factor, eps=self.eps)  # set training to True so it calculates the norm of the batch


class BatchNorm1d(BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
