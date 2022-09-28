# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    inputs = torch.sigmoid(pred)

    inputs = inputs > 0.5

    inputs = inputs.view(-1)
    targets = target.view(-1)

    precision = (inputs * targets).sum() / (inputs.sum() + 1e-6)
    recall = (inputs * targets).sum() / (targets.sum() + + 1e-6)

    f_measure = (2. * precision * recall + 1e-6) / (precision + recall + 1e-6)

    return f_measure


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1, ), thresh=None, ignore_index=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = 0

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh,
                        self.ignore_index)
