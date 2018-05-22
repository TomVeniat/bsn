import torch.nn.functional as F
from torch import nn, torch


def cross_entropy_sample(predictions, labels):
    # old_loss = -F.log_softmax(predictions, dim=1).gather(1, labels.view(-1, 1)).squeeze()
    return F.cross_entropy(predictions, labels, reduce=False)


def segmentation_cross_entropy(predictions, labels):
    bs = predictions.size(0)
    softmax_pred = nn.Softmax2d()(predictions)
    flatten_preds = softmax_pred.view(bs, predictions.size(1), -1)
    flatten_labels = labels.view(bs, 1, -1)
    individual_losses = -torch.log(flatten_preds).gather(1, flatten_labels).view(bs, -1).mean(1)
    return individual_losses
