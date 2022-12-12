import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_loss(nn.Module):
    def __init__(self, unmber_classes):
        super(CE_loss, self).__init__()
        self.unmber_class = unmber_classes
    def forward(self, yHat, y):
        P_i = torch.nn.functional.log_softmax(yHat, dim=1)
        y = torch.nn.functional.one_hot(y, num_classes = self.unmber_class)
        y = y.permute(0, 3, 1, 2)
        loss = y*(P_i + 0.000000000001)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss, dim=(0,1,2))
        hand_cross_entropy = -1*loss
        return hand_cross_entropy
