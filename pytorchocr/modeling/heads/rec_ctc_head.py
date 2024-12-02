import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 return_logit=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,)
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.return_logit = return_logit


    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = [x, predicts]
        else:
            result = predicts

        if not self.training:
            if not self.return_logit:
                if self.return_feats:
                    result[1] = F.softmax(result[1], dim=2)
                else:
                    result = F.softmax(result, dim=2)
            

        return result