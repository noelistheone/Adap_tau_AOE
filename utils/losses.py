import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Adap_tau_Loss(nn.Module):
    def __init__(self):
        super(Adap_tau_Loss, self).__init__()

    def forward(self, y_pred, temperature_, w_0):
        pos_logits = torch.exp(y_pred[:, 0] *  w_0)
        neg_logits = torch.exp(y_pred[:, 1:] * temperature_.unsqueeze(1))

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        pos_logits_ = torch.exp(y_pred[:, 0])
        neg_logits_ = torch.exp(y_pred[:, 1:])

        Ng_ = neg_logits_.sum(dim=-1)

        loss_ = (- torch.log(pos_logits_ / Ng_)).detach()
        return loss, loss_

class SSM_Loss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SSM_Loss, self).__init__()
        self._temperature = temperature
        print("Here is SSM LOSS: tau is {}".format(self._temperature))

    def forward(self, y_pred):
        pos_logits = torch.exp(y_pred[:, 0] / self._temperature)
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))

        return loss, loss.detach()
