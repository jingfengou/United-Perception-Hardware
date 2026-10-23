from up_platform import get_platform
from up_platform import layers as platform_layers
import torch
import torch.nn as nn
import torch.nn.functional as F

from up.utils.general.registry_factory import MIMIC_LOSS_REGISTRY


@MIMIC_LOSS_REGISTRY.register('kd_loss')
class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network, NIPS2014.
    https://arxiv.org/pdf/1503.02531.pdf
    """
    def __init__(self, T=1, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.t = T

    def single_kl(self, s_preds, t_preds, mask=None):
        if mask is not None:
            if mask.sum() > 0:
                p = get_platform().category('activation').log_softmax(s_preds / self.t, dim=1)[mask]
                q = get_platform().category('activation').softmax(t_preds / self.t, dim=1)[mask]
                l_kl = get_platform().category('loss').kl_div(p, q, reduce=False)
                loss = torch.sum(l_kl)
                loss = loss / mask.sum()
            else:
                loss = get_platform().device.move(torch.Tensor([0]))
        else:
            p = get_platform().category('activation').log_softmax(s_preds / self.t, dim=1)
            q = get_platform().category('activation').softmax(t_preds / self.t, dim=1)
            l_kl = get_platform().category('loss').kl_div(p, q, reduce=False)
            loss = l_kl.sum() / l_kl.size(0)
        return loss * (self.t**2)

    def forward(self, s_preds, t_preds, masks=None):
        if masks is not None:
            assert isinstance(masks, list) and len(masks) == len(s_preds), 'masks must be consistent with preds!'
        else:
            masks = [None for _ in range(len(s_preds))]

        loss = 0
        for idx, (s, t) in enumerate(zip(s_preds, t_preds)):
            loss += self.single_kl(s, t, masks[idx])
        return loss * self.loss_weight


@MIMIC_LOSS_REGISTRY.register('gkd_loss')
class GKDLoss(nn.Module):
    """
    Guided Knowledge Distillation Loss
    Use correctly classified soft labels to guide student.
    """

    def __init__(self, T=1, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.t = T

    def forward(self, s_preds, t_preds, **kwargs):
        label = kwargs['label']
        loss = 0
        for s_pred, t_pred in zip(s_preds, t_preds):
            s = get_platform().category('activation').log_softmax(s_pred / self.t, dim=1)
            t = get_platform().category('activation').softmax(t_pred / self.t, dim=1)
            t_argmax = torch.argmax(t, dim=1)
            mask = torch.eq(label, t_argmax).float()
            count = (mask[mask == 1]).size(0)
            mask = mask.unsqueeze(-1)
            correct_s = s.mul(mask)
            correct_t = t.mul(mask)
            correct_t[correct_t == 0.0] = 1.0

            loss += get_platform().category('loss').kl_div(correct_s, correct_t, reduction='sum') * (self.t**2) / count
        return loss * self.loss_weight


@MIMIC_LOSS_REGISTRY.register('kl_loss')
class KLLoss(nn.Module):
    """
    KL Divergence loss
    """
    def __init__(self, norm='softmax', loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.norm = norm

    def forward(self, s_features, t_features, **kwargs):
        loss = 0
        for s, t in zip(s_features, t_features):
            loss += self.kl(s, t)
        return loss * self.loss_weight

    def kl(self, pred_feas, target_feas):
        crit = platform_layers.KLDivLoss(reduction='batchmean')
        relu = platform_layers.ReLU()
        s = relu(pred_feas)
        t = relu(target_feas)
        if self.norm == 'softmax':
            s = get_platform().category('activation').log_softmax(s, dim=1)
            t = get_platform().category('activation').softmax(t, dim=1)
            t.detach_()
            # sum(y*(logy-logx))
            loss = crit(s, t)
        elif self.norm == 'l2':
            # sum(y*log(y/x))
            # use KLDivLoss for l2 norm may cause overflow
            loss = torch.sum(t / torch.sum(t) * (torch.log((t / torch.sum(t) + 1e-6) / (s / torch.sum(s) + 1e-6))))
        else:
            print('Invalid Normalization Function: {}'.format(self.norm))
            return None

        return loss
