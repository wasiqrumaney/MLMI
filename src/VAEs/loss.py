import torch
from torch import nn
from torch.nn import functional as F


## https://tuatini.me/practical-image-segmentation-with-unet/
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        smooth = 1.
        num = targets.size(0)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num    #average over batches?
        return score


class LossFunction(object):
    def __init__(self, c_rec=1, c_kld=1, loss='dice'):
        self.c_rec = c_rec
        self.c_kld = c_kld
        if loss is 'dice':
            self.rec_loss_func = SoftDiceLoss()
        elif loss is 'bce':
            self.rec_loss_func = nn.BCELoss(reduction='elementwise_mean')
        else:
            raise Exception("wrong value for 'type'")

    def __call__(self, recon_x, x, mu, logvar,):
        batch_size = recon_x.size(0)
        vol_size = x.size(2)

        rec_loss = self.rec_loss_func(recon_x, x)

        # measure of distance btw learned distribution and Normal Gaussian
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= batch_size * vol_size * vol_size * vol_size

        rec_loss *= self.c_rec
        KLD *= self.c_kld

        loss = rec_loss + KLD
        return loss, rec_loss, KLD
