import torch

from vesicle_seg.models.losses import combined_dice_focal
from vesicle_seg.models.residual_unet3d import ResidualUNet3d


def test_forward_shape():
    net = ResidualUNet3d(base=8, dropout=0.0)
    x = torch.zeros(2, 1, 8, 32, 32)
    y = net(x)
    assert y.shape == (2, 1, 8, 32, 32)


def test_loss_is_finite():
    net = ResidualUNet3d(base=8, dropout=0.0)
    x = torch.randn(1, 1, 8, 32, 32)
    target = torch.zeros(1, 1, 8, 32, 32)
    target[:, :, 2:5, 8:16, 8:16] = 1
    loss = combined_dice_focal(net(x), target)
    assert torch.isfinite(loss)
    loss.backward()
