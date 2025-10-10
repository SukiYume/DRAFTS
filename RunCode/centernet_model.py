import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv   = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):

        x1        = self.up(x1)
        if x2 is not None:
            x     = torch.cat([x2, x1], dim=1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1    = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        else:
            x = x1
        x = self.conv(x)

        return x


class centernet(nn.Module):

    def __init__(self, n_classes=1, model_name='resnet18'):
        super(centernet, self).__init__()

        model_dict = {
            'resnet18':   [torchvision.models.resnet18(weights=None), 512],
            'resnet34':   [torchvision.models.resnet34(weights=None), 512],
            'resnet50':   [torchvision.models.resnet50(weights=None), 2048],
            'resnet50v2': [torchvision.models.wide_resnet50_2(weights=None), 2048]
        }

        basemodel, num_ch = model_dict[model_name]
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        self.base_model = basemodel
        self.up1  = up(num_ch, 512)
        self.up2  = up(512, 256)
        self.up3  = up(256, 256)
        self.class_head  = nn.Conv2d(256, n_classes, 1)
        self.wh_head     = nn.Conv2d(256, 2, 1)
        self.offset_head = nn.Conv2d(256, 2, 1)

    def forward(self, x):

        x      = self.base_model(x)
        x      = self.up1(x)
        x      = self.up2(x)
        x      = self.up3(x)
        hm     = self.class_head(x).sigmoid_()
        wh     = self.wh_head(x)
        offset = self.offset_head(x)

        return hm, wh, offset


def focal_loss(pred, target):

    pos_inds    = target.eq(1).float()
    neg_inds    = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pred     = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = - neg_loss
    else:
        loss = - (pos_loss + neg_loss) / num_pos

    return loss


def reg_l1_loss(pred, target, mask):

    pred = pred.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)

    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)

    return loss


def centerloss(pred, targ):

    hm, wh, offset = pred[:, 0], pred[:, 1:3], pred[:, 3:]
    hm_gt, wh_gt, reg_gt, reg_mask = targ[:, 0], targ[:, 1:3], targ[:, 3:5], targ[:, 5]

    cls_loss = focal_loss(hm, hm_gt)
    wh_loss  = reg_l1_loss(wh, wh_gt, reg_mask) * 0.1
    off_loss = reg_l1_loss(offset, reg_gt, reg_mask)
    loss     = cls_loss + wh_loss + off_loss

    return loss, cls_loss, wh_loss + off_loss
