import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, inputs):
        return self.act(self.conv(inputs))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return inputs * self.fc(self.pool(inputs))


class StableResidualSEBlock(nn.Module):
    def __init__(self, channels, dilation=1, residual_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.se = SEBlock(channels)
        self.act_out = nn.SiLU(inplace=True)
        self.residual_scale = residual_scale
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs):
        output = self.act1(self.conv1(inputs))
        output = self.se(self.conv2(output))
        return self.act_out(inputs + self.residual_scale * output)


def conv_block(in_dim, middle_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, 3, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, 3, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def center_in(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, 3, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def center_out(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, 3, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, 3, stride=2, padding=1, output_padding=1),
    )


def up_conv_block(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels=11, num_classes=1, img_res=128, dropout=0.0):
        super().__init__()
        feats = 16
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_res = img_res
        self.dropout_p = dropout
        self.en3 = conv_block(in_channels, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(2)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(2)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)
        self.final = nn.Conv3d(feats * 2, num_classes, 3, padding=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, inputs):
        inputs = inputs["img"] if isinstance(inputs, dict) else inputs
        inputs = inputs.permute(0, 2, 1, 3, 4)
        en3 = self.en3(inputs)
        en4 = self.en4(self.pool_3(en3))
        center = self.center_out(self.center_in(self.pool_4(en4)))
        center = F.interpolate(center, size=en4.shape[2:], mode="trilinear", align_corners=True)
        dc4 = self.dc4(torch.cat([center, en4], dim=1))
        trans3 = self.trans3(dc4)
        trans3 = F.interpolate(trans3, size=en3.shape[2:], mode="trilinear", align_corners=True)
        output = self.final(self.dropout(self.dc3(torch.cat([trans3, en3], dim=1))))
        output = self.temporal_pool(output).squeeze(2)
        if output.shape[-2:] != (self.img_res, self.img_res):
            output = F.interpolate(output, (self.img_res, self.img_res), mode="bilinear", align_corners=True)
        return {"segmentation": output}
