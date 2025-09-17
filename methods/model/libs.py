import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """

        def _get_act_fn(act_name, inplace=True):
            if act_name == "relu":
                return nn.ReLU(inplace=inplace)
            elif act_name == "leaklyrelu":
                return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
            elif act_name == "gelu":
                return nn.GELU()
            elif act_name == "sigmoid":
                return nn.Sigmoid()
            else:
                raise NotImplementedError
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))
def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
#金字塔挤压注意(PSA)模块
class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
            """standard convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, groups=groups, bias=False)
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1) #size[3,4,16,128,128]
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])#重塑 size[3,64,128,128]
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out
class FSP(nn.Module):
    def __init__(self, in_dim,BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.conv_l_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.process_m = nn.Sequential(
            PSAModule(in_dim, in_dim),
            BatchNorm(in_dim)
        )
        self.process_l = nn.Sequential(
            PSAModule(in_dim, in_dim),
            BatchNorm(in_dim)
        )
        self.up = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            BatchNorm(in_dim)
        )
    def forward(self, l, m):
        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, m.shape[2:]) + F.adaptive_avg_pool2d(l, m.shape[2:])
        l = self.conv_l(l)
        m = self.conv_m(m)
        process_l=self.process_l(l)
        process_m=self.process_m(m)
        sim_map = torch.sigmoid(self.up(process_l * process_m))
        out = (1 - sim_map) * l + sim_map * m
        return out

class GRM_num(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32, num_layers=6):
        super(GRM_num, self).__init__()
        self.in_ch = in_ch
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.gate = None
        self.fuse = None
        self.norm_in = None
        self.norm_out = None
        self._build_layers()

    def _build_layers(self):
        in_ch_ = self.in_ch
        for i in range(self.num_layers):
            layer = ConvBNReLU(in_ch_, self.growth_rate, 3, padding=1, dilation=1)
            self.layers.append(layer)
            in_ch_ += self.growth_rate

        self.fuse = ConvBNReLU(in_ch_, self.in_ch, 3, padding=1)
        self.norm_in = nn.GroupNorm(num_groups=1, num_channels=self.in_ch)
        self.norm_out = ConvBNReLU(self.in_ch, self.in_ch, 3, 1, 1)

        # Gate mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch_, in_ch_, 1),
            nn.ReLU(True),
            nn.Conv2d(in_ch_, in_ch_, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.norm_in(x)
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = torch.cat(features, dim=1)

        gate = self.gate(x)
        x = self.fuse(x * gate)
        out = x + self.norm_out(x)

        return out
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = GRM_num(64)
    input1 = torch.rand(3, 64, 64, 64)
    input2 = torch.rand(3, 64, 128, 128)
    output = block(input1)
    print(output.size())
    total_params = sum(p.numel() for p in block.parameters())

    # 打印参数数量
    print(f"Total number of parameters: {total_params}")
