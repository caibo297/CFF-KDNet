import abc
import logging

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.efficientnet import EfficientNet
from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b4
from ..backbone.res2net_v1b import res2net50_v1b
from .ops import ConvBNReLU, PixelNormalizer, resize_to
from .libs import FSP,GRM_num
from .PC_loss import PCLoss

LOGGER = logging.getLogger("main")

class CFFKDNet(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self):
        pass

    def forward(self, data, iter_percentage=1, **kwargs):
        logits = self.body(data=data)

        if self.training:
            mask = data["mask"]
            logits = resize_to(logits, tgt_hw=mask.shape[-2:])
            prob = logits.sigmoid()
            losses = []
            loss_str = []

            sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")
            return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
        else:
            return logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups
class CFFKDNet_RES2NET50(CFFKDNet):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        use_checkpoint=False,
    ):
        super().__init__()
        self.encoder = res2net50_v1b()
        if pretrained:
            checkpoint = torch.load('./weight/res2net50.pth')
            self.encoder.load_state_dict(checkpoint, strict=False)

        # Feature transformation layers
        self.tra_4 = ConvBNReLU(2048, mid_dim, 3, 1, 1)
        self.FSP_4 = FSP(mid_dim)
        self.GRM_4 = GRM_num(mid_dim, 64, 8)

        self.tra_3 = ConvBNReLU(1024, mid_dim, 3, 1, 1)
        self.FSP_3 = FSP(mid_dim)
        self.GRM_3 = GRM_num(mid_dim, 64, 8)

        self.tra_2 = ConvBNReLU(512, mid_dim, 3, 1, 1)
        self.FSP_2 = FSP(mid_dim)
        self.GRM_2 = GRM_num(mid_dim, 64, 8)

        self.tra_1 = ConvBNReLU(256, mid_dim, 3, 1, 1)
        self.FSP_1 = FSP(mid_dim)
        self.GRM_1 = GRM_num(mid_dim, 64, 8)

        self.tras = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = res2net50_v1b(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features[0]
        c3 = features[1]
        c4 = features[2]
        c5 = features[3]

        return c2, c3, c4, c5

    def body(self, data):

        m_trans_feats = self.normalize_encoder(data["image_m"])
        l_trans_feats = self.normalize_encoder(data["image_l"])



        m, l = (
            self.tra_4(m_trans_feats[3]),
            self.tra_4(l_trans_feats[3]),
        )
        md4 = self.FSP_4(l, m)
        x = self.GRM_4(md4)

        m, l = (
            self.tra_3(m_trans_feats[2]),
            self.tra_3(l_trans_feats[2]),
        )
        md3 = self.FSP_3(l, m)
        x = self.GRM_3(md3 + resize_to(x, tgt_hw=md3.shape[-2:]))

        m, l = (
            self.tra_2(m_trans_feats[1]),
            self.tra_2(l_trans_feats[1]),
        )
        md2 = self.FSP_2(l, m)
        x = self.GRM_2(md2 + resize_to(x, tgt_hw=md2.shape[-2:]))

        m, l = (
            self.tra_1(m_trans_feats[0]),
            self.tra_1(l_trans_feats[0]),
        )
        md1 = self.FSP_1(l, m)
        x = self.GRM_1(md1 + resize_to(x, tgt_hw=md1.shape[-2:]))

        x = self.tras(x)
        return self.predictor(x)
class CFFKDNet_PVTB2(CFFKDNet):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)
        self.embed_dims = self.encoder.embed_dims

        # Feature transformation layers
        self.tra_4 = ConvBNReLU(self.embed_dims[3], mid_dim, 3, 1, 1)
        self.FSP_4 = FSP(mid_dim)
        self.GRM_4 = GRM_num(mid_dim, 64, 8)

        self.tra_3 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.FSP_3 = FSP(mid_dim)
        self.GRM_3 = GRM_num(mid_dim, 64, 8)

        self.tra_2 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.FSP_2 = FSP(mid_dim)
        self.GRM_2 = GRM_num(mid_dim, 64, 8)

        self.tra_1 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.FSP_1 = FSP(mid_dim)
        self.GRM_1 = GRM_num(mid_dim, 64, 8)

        self.tras = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]

        return c2, c3, c4, c5

    def body(self, data):

        m_trans_feats = self.normalize_encoder(data["image_m"])
        l_trans_feats = self.normalize_encoder(data["image_l"])

        m, l = (
            self.tra_4(m_trans_feats[3]),
            self.tra_4(l_trans_feats[3]),
        )
        md4 = self.FSP_4(l, m)
        x = self.GRM_4(md4)

        m, l = (
            self.tra_3(m_trans_feats[2]),
            self.tra_3(l_trans_feats[2]),
        )
        md3 = self.FSP_3(l, m)
        x = self.GRM_3(md3 + resize_to(x, tgt_hw=md3.shape[-2:]))

        m, l = (
            self.tra_2(m_trans_feats[1]),
            self.tra_2(l_trans_feats[1]),
        )
        md2 = self.FSP_2(l, m)
        x = self.GRM_2(md2 + resize_to(x, tgt_hw=md2.shape[-2:]))

        m, l = (
            self.tra_1(m_trans_feats[0]),
            self.tra_1(l_trans_feats[0]),
        )
        md1 = self.FSP_1(l, m)
        x = self.GRM_1(md1 + resize_to(x, tgt_hw=md1.shape[-2:]))

        x = self.tras(x)
        return self.predictor(x)
class CFFKDNet_PVTB4(CFFKDNet_PVTB2):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b4(pretrained=pretrained, use_checkpoint=use_checkpoint)

class PVTB2_KD(CFFKDNet_PVTB2):
    """Student model that keeps teacher as a class variable instead of instance variable"""
    _teacher_model = None

    @classmethod
    def set_teacher(cls, teacher):
        cls._teacher_model = teacher
        if cls._teacher_model is not None:
            cls._teacher_model.eval()
            for param in cls._teacher_model.parameters():
                param.requires_grad = False

    def __init__(
            self,
            pretrained=True,
            num_frames=1,
            input_norm=True,
            mid_dim=64,
            use_checkpoint=False,
    ):
        super().__init__(pretrained, num_frames, input_norm, mid_dim, use_checkpoint)
        self.T = 10  # Temperature for distillation

        # PC distillation
        self.pc_loss = PCLoss()
        self.pc_weight = 5

        self._register_hooks_multi()
        self.student_features = {}
        self.teacher_features = {}

    def _register_hooks_multi(self):


        def get_student_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'student_features'):
                    self.student_features = {}
                self.student_features[name] = output

            return hook

        def get_teacher_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'teacher_features'):
                    self.teacher_features = {}
                self.teacher_features[name] = output

            return hook

        # 注册多个层的hook
        self.FSP_4.register_forward_hook(get_student_features('layer4'))
        self.FSP_3.register_forward_hook(get_student_features('layer3'))
        self.FSP_2.register_forward_hook(get_student_features('layer2'))
        # self.FSP_1.register_forward_hook(get_student_features('layer1'))

        if self._teacher_model is not None:
            self._teacher_model.FSP_4.register_forward_hook(get_teacher_features('layer4'))
            self._teacher_model.FSP_3.register_forward_hook(get_teacher_features('layer3'))
            self._teacher_model.FSP_2.register_forward_hook(get_teacher_features('layer2'))
            # self._teacher_model.FSP_1.register_forward_hook(get_teacher_features('layer1'))
    def compute_distillation_loss(self, student_output, teacher_output):
        alpha = 0.6
        # MSE loss
        soft_student_mse = (student_output / self.T).sigmoid()
        soft_teacher_mse = (teacher_output / self.T).sigmoid()
        mse_loss = F.mse_loss(soft_student_mse, soft_teacher_mse)

        # KL loss
        B, C, H, W = student_output.shape
        student_output = student_output.view(B, C, -1)
        teacher_output = teacher_output.view(B, C, -1)

        soft_student_kl = F.log_softmax(student_output / self.T, dim=2)
        soft_teacher_kl = F.softmax(teacher_output / self.T, dim=2)

        kl_loss = F.kl_div(soft_student_kl, soft_teacher_kl, reduction='batchmean') * (self.T ** 2)
        # Combined loss
        distillation_loss = alpha * mse_loss + (1 - alpha) * kl_loss * 0.1
        return distillation_loss, alpha * mse_loss, (1 - alpha) * kl_loss * 0.1

    def forward(self, data, iter_percentage=1, **kwargs):
        student_logits = self.body(data)
        prob = student_logits.sigmoid()

        if self.training and self._teacher_model is not None:
            mask = data["mask"]
            losses = []
            loss_str = []

            # 1. BCE loss
            sod_loss = F.binary_cross_entropy_with_logits(input=student_logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            # 2. UAL loss
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

            # 3. Distillation loss
            with torch.no_grad():
                teacher_logits = self._teacher_model.body(data)
            distillation_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            losses.append(distillation_loss[0])
            loss_str.append(f"total_loss: {distillation_loss[0].item():.5f}")
            loss_str.append(f"mse_loss: {distillation_loss[1].item():.5f}")
            loss_str.append(f"kl_loss: {distillation_loss[2].item():.5f}")

            #PC distillation multi
            if hasattr(self, 'student_features') and hasattr(self, 'teacher_features'):
                pc_loss_total = 0
                for layer_name in self.student_features.keys():
                    pc_loss_layer = self.pc_loss(
                        self.student_features[layer_name],
                        self.teacher_features[layer_name],
                        mask
                    )
                    pc_loss_total += pc_loss_layer
                    loss_str.append(f"pc_distill_{layer_name}: {pc_loss_layer.item():.5f}")

                pc_weighted_loss = self.pc_weight * pc_loss_total
                losses.append(pc_weighted_loss)
            else:
                loss_str.append("pc_distill: features not available")
            total_loss = sum(losses)
            return dict(
                vis=dict(sal=prob),
                loss=total_loss,
                loss_str=" ".join(loss_str)
            )
        else:
            return student_logits

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # 确保不包含任何teacher相关的参数
        return state_dict

class Res2Net50_KD(CFFKDNet_RES2NET50):
    """Student model that keeps teacher as a class variable instead of instance variable"""
    _teacher_model = None  # 类变量，所有实例共享

    @classmethod
    def set_teacher(cls, teacher):
        cls._teacher_model = teacher
        if cls._teacher_model is not None:
            cls._teacher_model.eval()
            for param in cls._teacher_model.parameters():
                param.requires_grad = False

    def __init__(
            self,
            pretrained=True,
            num_frames=1,
            input_norm=True,
            mid_dim=64,
            use_checkpoint=False,
    ):
        super().__init__(pretrained, num_frames, input_norm, mid_dim, use_checkpoint)
        self.T = 10

        # PC distillation
        self.pc_loss = PCLoss()
        self.pc_weight = 5


        self._register_hooks_multi()
        self.student_features = {}
        self.teacher_features = {}

    def _register_hooks_multi(self):


        def get_student_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'student_features'):
                    self.student_features = {}
                self.student_features[name] = output

            return hook

        def get_teacher_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'teacher_features'):
                    self.teacher_features = {}
                self.teacher_features[name] = output

            return hook

        # 注册多个层的hook
        self.FSP_4.register_forward_hook(get_student_features('layer4'))
        self.FSP_3.register_forward_hook(get_student_features('layer3'))
        self.FSP_2.register_forward_hook(get_student_features('layer2'))
        # self.FSP_1.register_forward_hook(get_student_features('layer1'))

        if self._teacher_model is not None:
            self._teacher_model.FSP_4.register_forward_hook(get_teacher_features('layer4'))
            self._teacher_model.FSP_3.register_forward_hook(get_teacher_features('layer3'))
            self._teacher_model.FSP_2.register_forward_hook(get_teacher_features('layer2'))
            # self._teacher_model.FSP_1.register_forward_hook(get_teacher_features('layer1'))
    def compute_distillation_loss(self, student_output, teacher_output):
        alpha = 0.6
        # MSE loss
        soft_student_mse = (student_output / self.T).sigmoid()
        soft_teacher_mse = (teacher_output / self.T).sigmoid()
        mse_loss = F.mse_loss(soft_student_mse, soft_teacher_mse)

        # KL loss
        B, C, H, W = student_output.shape
        student_output = student_output.view(B, C, -1)
        teacher_output = teacher_output.view(B, C, -1)

        soft_student_kl = F.log_softmax(student_output / self.T, dim=2)
        soft_teacher_kl = F.softmax(teacher_output / self.T, dim=2)

        kl_loss = F.kl_div(soft_student_kl, soft_teacher_kl, reduction='batchmean') * (self.T ** 2)
        # Combined loss
        distillation_loss = alpha * mse_loss + (1 - alpha) * kl_loss * 0.1
        return distillation_loss, alpha * mse_loss, (1 - alpha) * kl_loss * 0.1

    def forward(self, data, iter_percentage=1, **kwargs):
        student_logits = self.body(data)
        prob = student_logits.sigmoid()

        if self.training and self._teacher_model is not None:
            mask = data["mask"]
            losses = []
            loss_str = []

            # 1. BCE loss
            sod_loss = F.binary_cross_entropy_with_logits(input=student_logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            # 2. UAL loss
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

            # 3. Distillation loss
            with torch.no_grad():
                teacher_logits = self._teacher_model.body(data)
            distillation_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            losses.append(distillation_loss[0])
            loss_str.append(f"total_loss: {distillation_loss[0].item():.5f}")
            loss_str.append(f"mse_loss: {distillation_loss[1].item():.5f}")
            loss_str.append(f"kl_loss: {distillation_loss[2].item():.5f}")

            #PC distillation multi
            if hasattr(self, 'student_features') and hasattr(self, 'teacher_features'):
                pc_loss_total = 0
                for layer_name in self.student_features.keys():
                    pc_loss_layer = self.pc_loss(
                        self.student_features[layer_name],
                        self.teacher_features[layer_name],
                        mask
                    )
                    pc_loss_total += pc_loss_layer
                    loss_str.append(f"pc_distill_{layer_name}: {pc_loss_layer.item():.5f}")

                pc_weighted_loss = self.pc_weight * pc_loss_total
                losses.append(pc_weighted_loss)
            else:
                loss_str.append("pc_distill: features not available")
            total_loss = sum(losses)
            return dict(
                vis=dict(sal=prob),
                loss=total_loss,
                loss_str=" ".join(loss_str)
            )
        else:
            return student_logits

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

class CFFKDNet_EFFB4(CFFKDNet):
    def __init__(self, pretrained, num_frames=1, input_norm=True, mid_dim=64, siu_groups=4, hmu_groups=6):
        super().__init__()
        self.set_backbone(pretrained)

        self.tra_4 = ConvBNReLU(self.embed_dims[4], mid_dim,3,1,1)
        self.FSP_4 = FSP(mid_dim)
        self.GRM_4 = GRM_num(mid_dim,64,8)

        self.tra_3 = ConvBNReLU(self.embed_dims[3], mid_dim, 3, 1, 1)
        self.FSP_3 = FSP(mid_dim)
        self.GRM_3 = GRM_num(mid_dim, 64,8)

        self.tra_2 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.FSP_2 = FSP(mid_dim)
        self.GRM_2 = GRM_num(mid_dim, 64,8)

        self.tra_1 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.FSP_1 = FSP(mid_dim)
        self.GRM_1 =GRM_num(mid_dim,64,8)

        self.tra_0 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.FSP_0 = FSP(mid_dim)
        self.GRM_0 = GRM_num(mid_dim, 64,8)

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def set_backbone(self, pretrained):
        self.encoder = EfficientNet.from_pretrained("efficientnet-b2", pretrained=pretrained)
        # self.embed_dims = [24, 32, 56, 160, 448]
        self.embed_dims = [16, 24, 48, 120, 352]
    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder.extract_endpoints(x)
        c1 = features["reduction_1"]
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c1, c2, c3, c4, c5

    def body(self, data):
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])

        l, m = self.tra_4(l_trans_feats[4]), self.tra_4(m_trans_feats[4])
        lms = self.FSP_4(l=l, m=m)
        x = self.GRM_4(lms)

        l, m = self.tra_3(l_trans_feats[3]), self.tra_3(m_trans_feats[3])
        lms = self.FSP_3(l=l, m=m)
        x = self.GRM_3(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m = self.tra_2(l_trans_feats[2]), self.tra_2(m_trans_feats[2])
        lms = self.FSP_2(l=l, m=m)
        x = self.GRM_2(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m = self.tra_1(l_trans_feats[1]), self.tra_1(m_trans_feats[1])
        lms = self.FSP_1(l=l, m=m)
        x = self.GRM_1(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        l, m= self.tra_0(l_trans_feats[0]), self.tra_0(m_trans_feats[0])
        lms = self.FSP_0(l=l, m=m)
        x = self.GRM_0(lms + resize_to(x, tgt_hw=lms.shape[-2:]))

        return self.predictor(x)
class EffB4_KD(CFFKDNet_EFFB4):
    """Student model that keeps teacher as a class variable instead of instance variable"""
    _teacher_model = None

    @classmethod
    def set_teacher(cls, teacher):
        cls._teacher_model = teacher
        if cls._teacher_model is not None:
            cls._teacher_model.eval()
            for param in cls._teacher_model.parameters():
                param.requires_grad = False

    def __init__(
            self,
            pretrained=True,
            num_frames=1,
            input_norm=True,
            mid_dim=64,
            use_checkpoint=False,
    ):
        super().__init__(pretrained, num_frames, input_norm, mid_dim, use_checkpoint)
        self.T = 10


        self.pc_loss = PCLoss()
        self.pc_weight = 5


        self._register_hooks_multi()
        self.student_features = {}
        self.teacher_features = {}

    def _register_hooks_multi(self):



        def get_student_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'student_features'):
                    self.student_features = {}
                self.student_features[name] = output

            return hook

        def get_teacher_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'teacher_features'):
                    self.teacher_features = {}
                self.teacher_features[name] = output

            return hook

        # 注册多个层的hook
        self.FSP_4.register_forward_hook(get_student_features('layer4'))
        self.FSP_3.register_forward_hook(get_student_features('layer3'))
        self.FSP_2.register_forward_hook(get_student_features('layer2'))
        # self.FSP_1.register_forward_hook(get_student_features('layer1'))

        if self._teacher_model is not None:
            self._teacher_model.FSP_4.register_forward_hook(get_teacher_features('layer4'))
            self._teacher_model.FSP_3.register_forward_hook(get_teacher_features('layer3'))
            self._teacher_model.FSP_2.register_forward_hook(get_teacher_features('layer2'))
            # self._teacher_model.FSP_1.register_forward_hook(get_teacher_features('layer1'))
    def compute_distillation_loss(self, student_output, teacher_output):
        alpha = 0.6
        # MSE loss
        soft_student_mse = (student_output / self.T).sigmoid()
        soft_teacher_mse = (teacher_output / self.T).sigmoid()
        mse_loss = F.mse_loss(soft_student_mse, soft_teacher_mse)

        # KL loss
        B, C, H, W = student_output.shape
        student_output = student_output.view(B, C, -1)
        teacher_output = teacher_output.view(B, C, -1)

        soft_student_kl = F.log_softmax(student_output / self.T, dim=2)
        soft_teacher_kl = F.softmax(teacher_output / self.T, dim=2)

        kl_loss = F.kl_div(soft_student_kl, soft_teacher_kl, reduction='batchmean') * (self.T ** 2)
        # Combined loss
        distillation_loss = alpha * mse_loss + (1 - alpha) * kl_loss * 0.1
        return distillation_loss, alpha * mse_loss, (1 - alpha) * kl_loss * 0.1

    def forward(self, data, iter_percentage=1, **kwargs):
        student_logits = self.body(data)
        prob = student_logits.sigmoid()

        if self.training and self._teacher_model is not None:
            mask = data["mask"]
            losses = []
            loss_str = []

            # 1. BCE loss
            sod_loss = F.binary_cross_entropy_with_logits(input=student_logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            # 2. UAL loss
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

            # 3. Distillation loss
            with torch.no_grad():
                teacher_logits = self._teacher_model.body(data)
            distillation_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            losses.append(distillation_loss[0])
            loss_str.append(f"total_loss: {distillation_loss[0].item():.5f}")
            loss_str.append(f"mse_loss: {distillation_loss[1].item():.5f}")
            loss_str.append(f"kl_loss: {distillation_loss[2].item():.5f}")

            #PC distillation multi
            if hasattr(self, 'student_features') and hasattr(self, 'teacher_features'):
                pc_loss_total = 0
                for layer_name in self.student_features.keys():
                    pc_loss_layer = self.pc_loss(
                        self.student_features[layer_name],
                        self.teacher_features[layer_name],
                        mask
                    )
                    pc_loss_total += pc_loss_layer
                    loss_str.append(f"pc_distill_{layer_name}: {pc_loss_layer.item():.5f}")

                pc_weighted_loss = self.pc_weight * pc_loss_total
                losses.append(pc_weighted_loss)
            else:
                loss_str.append("pc_distill: features not available")
            total_loss = sum(losses)
            return dict(
                vis=dict(sal=prob),
                loss=total_loss,
                loss_str=" ".join(loss_str)
            )
        else:
            return student_logits

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

from methods.backbone.smt import smt_t

class CFFKDNet_smt(CFFKDNet):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)
        self.embed_dims = [64,128,256,512]

        # Feature transformation layers
        self.tra_4 = ConvBNReLU(self.embed_dims[3], mid_dim, 3, 1, 1)
        self.FSP_4 = FSP(mid_dim)
        self.GRM_4 = GRM_num(mid_dim, 64, 8)

        self.tra_3 = ConvBNReLU(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.FSP_3 = FSP(mid_dim)
        self.GRM_3 = GRM_num(mid_dim, 64, 8)

        self.tra_2 = ConvBNReLU(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.FSP_2 = FSP(mid_dim)
        self.GRM_2 = GRM_num(mid_dim, 64, 8)

        self.tra_1 = ConvBNReLU(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.FSP_1 = FSP(mid_dim)
        self.GRM_1 = GRM_num(mid_dim, 64, 8)

        self.tras = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.predictor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(mid_dim, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )



    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = smt_t()

        if pretrained:
            checkpoint = torch.load('./weight/smt_tiny.pth', map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained weights from checkpoint")

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        return features

    def body(self, data):

        m_trans_feats = self.normalize_encoder(data["image_m"])
        l_trans_feats = self.normalize_encoder(data["image_l"])


        m, l = (
            self.tra_4(m_trans_feats[3]),
            self.tra_4(l_trans_feats[3]),
        )
        md4 = self.FSP_4(l, m)
        x = self.GRM_4(md4)

        m, l = (
            self.tra_3(m_trans_feats[2]),
            self.tra_3(l_trans_feats[2]),
        )
        md3 = self.FSP_3(l, m)
        x = self.GRM_3(md3 + resize_to(x, tgt_hw=md3.shape[-2:]))

        m, l = (
            self.tra_2(m_trans_feats[1]),
            self.tra_2(l_trans_feats[1]),
        )
        md2 = self.FSP_2(l, m)
        x = self.GRM_2(md2 + resize_to(x, tgt_hw=md2.shape[-2:]))

        m, l = (
            self.tra_1(m_trans_feats[0]),
            self.tra_1(l_trans_feats[0]),
        )
        md1 = self.FSP_1(l, m)
        x = self.GRM_1(md1 + resize_to(x, tgt_hw=md1.shape[-2:]))

        x = self.tras(x)
        return self.predictor(x)

class smt_KD(CFFKDNet_smt):
    """Student model that keeps teacher as a class variable instead of instance variable"""
    _teacher_model = None

    @classmethod
    def set_teacher(cls, teacher):
        cls._teacher_model = teacher
        if cls._teacher_model is not None:
            cls._teacher_model.eval()
            for param in cls._teacher_model.parameters():
                param.requires_grad = False

    def __init__(
            self,
            pretrained=True,
            num_frames=1,
            input_norm=True,
            mid_dim=64,
            use_checkpoint=False,
    ):
        super().__init__(pretrained, num_frames, input_norm, mid_dim, use_checkpoint)
        self.T = 10

        self.pc_loss = PCLoss()
        self.pc_weight = 5


        self._register_hooks_multi()
        self.student_features = {}
        self.teacher_features = {}

    def _register_hooks_multi(self):

        def get_student_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'student_features'):
                    self.student_features = {}
                self.student_features[name] = output

            return hook

        def get_teacher_features(name):
            def hook(module, input, output):
                if not hasattr(self, 'teacher_features'):
                    self.teacher_features = {}
                self.teacher_features[name] = output

            return hook

        # 注册多个层的hook
        self.FSP_4.register_forward_hook(get_student_features('layer4'))
        self.FSP_3.register_forward_hook(get_student_features('layer3'))
        self.FSP_2.register_forward_hook(get_student_features('layer2'))
        # self.FSP_1.register_forward_hook(get_student_features('layer1'))

        if self._teacher_model is not None:
            self._teacher_model.FSP_4.register_forward_hook(get_teacher_features('layer4'))
            self._teacher_model.FSP_3.register_forward_hook(get_teacher_features('layer3'))
            self._teacher_model.FSP_2.register_forward_hook(get_teacher_features('layer2'))
            # self._teacher_model.FSP_1.register_forward_hook(get_teacher_features('layer1'))
    def compute_distillation_loss(self, student_output, teacher_output):
        alpha = 0.6
        # MSE loss
        soft_student_mse = (student_output / self.T).sigmoid()
        soft_teacher_mse = (teacher_output / self.T).sigmoid()
        mse_loss = F.mse_loss(soft_student_mse, soft_teacher_mse)

        # KL loss
        B, C, H, W = student_output.shape
        student_output = student_output.view(B, C, -1)
        teacher_output = teacher_output.view(B, C, -1)

        soft_student_kl = F.log_softmax(student_output / self.T, dim=2)
        soft_teacher_kl = F.softmax(teacher_output / self.T, dim=2)

        kl_loss = F.kl_div(soft_student_kl, soft_teacher_kl, reduction='batchmean') * (self.T ** 2)
        # Combined loss
        distillation_loss = alpha * mse_loss + (1 - alpha) * kl_loss * 0.1
        return distillation_loss, alpha * mse_loss, (1 - alpha) * kl_loss * 0.1

    def forward(self, data, iter_percentage=1, **kwargs):
        student_logits = self.body(data)
        prob = student_logits.sigmoid()

        if self.training and self._teacher_model is not None:
            mask = data["mask"]
            losses = []
            loss_str = []

            # 1. BCE loss
            sod_loss = F.binary_cross_entropy_with_logits(input=student_logits, target=mask, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            # 2. UAL loss
            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

            # 3. Distillation loss
            with torch.no_grad():
                teacher_logits = self._teacher_model.body(data)
            distillation_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            losses.append(distillation_loss[0])
            loss_str.append(f"total_loss: {distillation_loss[0].item():.5f}")
            loss_str.append(f"mse_loss: {distillation_loss[1].item():.5f}")
            loss_str.append(f"kl_loss: {distillation_loss[2].item():.5f}")

            #PC distillation multi
            if hasattr(self, 'student_features') and hasattr(self, 'teacher_features'):
                pc_loss_total = 0
                for layer_name in self.student_features.keys():
                    pc_loss_layer = self.pc_loss(
                        self.student_features[layer_name],
                        self.teacher_features[layer_name],
                        mask
                    )
                    pc_loss_total += pc_loss_layer
                    loss_str.append(f"pc_distill_{layer_name}: {pc_loss_layer.item():.5f}")

                pc_weighted_loss = self.pc_weight * pc_loss_total
                losses.append(pc_weighted_loss)
            else:
                loss_str.append("pc_distill: features not available")
            total_loss = sum(losses)
            return dict(
                vis=dict(sal=prob),
                loss=total_loss,
                loss_str=" ".join(loss_str)
            )
        else:
            return student_logits

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict
