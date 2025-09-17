import torch
import torch.nn as nn
import torch.nn.functional as F


class PCModule(nn.Module):
    """
    Prototype Contrast Module for feature distribution distillation.

    This module calculates the relative feature distribution represented by the
    relationship between pixel features and corresponding positive/negative prototypes.
    """

    def __init__(self):
        super(PCModule, self).__init__()

    def forward(self, feature_map, ground_truth):
        """
        Calculate the PC map from feature map and ground truth mask.

        Args:
            feature_map: Tensor [B, C, H, W] - feature map
            ground_truth: Tensor [B, 1, H, W] - binary mask (0: non-change, 1: change)

        Returns:
            pc_map: Tensor [B, H, W] - prototype contrast map
        """
        B, C, H_feat, W_feat = feature_map.shape
        _, _, H_mask, W_mask = ground_truth.shape

        # 将特征图调整到掩码尺寸
        if H_feat != H_mask or W_feat != W_mask:
            feature_map = F.interpolate(
                feature_map,
                size=(H_mask, W_mask),
                mode='bilinear',
                align_corners=False
            )

        pc_map = torch.zeros((B, H_mask, W_mask), device=feature_map.device)

        for b in range(B):
            features = feature_map[b]  # [C, H, W]
            gt = ground_truth[b].squeeze(0)  # [H, W]

            # Reshape features for easier manipulation
            features = features.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
            gt_flat = gt.reshape(-1)  # [H*W]

            # Get masks for change and non-change regions
            change_mask = (gt_flat == 1)
            nonchange_mask = (gt_flat == 0)

            # Skip computation if either class is missing
            if not torch.any(change_mask) or not torch.any(nonchange_mask):
                continue

            # Extract features for each class
            change_features = features[change_mask]  # [N_c, C]
            nonchange_features = features[nonchange_mask]  # [N_n, C]

            # Calculate prototypes by averaging (Eq. 8)
            pc = torch.mean(change_features, dim=0)  # Change prototype [C]
            pn = torch.mean(nonchange_features, dim=0)  # Non-change prototype [C]

            # Normalize features and prototypes for cosine similarity
            features_norm = F.normalize(features, p=2, dim=1)
            pc_norm = F.normalize(pc, p=2, dim=0)
            pn_norm = F.normalize(pn, p=2, dim=0)

            # Calculate cosine similarities
            sim_pc = torch.mm(features_norm, pc_norm.unsqueeze(1)).squeeze()  # [H*W]
            sim_pn = torch.mm(features_norm, pn_norm.unsqueeze(1)).squeeze()  # [H*W]

            # Calculate PC map values (Eq. 9)
            pc_values = torch.zeros_like(sim_pc).float()

            # For change pixels: exp(sim(f, pc) - sim(f, pn))
            pc_values[change_mask] = torch.exp(sim_pc[change_mask] - sim_pn[change_mask])

            # For non-change pixels: exp(sim(f, pn) - sim(f, pc))
            pc_values[nonchange_mask] = torch.exp(sim_pn[nonchange_mask] - sim_pc[nonchange_mask])

            # Reshape back to spatial dimensions
            pc_map[b] = pc_values.reshape(H_feat, W_feat)

        return pc_map


# class PCLoss(nn.Module):
#     """
#     Prototype Contrast Loss for distillation.
#     """
#
#     def __init__(self):
#         super(PCLoss, self).__init__()
#         self.pc_module = PCModule()
#
#     def forward(self, student_features, teacher_features, mask):
#         """
#         Calculate PC distillation loss.
#
#         Args:
#             student_features: Tensor [B, C, H, W] - student feature map
#             teacher_features: Tensor [B, C, H, W] - teacher feature map
#             mask: Tensor [B, 1, H, W] - ground truth mask
#
#         Returns:
#             loss: PC distillation loss (mean squared error between PC maps)
#         """
#         # Generate PC maps
#         student_pc_map = self.pc_module(student_features, mask)
#         with torch.no_grad():
#             teacher_pc_map = self.pc_module(teacher_features, mask)
#
#         # Mean squared error between PC maps (Eq. 10)
#         loss = F.mse_loss(student_pc_map, teacher_pc_map)
#
#         return loss


class PCLoss(nn.Module):
    def __init__(self):
        super(PCLoss, self).__init__()
        self.pc_module = PCModule()

    def forward(self, student_features, teacher_features, mask):

        # 获取特征图的空间尺寸
        _, _, H, W = student_features.shape

        # 调整掩码大小
        if mask.shape[2] != H or mask.shape[3] != W:
            mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
        else:
            mask_resized = mask
        if teacher_features.shape[2] != H or teacher_features.shape[3] != W:
            teacher_features = F.interpolate(teacher_features, size=(H, W), mode='nearest')


        # 生成PC maps
        student_pc_map = self.pc_module(student_features, mask_resized)
        with torch.no_grad():
            teacher_pc_map = self.pc_module(teacher_features, mask_resized)

        # 计算损失
        loss = F.mse_loss(student_pc_map, teacher_pc_map)

        return loss