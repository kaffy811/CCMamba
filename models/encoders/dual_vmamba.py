import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock

logger = get_logger()

# ==========================================================
# 新增：基于 EAEF 思想的空间置信度门控模块 (Spatial Confidence Gate)
# 用于显式评估当前模态局部区域的信息量
# ==========================================================
class SpatialConfidenceGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # 输出 0~1 之间的置信度权重
        )
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        # return shape: (B, 1, H, W)
        return self.gate(x)
# ==========================================================


class RGBXTransformer(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )
        
        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        
        # ==========================================================
        # 新增：为每一层级实例化 RGB 和 X 的置信度评估模块
        # ==========================================================
        self.conf_rgb = nn.ModuleList([SpatialConfidenceGate(dims * (2 ** i)) for i in range(4)])
        self.conf_x = nn.ModuleList([SpatialConfidenceGate(dims * (2 ** i)) for i in range(4)])
        # ==========================================================

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W
        """
        B = x_rgb.shape[0]
        outs_fused = []
        
        outs_rgb = self.vssm(x_rgb) # B x C x H x W
        outs_x = self.vssm(x_e) # B x C x H x W
        
        for i in range(4):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]
            
            # ==========================================================
            # EAEF 增强：在送入 Cross Mamba 前，显式抑制被 Mask 或无信息的特征区域
            # ==========================================================
            conf_map_rgb = self.conf_rgb[i](out_rgb)  # (B, 1, H, W)
            conf_map_x = self.conf_x[i](out_x)        # (B, 1, H, W)
            
            out_rgb_weighted = out_rgb * conf_map_rgb
            out_x_weighted = out_x * conf_map_x
            # ==========================================================

            # cross attention (传入加权后的特征)
            cma = True
            cam = True
            if cma and cam:
                cross_rgb, cross_x = self.cross_mamba[i](out_rgb_weighted.permute(0, 2, 3, 1).contiguous(), out_x_weighted.permute(0, 2, 3, 1).contiguous()) # B x H x W x C
                x_fuse = self.channel_attn_mamba[i](cross_rgb, cross_x).permute(0, 3, 1, 2).contiguous()
            elif cam and not cma:
                x_fuse = self.channel_attn_mamba[i](out_rgb_weighted.permute(0, 2, 3, 1).contiguous(), out_x_weighted.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            elif not cam and not cma:
                x_fuse = (out_rgb_weighted + out_x_weighted)
            outs_fused.append(x_fuse)        
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
