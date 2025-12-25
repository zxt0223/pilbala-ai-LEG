# 路径: backbone/legnet.py
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor
from collections import OrderedDict
from .feature_pyramid_network import BackboneWithFPN

# --- 辅助模块 ---
def get_norm(dim):
    return nn.BatchNorm2d(dim)

class DRFD(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = get_norm(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = get_norm(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        self.gaussian = Gaussian(self.outdim, 5, 0.5, act_layer, feature_extra=False)
        self.norm_g = get_norm(self.outdim)

    def forward(self, x):
        x = self.conv(x)
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max_feat = self.norm_m(self.max_m(x))
        conv_feat = self.norm_c(self.act_c(self.conv_c(x)))
        x = torch.cat([conv_feat, max_feat], dim=1)
        x = self.fusion(x)
        return x

class Conv_Extra(nn.Module):
    def __init__(self, channel, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   get_norm(64), act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   get_norm(64), act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   get_norm(channel))
    def forward(self, x):
        return self.block(x)

class Scharr(nn.Module):
    def __init__(self, channel, act_layer):
        super(Scharr, self).__init__()
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = get_norm(channel)
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, act_layer)

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)
        return out

class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        self.register_buffer('gaussian_kernel_weight', gaussian.repeat(dim, 1, 1, 1))
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = self.gaussian_kernel_weight
        self.gaussian.weight.requires_grad = False 
        self.norm = get_norm(dim)
        self.act = act_layer()
        if feature_extra:
            self.conv_extra = Conv_Extra(dim, act_layer)

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def forward(self, x):
        if self.gaussian.weight.device != x.device:
             self.gaussian.weight.data = self.gaussian.weight.data.to(x.device)
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

class LFEA(nn.Module):
    def __init__(self, channel, act_layer):
        super(LFEA, self).__init__()
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            get_norm(channel), act_layer())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = get_norm(channel)

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)
        return x

class LFE_Module(nn.Module):
    def __init__(self, dim, stage, mlp_ratio, drop_path, act_layer, use_scharr=True, use_gaussian=True):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            get_norm(mlp_hidden_dim), act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False))
        self.LFEA = LFEA(dim, act_layer)
        
        # === 核心修改：支持消融实验 ===
        if stage == 0:
            if use_scharr:
                self.edge_extractor = Scharr(dim, act_layer)
            else:
                # Baseline: 使用普通卷积代替 Scharr，模拟“无先验”情况
                self.edge_extractor = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                    get_norm(dim),
                    act_layer(),
                    Conv_Extra(dim, act_layer) # 保持结构一致
                )
        else:
            if use_gaussian:
                self.gaussian = Gaussian(dim, 5, 1.0, act_layer)
            else:
                # Baseline: 使用普通卷积代替 Gaussian
                self.gaussian = nn.Sequential(
                    nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
                    get_norm(dim),
                    act_layer()
                )
        self.norm = get_norm(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.stage == 0:
            att = self.edge_extractor(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, mlp_ratio, drop_path, act_layer, use_scharr=True, use_gaussian=True):
        super().__init__()
        blocks_list = [
            LFE_Module(dim, stage, mlp_ratio, drop_path[i], act_layer, use_scharr, use_gaussian)
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, act_layer):
        super(LoGFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.LoG.weight.requires_grad = False
        self.act = act_layer()
        self.norm1 = get_norm(out_c)
        self.norm2 = get_norm(out_c)
    
    def forward(self, x):
        x = self.conv_init(x)
        if self.LoG.weight.device != x.device:
             self.LoG.weight.data = self.LoG.weight.data.to(x.device)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans, stem_dim, act_layer, use_log=True):
        super().__init__()
        self.use_log = use_log
        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            get_norm(out_c12))
        
        # === 核心修改：支持消融实验 ===
        if self.use_log:
            self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, act_layer)
        else:
            # Baseline: 用普通 Conv 替代 LoG
            self.LoG = nn.Sequential(
                nn.Conv2d(in_chans, out_c14, kernel_size=7, stride=1, padding=3, bias=False),
                get_norm(out_c14),
                act_layer()
            )
            
        self.gaussian = Gaussian(out_c12, 9, 0.5, act_layer)
        self.norm = get_norm(out_c12)
        self.drfd = DRFD(out_c12, act_layer)

    def forward(self, x):
        x = self.LoG(x)
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)
        return x

# --- 主网络 LWEGNet ---
class LWEGNet(nn.Module):
    def __init__(self, in_chans=3, stem_dim=32, depths=(1, 4, 4, 2), act_layer=nn.ReLU, 
                 mlp_ratio=2., drop_path_rate=0.1, pretrained=None,
                 # === 开关参数 ===
                 use_log=True, use_scharr=True, use_gaussian=True):
        super().__init__()
        self.num_stages = len(depths)
        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, use_log=use_log)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stages_list = []
        for i_stage in range(self.num_stages):
            dim = int(stem_dim * 2 ** i_stage)
            stage = BasicStage(dim=dim, stage=i_stage, depth=depths[i_stage], mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])], act_layer=act_layer,
                               use_scharr=use_scharr, use_gaussian=use_gaussian)
            stages_list.append(stage)
            if i_stage < self.num_stages - 1:
                stages_list.append(DRFD(dim=dim, act_layer=act_layer))
        self.stages = nn.Sequential(*stages_list)
        
        if pretrained:
            self._load_pretrained_weights(pretrained)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self, path):
        print(f"Loading LEGNet pretrained weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v
            
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            if len(missing) == 0:
                print(">> Success: All LEGNet weights loaded perfectly!")
            else:
                print(f">> Loaded partially. Missing keys: {len(missing)}")
        except Exception as e:
            print(f"[Error] Failed to load weights: {e}")

    def forward(self, x):
        x = self.Stem(x)
        outs = OrderedDict()
        out_indices = {0: '0', 2: '1', 4: '2', 6: '3'}
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in out_indices:
                outs[out_indices[idx]] = x
        return outs

def legnet_fpn_backbone(pretrain_path="", ablation_mode="full", **kwargs):
    """
    Args:
        pretrain_path: 预训练权重路径
        ablation_mode: 消融实验模式
            - "full": 完整版 (LoG + Scharr + Gaussian) - 默认
            - "baseline": 没有任何算子 (全部替换为普通Conv)
            - "no_log": 只有 Scharr + Gaussian
            - "no_scharr": 只有 LoG + Gaussian
            - "no_gaussian": 只有 LoG + Scharr
    """
    # 默认全部开启
    use_log = True
    use_scharr = True
    use_gaussian = True
    
    # 根据模式调整开关
    if ablation_mode == "baseline":
        use_log = False
        use_scharr = False
        use_gaussian = False
        print("!!! [Ablation Study] Mode: BASELINE (All operators disabled) !!!")
    elif ablation_mode == "no_log":
        use_log = False
        print("!!! [Ablation Study] Mode: NO LoG !!!")
    elif ablation_mode == "no_scharr":
        use_scharr = False
        print("!!! [Ablation Study] Mode: NO Scharr !!!")
    elif ablation_mode == "no_gaussian":
        use_gaussian = False
        print("!!! [Ablation Study] Mode: NO Gaussian !!!")
    else:
        print("!!! [Ablation Study] Mode: FULL (Standard LEGNet) !!!")

    backbone = LWEGNet(stem_dim=32, depths=(1, 4, 4, 2), 
                       pretrained=pretrain_path if pretrain_path else None,
                       use_log=use_log,
                       use_scharr=use_scharr,
                       use_gaussian=use_gaussian)
    
    in_channels_list = [32, 64, 128, 256]
    
    return BackboneWithFPN(backbone, 
                           return_layers=None, 
                           in_channels_list=in_channels_list, 
                           out_channels=256,
                           re_getter=False)