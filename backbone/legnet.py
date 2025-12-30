import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor
from collections import OrderedDict
from .feature_pyramid_network import BackboneWithFPN

# ==========================================
# 1. 基础辅助模块
# ==========================================
def get_norm(dim):
    return nn.BatchNorm2d(dim)

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

# ==========================================
# 2. 核心算子模块 (Scharr, Gaussian, LoG)
# ==========================================
class Scharr(nn.Module):
    def __init__(self, channel, act_layer):
        super(Scharr, self).__init__()
        # 定义 Scharr 算子核
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        # 固定权重，不可训练
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False
        
        self.norm = get_norm(channel)
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, act_layer)
        
        # [可视化] 初始化 debug 容器
        self.debug_feat = None

    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # 计算纯边缘强度 (不经过融合)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
        
        # >>>>> [关键] 埋点：保存纯边缘特征供可视化 (Pre-Fusion) <<<<<
        self.debug_feat = scharr_edge.detach() 
        
        # 后续正常处理 (归一化 -> 激活 -> 融合)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)
        return out

class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        # [DDP 修复] 使用 register_buffer
        self.register_buffer('gaussian_kernel_weight', gaussian.repeat(dim, 1, 1, 1))
        
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = self.gaussian_kernel_weight
        self.gaussian.weight.requires_grad = False  # 固定高斯核
        
        self.norm = get_norm(dim)
        self.act = act_layer()
        if feature_extra:
            self.conv_extra = Conv_Extra(dim, act_layer)
            
        # [可视化] 初始化 debug 容器
        self.debug_feat = None

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def forward(self, x):
        edges_o = self.gaussian(x)
        
        # >>>>> [关键] 埋点：保存纯高斯热力图 (Pre-Fusion) <<<<<
        self.debug_feat = edges_o.detach()
        
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, act_layer):
        super(LoGFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        
        # [警告修复] 显式指定 indexing='ij'
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('log_kernel_weight', log_kernel.repeat(out_c, 1, 1, 1))
        
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = self.log_kernel_weight
        self.LoG.weight.requires_grad = False
        
        self.act = act_layer()
        self.norm1 = get_norm(out_c)
        self.norm2 = get_norm(out_c)
    
    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x

# ==========================================
# 3. 网络结构组件 (LFEA, DRFD, LFE_Module)
# ==========================================
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
    def __init__(self, dim, stage, mlp_ratio, drop_path, act_layer, 
                 use_scharr=True, use_gaussian=True, use_lfea=True):
        super().__init__()
        self.stage = stage
        self.use_lfea = use_lfea
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            get_norm(mlp_hidden_dim), act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False))
        
        # [真消融] 真正可选的 LFEA
        if self.use_lfea:
            self.LFEA = LFEA(dim, act_layer)
        else:
            self.LFEA = None

        # [真消融] 真正可选的 Scharr 和 Gaussian (设为 None)
        if stage == 0:
            if use_scharr:
                self.edge_extractor = Scharr(dim, act_layer)
            else:
                self.edge_extractor = None  # No Scharr = 该分支不存在
            
            # [改进] 智能初始化，避免梯度为 0
            self.scharr_weight = nn.Parameter(torch.tensor([0.1])) 
        else:
            if use_gaussian:
                self.gaussian = Gaussian(dim, 5, 1.0, act_layer)
            else:
                self.gaussian = None  # No Gaussian = 该分支不存在
                
        self.norm = get_norm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # Stage 0: 处理边缘 (Scharr)
        if self.stage == 0:
            if self.edge_extractor is not None:
                att = self.edge_extractor(x)
                x_att = x + self.scharr_weight * att
            else:
                # [真消融] No Scharr 模式：直接透传
                x_att = x 
        
        # Stage 1-3: 处理高斯 (Gaussian)
        else:
            if self.gaussian is not None:
                att = self.gaussian(x)
                if self.use_lfea:
                    x_att = self.LFEA(x, att)
                else:
                    # [真消融] No LFEA 模式：直接相加
                    x_att = x + att 
            else:
                # [真消融] No Gaussian 模式
                x_att = x 
            
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, mlp_ratio, drop_path, act_layer, 
                 use_scharr=True, use_gaussian=True, use_lfea=True):
        super().__init__()
        blocks_list = [
            LFE_Module(dim, stage, mlp_ratio, drop_path[i], act_layer, 
                       use_scharr, use_gaussian, use_lfea)
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

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
        
        if self.use_log:
            self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, act_layer)
        else:
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

# ==========================================
# 4. 主网络 LWEGNet
# ==========================================
class LWEGNet(nn.Module):
    def __init__(self, in_chans=3, stem_dim=32, depths=(1, 4, 4, 2), act_layer=nn.ReLU, 
                 mlp_ratio=2., drop_path_rate=0.1, pretrained=None, trainable_layers=5,
                 use_log=True, use_scharr=True, use_gaussian=True, use_lfea=True):
        super().__init__()
        self.num_stages = len(depths)
        self.stem_dim = stem_dim
        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, use_log=use_log)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stages_list = []
        for i_stage in range(self.num_stages):
            dim = int(stem_dim * 2 ** i_stage)
            stage = BasicStage(dim=dim, stage=i_stage, depth=depths[i_stage], mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])], act_layer=act_layer,
                               use_scharr=use_scharr, use_gaussian=use_gaussian, use_lfea=use_lfea)
            stages_list.append(stage)
            if i_stage < self.num_stages - 1:
                stages_list.append(DRFD(dim=dim, act_layer=act_layer))
        self.stages = nn.Sequential(*stages_list)
        
        # 权重加载
        if pretrained:
            self._load_pretrained_weights(pretrained)
        else:
            self.apply(self._init_weights)

        # 冻结层逻辑
        if trainable_layers < 5:
            self._freeze_stages(trainable_layers)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _freeze_stages(self, trainable_layers):
        if trainable_layers < 1:
            for p in self.Stem.parameters(): p.requires_grad = False
        
        layers_to_freeze = 5 - trainable_layers
        if layers_to_freeze > 0:
            for i in range(layers_to_freeze - 1):
                if i < len(self.stages):
                    for p in self.stages[i].parameters(): p.requires_grad = False
                    print(f"[Info] Freezing Stage {i}")

    def _load_pretrained_weights(self, path):
        print(f"Loading LEGNet pretrained weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else (checkpoint['model'] if 'model' in checkpoint else checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('backbone.', '')
                new_state_dict[k] = v
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f">> Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
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

# ==========================================
# 5. 对外接口 legnet_fpn_backbone
# ==========================================
def legnet_fpn_backbone(pretrain_path="", ablation_mode="full", trainable_layers=3, **kwargs):
    # 默认配置
    use_log = True
    use_scharr = True
    use_gaussian = True
    use_lfea = True
    
    # === 智能模式切换逻辑 ===
    mode = str(ablation_mode).lower().strip()
    
    if mode == "baseline":
        use_log = False; use_scharr = False; use_gaussian = False; use_lfea = False
        print(f"!!! [LEGNet] Mode: BASELINE (All Modules Disabled) !!!")
    elif mode == "no_log":
        use_log = False
        print(f"!!! [LEGNet] Mode: NO LoG (LoG -> Conv) !!!")
    elif mode == "no_scharr":
        use_scharr = False
        print(f"!!! [LEGNet] Mode: NO Scharr (Branch Removed) !!!")
    elif mode == "no_gaussian":
        use_gaussian = False
        print(f"!!! [LEGNet] Mode: NO Gaussian (Branch Removed) !!!")
    elif mode == "no_lfea":
        use_lfea = False
        print(f"!!! [LEGNet] Mode: NO LFEA (Use Add) !!!")
    else:
        print(f"!!! [LEGNet] Mode: FULL (All Modules Enabled) !!!")

    stem_dim = 32
    
    backbone = LWEGNet(stem_dim=stem_dim, depths=(1, 4, 4, 2), 
                       pretrained=pretrain_path if pretrain_path else None,
                       trainable_layers=trainable_layers,
                       use_log=use_log, 
                       use_scharr=use_scharr, 
                       use_gaussian=use_gaussian,
                       use_lfea=use_lfea)
    
    # 动态计算 Channels
    in_channels_list = [int(stem_dim * 2 ** i) for i in range(4)]
    
    return BackboneWithFPN(backbone, return_layers=None, 
                           in_channels_list=in_channels_list, out_channels=256, re_getter=False)