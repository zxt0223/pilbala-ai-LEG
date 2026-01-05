""""
2. 如何看这张“分解图”？（演示与讲解逻辑）
运行完代码，你会得到一张包含 8 个子图的大图。这就是你要的 “处理过程分解图”。

你可以按照以下顺序，指着图给老师或审稿人讲解：

第一阶段：浅层几何提取 (Stage 1)
0_Original_Image: 指出石头粘连严重，表面有杂点。

1_LoG_Layer (LoG 滤波):

现象： 你应该能看到一个个红色的光斑或圆环。

讲解： “老师请看，LoG 算子对斑点敏感。它成功定位了每块石头的中心位置（红色高亮），并且辅助区分了粘连处的拓扑结构。”

2_Scharr_Layer (Scharr 边缘):

现象： 背景应该是蓝色的，石头的轮廓线是红色的，非常清晰。

讲解： “Scharr 算子专门提取梯度。可以看到，它把粘连石头中间那条微弱的缝隙都高亮出来了，这就是我们解决粘连问题的关键。”

3_LFEA_Stage1 (浅层融合):

现象： 这张图应该是 LoG、Scharr 和原始特征的结合体。既有轮廓，又有实体感。

讲解： “经过 LFEA 模块的自适应加权，网络保留了重要的边缘和形状信息，丢弃了部分背景噪声。”

第二阶段：深层语义去噪 (Stage 2/3/4)
4_Gaussian_Stage2 -> 5_Gaussian_Stage3 -> 6_Gaussian_Stage4:

现象： 随着层数变深（Stage 2 -> 4），图像分辨率变低（越来越马赛克），但颜色变得越来越均匀。石头内部原本杂乱的纹理（红蓝相间）逐渐融合成一大块红色或黄色。

讲解： “进入深层后，Gaussian 模块开始工作。请对比 Stage 2 和 Stage 4，可以看到石头表面的高频纹理噪声被逐渐抹平（Smoothed out）。这使得网络在做语义分割时，能把石头看作一个整体，而不是一堆碎片。”

第三阶段：最终融合
7_LFEA_Stage4 (深层融合):

现象： 高度抽象的特征图，热力点非常集中在石头主体上。

讲解： “这是进入 RPN 之前的最终特征。它非常干净，只保留了具有高语义价值的石头区域，为后续生成高质量的 Mask 打下了基础。”


3. 如何画“演示处理过程”的原理图？(Paper Drawing)
如果你想画一张用于论文 Methodology 章节的精美流程图（不仅仅是代码跑出来的热力图，而是带箭头的框图），建议使用 Visio 或 PPT。

构图思路：

左边： 放一张“石头原图”。

中间上路 (浅层)：

画两个分支箭头。

上分支 -> 图标 [Scharr矩阵] -> 贴上代码跑出来的 Scharr 热力图。

下分支 -> 图标 [LoG公式] -> 贴上代码跑出来的 LoG 热力图。

汇聚 -> 图标 [LFEA] -> 贴上 LFEA_Stage1 热力图。

中间下路 (深层)：

画一个大箭头指向深层。

图标 [Gaussian钟形曲线] -> 贴上 Gaussian_Stage4 热力图。

注解：写上 "Texture Suppression" (纹理抑制)。

右边： 放一张最终的“分割结果图” (Mask)。

总结： 用 visualize_full_pipeline.py 跑出来的真实热力图，嵌入到你画的 Visio 流程框图中，这种 "原理框图 + 真实数据验证" 的图，是顶刊（如 IEEE Trans）最喜欢的风格！
"""
"""
LEGNet 可视化流水线 (Paper Ready)
功能：生成包含 [原图 -> LoG -> Scharr -> LFEA -> Gaussian Deep -> Mask] 的全流程分解图
改进：
1. 修正了模型层级索引 (stages[0] 等)。
2. 直接提取算子内部的 debug_feat (纯净特征)，而非融合后的 output。
3. 针对不同特征类型使用了不同的色卡 (Edges用HOT, Areas用JET)。
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import glob
import re

# 引入你的模型
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN

# ================= 配置区 =================
# 图片路径
IMG_PATH = r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\test_image\p2.jpg" 
# 权重目录 (脚本会自动找里面最新的 .pth)
WEIGHTS_DIR = "./save_weight-1.3"
# 输出文件夹
OUTPUT_DIR = "vis_pipeline_paper"
# =========================================

def find_best_weights_simple(base_dir):
    """简单寻找目录下最新的pth文件 (支持任意文件名)"""
    if not os.path.exists(base_dir): return ""
    # 修改点：搜索所有 *.pth
    pths = glob.glob(os.path.join(base_dir, "*.pth"))
    # 同时也搜子目录
    pths += glob.glob(os.path.join(base_dir, "**", "*.pth"), recursive=True)
    
    if not pths: return ""
    
    # 按数字排序，如果没有数字则返回第一个
    def get_epoch(p):
        nums = re.findall(r'\d+', os.path.basename(p))
        return int(nums[-1]) if nums else -1
        
    return max(pths, key=get_epoch)

def normalize_to_heatmap(tensor, colormap=cv2.COLORMAP_JET):
    if tensor is None: return None
    if tensor.dim() == 4: tensor = tensor.squeeze(0) 
    map_data = torch.max(tensor, dim=0)[0].cpu().detach().numpy()
    min_val, max_val = map_data.min(), map_data.max()
    if max_val - min_val > 1e-6:
        map_data = (map_data - min_val) / (max_val - min_val)
    else:
        map_data = np.zeros_like(map_data)
    map_data = np.uint8(255 * map_data)
    heatmap = cv2.applyColorMap(map_data, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def visualize_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("构建 LEGNet 模型 (Full Mode)...")
    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode="full")
    model = MaskRCNN(backbone, num_classes=2)
    
    weights_path = find_best_weights_simple(WEIGHTS_DIR)
    if weights_path:
        print(f"加载权重: {weights_path}")
        try:
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"[Warn] 权重加载失败: {e}")
    else:
        print(f"[Warn] 在 {WEIGHTS_DIR} 未找到任何 .pth 文件！")

    model.to(device)
    model.eval()

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Hook LoG Raw
    model.backbone.body.Stem.LoG.LoG.register_forward_hook(get_activation('1_LoG_Raw'))
    
    # Hook LFEA Stage 1
    if hasattr(model.backbone.body.stages[0].blocks[0], 'LFEA'):
        model.backbone.body.stages[0].blocks[0].LFEA.register_forward_hook(get_activation('3_LFEA_Stage1'))

    # Hook LFEA Stage 4
    if hasattr(model.backbone.body.stages[6].blocks[0], 'LFEA'):
        model.backbone.body.stages[6].blocks[0].LFEA.register_forward_hook(get_activation('7_LFEA_Stage4'))

    if not os.path.exists(IMG_PATH):
        print(f"[Error] 图片路径不存在: {IMG_PATH}")
        return

    img = Image.open(IMG_PATH).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    print("开始推理...")
    model(img_tensor)

    # Debug Feats
    activations['2_Scharr_Raw'] = model.backbone.body.stages[0].blocks[0].edge_extractor.debug_feat
    activations['4_Gaussian_Stage2'] = model.backbone.body.stages[2].blocks[0].gaussian.debug_feat
    activations['5_Gaussian_Stage3'] = model.backbone.body.stages[4].blocks[0].gaussian.debug_feat
    activations['6_Gaussian_Stage4'] = model.backbone.body.stages[6].blocks[0].gaussian.debug_feat

    # Plot
    print("生成可视化图表...")
    plot_config = [
        ('1_LoG_Raw', 'LoG Filter', cv2.COLORMAP_HOT),
        ('2_Scharr_Raw', 'Scharr Operator', cv2.COLORMAP_HOT),
        ('3_LFEA_Stage1', 'LFEA Shallow', cv2.COLORMAP_JET),
        ('4_Gaussian_Stage2', 'Gaussian S2', cv2.COLORMAP_JET),
        ('5_Gaussian_Stage3', 'Gaussian S3', cv2.COLORMAP_JET),
        ('6_Gaussian_Stage4', 'Gaussian S4', cv2.COLORMAP_JET),
        ('7_LFEA_Stage4', 'LFEA Deep', cv2.COLORMAP_JET),
    ]
    
    num_plots = len(plot_config) + 1
    plt.figure(figsize=(24, 5))
    plt.subplot(1, num_plots, 1)
    plt.imshow(img); plt.title("Input"); plt.axis('off')

    for i, (key, title, cmap) in enumerate(plot_config):
        if key in activations and activations[key] is not None:
            heatmap = normalize_to_heatmap(activations[key], colormap=cmap)
            plt.subplot(1, num_plots, i + 2)
            plt.imshow(heatmap); plt.title(title); plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, "LEGNet_Pipeline_Vis.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"完成！图片已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_pipeline()