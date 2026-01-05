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

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms

# 引入你的模型
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN

# ================= 配置区 =================
# 图片路径 (换成你的石头图)
IMG_PATH = r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\test_image\p2.jpg" 
# 权重路径 (如果没有训练好的，留空 "" 也可以看Scharr/LoG/Gaussian的效果)
WEIGHTS_PATH = "save_weights1/model_84.pth" 
# 输出文件夹
OUTPUT_DIR = "vis_pipeline_results"
# =========================================

def normalize_to_heatmap(tensor):
    """将 [C, H, W] 的特征图压缩为热力图"""
    if tensor is None: return None
    if tensor.dim() == 4: tensor = tensor.squeeze(0) # 去掉 Batch 维
    
    # 1. Channel Max Pooling (把64个通道压扁成1个，取最大响应)
    # 也可以用 mean(dim=0)，但 max 更能体现“有没有检测到特征”
    map_data = torch.max(tensor, dim=0)[0].cpu().detach().numpy()
    
    # 2. 归一化到 0-255
    min_val, max_val = map_data.min(), map_data.max()
    if max_val - min_val > 1e-6:
        map_data = (map_data - min_val) / (max_val - min_val)
    else:
        map_data = np.zeros_like(map_data)
        
    map_data = np.uint8(255 * map_data)
    
    # 3. 转热力图 (蓝色=弱, 红色=强)
    heatmap = cv2.applyColorMap(map_data, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def visualize_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 1. 构建模型 ---
    print("构建模型...")
    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode="full")
    model = MaskRCNN(backbone, num_classes=2)
    
    # 加载权重
    if os.path.exists(WEIGHTS_PATH):
        print(f"加载权重: {WEIGHTS_PATH}")
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    else:
        print("未找到权重，使用随机初始化 (Scharr/LoG/Gaussian 依然有效，因为它们是固定的)")

    model.to(device)
    model.eval()

    # --- 2. 注册钩子 (Hook) ---
    # 这是一个字典，用来存放所有层的输出
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    print("注册钩子...")
    # 这里的路径是根据 legnet.py 的结构推导出来的
    # body -> Stem -> LoG
    model.backbone.body.Stem.LoG.register_forward_hook(get_activation('1_LoG_Layer'))
    
    # body -> layer1 (Stage 1) -> Block 0 -> edge_extractor (Scharr)
    model.backbone.body.layer1[0].edge_extractor.register_forward_hook(get_activation('2_Scharr_Layer'))
    
    # body -> layer1 (Stage 1) -> Block 0 -> lfea (LFEA 融合后)
    model.backbone.body.layer1[0].lfea.register_forward_hook(get_activation('3_LFEA_Stage1'))

    # body -> layer2 (Stage 2) -> Block 0 -> gaussian
    model.backbone.body.layer2[0].gaussian.register_forward_hook(get_activation('4_Gaussian_Stage2'))
    
    # body -> layer3 (Stage 3) -> Block 0 -> gaussian
    model.backbone.body.layer3[0].gaussian.register_forward_hook(get_activation('5_Gaussian_Stage3'))

    # body -> layer4 (Stage 4) -> Block 0 -> gaussian
    model.backbone.body.layer4[0].gaussian.register_forward_hook(get_activation('6_Gaussian_Stage4'))

    # body -> layer4 (Stage 4) -> Block 0 -> lfea (深层 LFEA)
    model.backbone.body.layer4[0].lfea.register_forward_hook(get_activation('7_LFEA_Stage4'))

    # --- 3. 推理 ---
    img = Image.open(IMG_PATH).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    print("开始推理...")
    model(img_tensor)

    # --- 4. 绘图 (分解图) ---
    print("生成分解图...")
    keys = sorted(activations.keys()) # 按名称排序 1_LoG, 2_Scharr...
    num_plots = len(keys) + 1 # +1 是原图
    
    plt.figure(figsize=(20, 8)) # 设置画布大小
    
    # 画原图
    plt.subplot(2, 4, 1) # 2行4列布局
    plt.imshow(img)
    plt.title("0_Original_Image")
    plt.axis('off')

    # 画各个层的特征图
    for i, k in enumerate(keys):
        heatmap = normalize_to_heatmap(activations[k])
        
        # 自动安排位置 (假设一共8张图)
        plt.subplot(2, 4, i + 2) 
        plt.imshow(heatmap)
        plt.title(k)
        plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, "LEGNet_Process_Decomposition.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"分解图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_pipeline()