##################-------------------------------------------------------------########################################
                                                                                #实现每层特征图的可视化
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

# 引入你的模型定义
# 请确保这些文件在同一目录下，或者在PYTHONPATH中
from network_files import MaskRCNN
from backbone.legnet import legnet_fpn_backbone

# -----------------------------------------------------------
# 1. 定义钩子与绘图工具
# -----------------------------------------------------------
feature_maps = {}

def get_activation(name):
    """
    钩子函数：用于在模型前向传播时抓取指定层的输出
    """
    def hook(model, input, output):
        # 确保只抓取 Tensor 类型的输出
        if isinstance(output, torch.Tensor):
            feature_maps[name] = output.detach()
    return hook

def plot_feature_map(tensor, save_path, title):
    """
    将 Tensor 转换为热力图并保存
    """
    # Tensor shape 可能是 [1, C, H, W] 或 [C, H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # 变成 [C, H, W]
    
    # 核心逻辑：取通道维度的最大值 (Max Projection)
    # 对于边缘检测网络，最大值投影能最好地保留显著的边缘响应
    map_data = torch.max(tensor, dim=0)[0].cpu().numpy()

    # 归一化到 0-255
    # 加入 1e-6 防止除以零
    map_data = (map_data - map_data.min()) / (map_data.max() - map_data.min() + 1e-6)
    map_data = np.uint8(255 * map_data)

    # 应用热力图颜色映射 (JET: 蓝->红, 代表 弱->强)
    heatmap = cv2.applyColorMap(map_data, cv2.COLORMAP_JET)
    
    # 使用 Matplotlib 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap)
    plt.title(title)
    plt.axis('off') # 去掉坐标轴
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close() # 关闭图表释放内存
    print(f"[Success] Saved feature map to: {save_path}")

# -----------------------------------------------------------
# 2. 核心可视化流程
# -----------------------------------------------------------
def visualize(img_path, model_path, output_dir="./vis_results"):
    """
    Args:
        img_path: 测试图片路径
        model_path: 权重文件路径 (.pth)
        output_dir: 结果保存目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 创建输出目录 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- 2. 实例化模型 ---
    print("Building LEGNet model...")
    # 注意：这里不需要传 pretrain_path，因为我们要加载自己训练的 .pth
    backbone = legnet_fpn_backbone(pretrain_path="") 
    
    # num_classes 必须与你训练时一致 (例如: 背景 + 矿石 = 2)
    # 如果只是看 backbone 特征，这里的类别数不影响 backbone 结构，设为默认即可
    model = MaskRCNN(backbone, num_classes=2) 
    
    # --- 3. 加载权重 ---
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # 兼容保存时包含 'model' 键的情况
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
            model.load_state_dict(weights, strict=False) # strict=False 防止 head 层不匹配报错
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"[Warning] Loading weights failed: {e}")
            print("Using random weights for visualization (Edge detection might still work due to fixed Scharr).")
    else:
        print(f"[Warning] Weights file not found: {model_path}")
        print("Using random weights (Fixed operators like Scharr/LoG will still work).")

    model.to(device)
    model.eval() # 切换到评估模式

    # --- 4. 注册钩子 (Register Hooks) ---
    # 根据 legnet.py 结构精确定位
    
    # (A) Stem LoG: 入口处的斑点/纹理提取
    model.backbone.body.Stem.LoG.register_forward_hook(get_activation('0_Stem_LoG'))

    # (B) Stage 0 Scharr: 浅层的锐利边缘 (LWEGNet.stages[0])
    model.backbone.body.stages[0].blocks[0].Scharr_edge.register_forward_hook(get_activation('1_Stage0_Scharr'))

    # (C) Stage 3 Gaussian: 深层的去噪语义 (LWEGNet.stages[6])
    # 索引解释: 0(Stage0) -> 1(DRFD) -> 2(Stage1) -> 3(DRFD) -> 4(Stage2) -> 5(DRFD) -> 6(Stage3)
    model.backbone.body.stages[6].blocks[0].gaussian.register_forward_hook(get_activation('2_Stage3_Gaussian'))

    # --- 5. 图像预处理 ---
    if not os.path.exists(img_path):
        print(f"[Error] Image not found: {img_path}")
        return

    print(f"Processing image: {img_path}")
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # --- 6. 推理 ---
    with torch.no_grad():
        model(img_tensor)

    # --- 7. 保存结果 ---
    # 原始图片名 (不含后缀)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    if '0_Stem_LoG' in feature_maps:
        save_path = os.path.join(output_dir, f"{base_name}_Stem_LoG.png")
        plot_feature_map(feature_maps['0_Stem_LoG'], save_path, 'Stem Layer (LoG Filter)')
    
    if '1_Stage0_Scharr' in feature_maps:
        save_path = os.path.join(output_dir, f"{base_name}_Stage0_Scharr.png")
        plot_feature_map(feature_maps['1_Stage0_Scharr'], save_path, 'Stage 0 (Scharr Edge)')
        
    if '2_Stage3_Gaussian' in feature_maps:
        save_path = os.path.join(output_dir, f"{base_name}_Stage3_Gaussian.png")
        plot_feature_map(feature_maps['2_Stage3_Gaussian'], save_path, 'Stage 3 (Gaussian Smooth)')

    print("Done!")

# -----------------------------------------------------------
# 3. 运行入口
# -----------------------------------------------------------
if __name__ == '__main__':
    # 【用户配置区】请修改这里的路径
    
    # 1. 你的测试图片路径
    my_image = r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\test_image\p2.jpg"  
    
    # 2. 你的权重文件路径 (建议使用训练过的权重，效果更好)
    # 如果没有，留空字符串 "" 也可以跑，因为 Scharr 是固定的算子
    my_model_weights = "save_weights1/model_84.pth" 
    
    # 3. 你想保存结果的文件夹
    my_output_dir = "paper_visualizations"

    visualize(my_image, my_model_weights, my_output_dir)