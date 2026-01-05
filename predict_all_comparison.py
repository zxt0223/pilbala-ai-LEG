import os
import time
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import warnings

# 屏蔽警告
warnings.filterwarnings("ignore")

# 导入必要模块
from network_files import MaskRCNN
from backbone.legnet import legnet_fpn_backbone
from draw_box_utils import draw_objs

# ==============================================================================
# 【用户配置区】请在这里直接填入您的权重路径
# ==============================================================================
WEIGHTS_CONFIG = {
    # 1. Full 模式 (完整模型)
    "full": r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\save_weight-1.3\LegNetFull_Run2_Epoch21.pth", 
    
    # 2. Baseline 模式 (基础 ResNet 或 全禁用)
    "baseline": r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\save_weight-1.3\LegNetBaseline_Run2_Epoch23.pth", 

    # 3. No LoG (无 LoG 滤波) -> [新增] 在这里填入 no_log 实验的权重
    "no_log":  r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\save_weight-1.3\LegNetNoLog_Run3_Epoch20.pth",  
    
    # 4. No Scharr (无边缘算子)
    "no_scharr": r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\save_weight-1.3\LegNetNoScharr_Run1_Epoch21.pth",  
    
    # 5. No Gaussian (无高斯去噪)
    "no_gaussian":  r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\save_weight-1.3\LegNetNoGaussian_Run2_Epoch23.pth",
    
    # 6. No LFEA (无融合模块)
    "no_lfea": r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\save_weight-1.3\LegNetNoLFEA_Run2_Epoch22.pth"
}
# ==============================================================================

def create_model(num_classes, box_thresh=0.5, ablation_mode="full"):
    """创建模型"""
    # [修改] 加入 'no_log' 到合法列表
    valid_modes = ["full", "baseline", "no_log", "no_scharr", "no_gaussian", "no_lfea"]
    if ablation_mode not in valid_modes:
        print(f"    [Warn] Unknown mode '{ablation_mode}', defaulting to 'full'")
        ablation_mode = "full"

    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode=ablation_mode)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=1000, max_size=1333,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备输出
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. 准备图片
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = []
    if os.path.isdir(args.input_dir):
        img_list = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(supported)]
    else:
        print(f"[Error] Input dir {args.input_dir} not found.")
        return

    # 3. 标签定义 (Key 必须是字符串)
    category_index = {'1': 'stone'}

    print(f"Found {len(img_list)} images. Starting comparison loop...")

    # 4. 遍历配置字典
    for mode, weights_path in WEIGHTS_CONFIG.items():
        print(f"\n>>> Processing Mode: {mode.upper()}")
        
        # 检查路径是否为空或文件是否存在
        if not weights_path:
            print(f"    [Skip] 路径配置为空，跳过。")
            continue
        
        if not os.path.exists(weights_path):
            print(f"    [Skip] 文件不存在: {weights_path}")
            continue
            
        print(f"    [Load] Weights: {weights_path}")

        # 创建模型
        model = create_model(num_classes=2, 
                             box_thresh=args.box_thresh, 
                             ablation_mode=mode)
        
        # 加载权重
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # 去掉 module. 前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict

            model.load_state_dict(state_dict, strict=False) 
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"    [Error] Loading weights failed: {e}")
            continue

        # 批量预测
        with torch.no_grad():
            for i, img_path in enumerate(img_list):
                file_name = os.path.basename(img_path)
                original_img = Image.open(img_path).convert('RGB')
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                img = torch.unsqueeze(img, dim=0).to(device)

                predictions = model(img)[0]
                
                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                predict_mask = np.squeeze(predict_mask, axis=1)

                if len(predict_boxes) == 0:
                    print(f"    -> {file_name}: No objects detected.")
                    continue

                plot_img = draw_objs(original_img,
                                     boxes=predict_boxes,
                                     classes=predict_classes,
                                     scores=predict_scores,
                                     masks=predict_mask,
                                     category_index=category_index,
                                     line_thickness=3,
                                     font='arial.ttf',
                                     font_size=20)
                
                # 保存文件名：full_image.jpg, no_log_image.jpg
                save_name = f"{mode}_{file_name}"
                plot_img.save(os.path.join(args.output_dir, save_name))
                print(f"    -> Saved: {save_name}")

    print(f"\nAll Done! Results in: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict LEGNet Comparison")
    parser.add_argument('--input-dir', default='./test_image', help='测试图片文件夹')
    parser.add_argument('--output-dir', default='./comparison_results_new', help='结果保存路径')
    parser.add_argument('--box-thresh', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    main(args)