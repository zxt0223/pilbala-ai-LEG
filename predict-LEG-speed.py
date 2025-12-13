import os
import time
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ----------------- 导入模块 -----------------
from network_files import MaskRCNN
from backbone.legnet import legnet_fpn_backbone 
from draw_box_utils import draw_objs

# [关键升级] 导入 CUDA 加速版 Soft-NMS
from mmcv.ops import soft_nms as mmcv_soft_nms

# ===================== [核心控制台] =====================
# 1. 模式选择
#    'original':      原版 Hard-NMS (基准速度)
#    'cuda_soft_nms': CUDA 加速 Soft-NMS (高精且快！)
TEST_MODE = 'original'  

# 2. 纯测速开关
#    True:  循环跑 100 次，不画图，只看 FPS (极速)
#    False: 跑一遍，保存图片 (检查效果)
BENCHMARK_ONLY = True   

# 3. 测速循环次数
BENCHMARK_LOOPS = 100
# ========================================================

def create_model(num_classes, mode='original'):
    backbone = legnet_fpn_backbone(pretrain_path="") 
    
    if mode == 'original':
        print("-> [配置] 模式: Original (Hard-NMS)")
        model = MaskRCNN(backbone, num_classes=num_classes,
                         min_size=1000, max_size=1333,
                         rpn_score_thresh=0.5,
                         box_score_thresh=0.5,
                         box_detections_per_img=100,
                         box_nms_thresh=0.5) # 开启硬 NMS
    else:
        print("-> [配置] 模式: CUDA Soft-NMS")
        model = MaskRCNN(backbone, num_classes=num_classes,
                         min_size=1000, max_size=1333,
                         rpn_score_thresh=0.2,
                         box_score_thresh=0.2,       # 放低门槛
                         box_detections_per_img=500, # 保留大量框
                         box_nms_thresh=1.0)          # 禁用硬 NMS
    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def run_inference_logic(model, img_tensor, mode, device, nms_sigma=0.5, final_thresh=0.25):
    """封装推理逻辑"""
    # 1. 模型前向传播
    predictions = model(img_tensor)[0]
    
    # 2. 根据模式处理
    if mode == 'original':
        # 原版直接返回
        return predictions['boxes'].cpu().numpy(), \
               predictions['labels'].cpu().numpy(), \
               predictions['scores'].cpu().numpy(), \
               np.squeeze(predictions['masks'].cpu().numpy(), axis=1)
               
    elif mode == 'cuda_soft_nms':
        # CUDA Soft-NMS 处理
        raw_boxes = predictions["boxes"]
        raw_scores = predictions["scores"]
        
        if len(raw_boxes) == 0:
            return [], [], [], []

        # [核心] 调用 MMCV 的 CUDA 算子
        # 这一步是在 GPU 上完成的，极快
        dets, keep_indices = mmcv_soft_nms(
            raw_boxes, 
            raw_scores, 
            method='gaussian', 
            sigma=nms_sigma, 
            iou_threshold=0.3, 
            min_score=0.001 
        )
        
        # 过滤最终阈值
        final_scores_tensor = dets[:, 4] # 第5列是分值
        keep_mask = final_scores_tensor > final_thresh
        valid_indices = keep_indices[keep_mask]
        
        # 此时才将数据搬运回 CPU 用于画图或输出
        final_boxes = raw_boxes[valid_indices].cpu().numpy()
        final_labels = predictions["labels"][valid_indices].cpu().numpy()
        final_scores = final_scores_tensor[keep_mask].cpu().numpy()
        final_masks = predictions["masks"][valid_indices].cpu().numpy()
        
        return final_boxes, final_labels, final_scores, np.squeeze(final_masks, axis=1)

def main():
    num_classes = 1 
    weights_path = "save_weights1/model_84.pth"
    img_dir = "test_image"
    output_dir = "test_result_final"
    label_json_path = 'coco91_indices.json'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 创建模型
    model = create_model(num_classes=num_classes + 1, mode=TEST_MODE)
    
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    # 2. 准备数据
    if not os.path.exists(img_dir): print(f"{img_dir} 不存在"); return
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not img_list: print("无图片"); return
    
    # 预加载一张图
    test_img_path = os.path.join(img_dir, img_list[0])
    original_img = Image.open(test_img_path).convert('RGB')
    img_tensor = transforms.Compose([transforms.ToTensor()])(original_img).unsqueeze(0).to(device)

    # ================= 纯测速模式 =================
    if BENCHMARK_ONLY:
        print(f"\n======== 极速跑分模式: {TEST_MODE} ========")
        print("正在 GPU 预热...")
        with torch.no_grad():
            for _ in range(10): model(img_tensor)
        
        print(f"开始冲刺 {BENCHMARK_LOOPS} 次...")
        t_start = time_synchronized()
        
        with torch.no_grad():
            for _ in range(BENCHMARK_LOOPS):
                # 跑完整流程，包括 NMS
                run_inference_logic(model, img_tensor, TEST_MODE, device)
                
        t_end = time_synchronized()
        avg_time = (t_end - t_start) / BENCHMARK_LOOPS
        fps = 1.0 / avg_time
        
        print(f"\n[最终成绩]")
        print(f"-------------------------------------------")
        print(f"模式: {TEST_MODE}")
        print(f"平均耗时: {avg_time*1000:.2f} ms")
        print(f"FPS     : {fps:.2f}")
        print(f"-------------------------------------------")
            
    # ================= 可视化模式 =================
    else:
        print(f"\n======== 效果验证模式: {TEST_MODE} ========")
        with open(label_json_path, 'r') as f: category_index = json.load(f)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        print(f"正在处理 {len(img_list)} 张图片...")
        with torch.no_grad():
            for i, img_name in enumerate(img_list):
                img_path = os.path.join(img_dir, img_name)
                original_img = Image.open(img_path).convert('RGB')
                img_tensor = transforms.Compose([transforms.ToTensor()])(original_img).unsqueeze(0).to(device)
                
                t0 = time_synchronized()
                boxes, classes, scores, masks = run_inference_logic(
                    model, img_tensor, TEST_MODE, device, nms_sigma=0.5, final_thresh=0.25
                )
                t1 = time_synchronized()
                
                print(f"[{i+1}/{len(img_list)}] {img_name} Time: {t1-t0:.3f}s | Objs: {len(boxes)}")
                
                if len(boxes) > 0:
                    plot_img = draw_objs(original_img, boxes, classes, scores, masks, 
                                         category_index, line_thickness=3, font='arial.ttf', font_size=20)
                    plot_img.save(os.path.join(output_dir, img_name))
        print("全部完成，请检查图片！")

if __name__ == '__main__':
    main()