"""
    只加上了Soft-NMS的预测，并且predict-LEG-speed.py进行速度检测文件

"""
import os
import time
import json
import random
import colorsys
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# ----------------- 导入模块 -----------------
from network_files import MaskRCNN
from backbone.legnet import legnet_fpn_backbone 
from mmcv.ops import soft_nms as mmcv_soft_nms

# ===================== [核心控制面板] =====================
# 1. 可视化开关 (True/False)
DRAW_BOX = False        # 是否画方框
DRAW_SCORE = False      # 是否画分数文字
DRAW_MASK = True        # 是否画掩膜
"""
想生成干净的论文图：
    DRAW_BOX = False
    DRAW_SCORE = False
    COLOR_STYLE = 'random
想生成分析对比图
    DRAW_BOX = True
    DRAW_SCORE = True
    COLOR_STYLE = 'green' # 或者 'random'
"""
# 2. 颜色风格
#    'random': 随机彩虹色 (SCI 实例分割标准，推荐)
#    'green':  单一绿色 (适合展示整体分布)
COLOR_STYLE = 'random' 
MASK_ALPHA = 0.45      # Mask 透明度 (0.0 - 1.0)

# 3. 结果保存
SAVE_JSON = True       # 是否导出标准 COCO 格式 JSON
# ========================================================

def generate_colors(num_colors):
    """生成区分度高的随机颜色列表 (RGB)"""
    colors = []
    for i in range(num_colors):
        h = i / num_colors
        s = 0.8 + (i % 2) * 0.1
        l = 0.5 + (i % 2) * 0.1
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    random.shuffle(colors)
    return colors

def binary_mask_to_polygon(binary_mask):
    """将二进制 mask 转换为多边形点集 (COCO 格式)"""
    # cv2.findContours 需要 uint8 类型
    mask = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4: # 至少3个点才能构成多边形
            segmentations.append(contour)
    return segmentations

def custom_draw(img_pil, boxes, scores, masks, color_style='random'):
    """
    自定义绘图函数，完全控制 Box/Score/Mask 的显示
    """
    image = np.array(img_pil) # 转 numpy (H, W, 3) RGB
    
    num_objs = len(boxes)
    if color_style == 'random':
        colors = generate_colors(num_objs)
    else:
        # 默认绿色 (0, 255, 0)
        colors = [(0, 255, 0)] * num_objs

    # 1. 绘制 Mask (使用半透明叠加)
    if DRAW_MASK:
        mask_overlay = image.copy()
        for i in range(num_objs):
            color = colors[i]
            # mask 是 (H, W) 的 bool 或 float
            # 找到 mask 区域并上色
            mask = masks[i] > 0.5
            mask_overlay[mask] = color
        
        # 混合原图和 Mask 层
        image = cv2.addWeighted(image, 1 - MASK_ALPHA, mask_overlay, MASK_ALPHA, 0)

    # 2. 绘制 Box 和 Text
    for i in range(num_objs):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = colors[i]
        
        if DRAW_BOX:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if DRAW_SCORE:
            text = f"{int(scores[i]*100)}%"
            # 计算文字大小
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # 画文字背景框 (让字看清楚)
            cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            # 画文字
            cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    return Image.fromarray(image)

def create_model(num_classes):
    backbone = legnet_fpn_backbone(pretrain_path="")
    model = MaskRCNN(backbone, num_classes=num_classes,
                     min_size=1000, max_size=1333,
                     rpn_score_thresh=0.1,
                     box_score_thresh=0.1,
                     box_detections_per_img=400,
                     box_nms_thresh=1.0)
    return model

def main():
    num_classes = 1 
    weights_path = "save_weights1/model_84.pth"
    img_dir = "test_image"
    output_dir = "test_result_pro"
    json_output_path = "test_result_pro/results.json"
    
    # Soft-NMS 参数
    nms_sigma = 0.5        
    final_thresh = 0.25
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(num_classes=num_classes + 1)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    # 用于存储所有图片结果的列表
    coco_results = []

    print(f"开始处理 {len(img_list)} 张图片...")
    print(f"配置: Box={DRAW_BOX}, Score={DRAW_SCORE}, Color={COLOR_STYLE}")

    with torch.no_grad():
        for i, img_name in enumerate(img_list):
            img_path = os.path.join(img_dir, img_name)
            original_img = Image.open(img_path).convert('RGB')
            w, h = original_img.size
            img_tensor = transforms.Compose([transforms.ToTensor()])(original_img).unsqueeze(0).to(device)

            # 推理
            predictions = model(img_tensor)[0]
            raw_boxes = predictions["boxes"]
            raw_scores = predictions["scores"]
            
            if len(raw_boxes) == 0: continue

            # CUDA Soft-NMS
            dets, keep_indices = mmcv_soft_nms(
                raw_boxes, raw_scores, method='gaussian', sigma=nms_sigma, iou_threshold=0.3, min_score=0.001
            )
            
            # 过滤
            final_scores = dets[:, 4]
            keep_mask = final_scores > final_thresh
            valid_indices = keep_indices[keep_mask]
            
            # 提取数据 (转 CPU)
            final_boxes = raw_boxes[valid_indices].cpu().numpy()
            final_scores = final_scores[keep_mask].cpu().numpy()
            final_masks = predictions["masks"][valid_indices].cpu().numpy()
            final_masks = np.squeeze(final_masks, axis=1)

            print(f"[{i+1}/{len(img_list)}] {img_name} -> {len(final_boxes)} instances")

            # 1. 绘图 (使用新的自定义函数)
            plot_img = custom_draw(original_img, final_boxes, final_scores, final_masks, color_style=COLOR_STYLE)
            plot_img.save(os.path.join(output_dir, img_name))

            # 2. 生成 JSON 数据
            if SAVE_JSON:
                for idx in range(len(final_boxes)):
                    box = final_boxes[idx].tolist() # [x1, y1, x2, y2]
                    # COCO 格式的 box 是 [x, y, w, h]
                    coco_box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                    
                    # 转换 mask 为 polygon
                    segmentation = binary_mask_to_polygon(final_masks[idx])
                    
                    res_item = {
                        "image_id": img_name,      # 使用文件名作为 ID
                        "category_id": 1,          # stone
                        "bbox": [round(x, 2) for x in coco_box],
                        "score": round(float(final_scores[idx]), 4),
                        "segmentation": segmentation # 这里存的是多边形点集
                    }
                    coco_results.append(res_item)

    # 保存 JSON 文件
    if SAVE_JSON:
        with open(json_output_path, 'w') as f:
            json.dump(coco_results, f)
        print(f"JSON 结果已保存至: {json_output_path}")

    print("全部完成！")

if __name__ == '__main__':
    main()