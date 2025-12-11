import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
# 确保这里使用的是 LEGNet backbone
from backbone.legnet import legnet_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    # 使用 LEGNet backbone，预测时不需要加载预训练权重
    backbone = legnet_fpn_backbone(pretrain_path="")
    
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    
    # ------------------- 配置区域 -------------------
    # 1. 权重文件路径
    weights_path = "save_weights/model_48.pth" # 请修改为你实际的权重文件名
    
    # 2. 输入图片文件夹 (改为文件夹路径)
    img_dir = "test_image" 
    
    # 3. 结果保存文件夹
    output_dir = "test_result"
    
    # 4. 类别索引文件
    label_json_path = 'coco91_indices.json'
    # -----------------------------------------------

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取文件夹下所有图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(supported_formats)]
    
    if len(img_list) == 0:
        print(f"文件夹 {img_dir} 中没有找到图片。")
        return

    print(f"发现 {len(img_list)} 张图片，开始批量预测...")

    # 进入验证模式
    model.eval()
    
    # ---------- 批量预测循环 ----------
    with torch.no_grad():
        for i, img_name in enumerate(img_list):
            img_path = os.path.join(img_dir, img_name)
            
            # load image
            original_img = Image.open(img_path).convert('RGB')

            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # init (optional, for warm up on first image)
            if i == 0:
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print(f"[{i+1}/{len(img_list)}] {img_name} inference+NMS time: {t_end - t_start:.3f}s")

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)

            if len(predict_boxes) == 0:
                print(f"  -> {img_name} 没有检测到任何目标!")
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
            
            # 如果不想每张图都弹窗显示，可以注释掉下面两行
            # plt.imshow(plot_img)
            # plt.show()
            
            # 保存预测结果，使用原文件名
            save_path = os.path.join(output_dir, img_name)
            plot_img.save(save_path)
            print(f"  -> 结果已保存至: {save_path}")

    print("所有图片处理完成！")

if __name__ == '__main__':
    main()