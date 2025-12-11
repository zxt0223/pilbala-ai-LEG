import os
import json
import shutil
import numpy as np
import glob
import PIL.Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------- 配置区域 -------------------
# 定义根目录名称
ROOT_DIR = "coco2017"

# 定义源数据和目标数据路径（自动基于ROOT_DIR拼接）
LABELME_IMAGES_DIR = os.path.join(ROOT_DIR, "labelme_images")
TRAIN_DIR = os.path.join(ROOT_DIR, "train2017")
VAL_DIR = os.path.join(ROOT_DIR, "val2017")
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "annotations")

# 验证集比例 (0.1 表示 10% 做验证集)
VAL_SIZE = 0.1

# 你的类别名称 (注意：这里要和labelme json里的label名字完全一致)
# 只有一类 stone，ID设为1
CLASS_NAME_TO_ID = {
    "stone": 1
}
# -----------------------------------------------

def check_and_create_dirs():
    """检查并自动创建所需的文件夹结构"""
    dirs_to_create = [ROOT_DIR, LABELME_IMAGES_DIR, TRAIN_DIR, VAL_DIR, ANNOTATIONS_DIR]
    for d in dirs_to_create:
        if not os.path.exists(d):
            print(f"创建文件夹: {d}")
            os.makedirs(d)

def get_coco_annotation(obj, image_id, annotation_id, class_name_to_id):
    label = obj['label']
    if label not in class_name_to_id:
        return None
    
    category_id = class_name_to_id[label]
    points = obj['points']
    
    contours = np.array(points)
    x = contours[:, 0]
    y = contours[:, 1]
    
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    w = xmax - xmin
    h = ymax - ymin
    
    area = w * h
    
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [list(np.asarray(points).flatten())],
        "area": area,
        "bbox": [xmin, ymin, w, h],
        "iscrowd": 0,
    }
    return annotation

def to_coco(img_files, dest_img_dir, ann_file_name):
    images = []
    annotations = []
    annotation_id = 1
    
    # 确保目标图片目录存在
    if not os.path.exists(dest_img_dir):
        os.makedirs(dest_img_dir)

    for img_id, img_file in enumerate(tqdm(img_files, desc=f"正在生成 {os.path.basename(ann_file_name)}")):
        # 支持 jpg 和 png，对应的 json 文件名
        json_file = os.path.splitext(img_file)[0] + ".json"
        
        if not os.path.exists(json_file):
            print(f"警告: 找不到对应的 json 文件: {img_file}")
            continue

        # 复制图片到训练/验证集目录
        shutil.copy(img_file, os.path.join(dest_img_dir, os.path.basename(img_file)))

        with open(json_file, "r") as f:
            data = json.load(f)
            
        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")
        
        if img_h is None or img_w is None:
            image = PIL.Image.open(img_file)
            img_w, img_h = image.size

        file_name = os.path.basename(img_file)
        
        image_info = {
            "id": img_id,
            "file_name": file_name,
            "height": img_h,
            "width": img_w,
        }
        images.append(image_info)
        
        for shape in data['shapes']:
            ann = get_coco_annotation(shape, img_id, annotation_id, CLASS_NAME_TO_ID)
            if ann:
                annotations.append(ann)
                annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in CLASS_NAME_TO_ID.items()]
    }

    with open(ann_file_name, "w") as f:
        json.dump(coco_format, f, indent=4)

def main():
    # 1. 检查并创建目录结构
    check_and_create_dirs()

    # 2. 获取 labelme_images 下的所有图片
    img_files = glob.glob(os.path.join(LABELME_IMAGES_DIR, "*.jpg")) + \
                glob.glob(os.path.join(LABELME_IMAGES_DIR, "*.png"))
    
    # 3. 如果文件夹为空，提示用户放入图片
    if len(img_files) == 0:
        print("\n" + "="*50)
        print(f"文件夹结构已创建！")
        print(f"请将你的 300 张原始图片(.jpg)和标注文件(.json)放入以下文件夹：")
        print(f"  -> {LABELME_IMAGES_DIR}")
        print("放入完成后，请重新运行此脚本。")
        print("="*50 + "\n")
        return

    print(f"检测到 {len(img_files)} 张图片，开始处理...")

    # 4. 划分训练集和验证集
    train_files, val_files = train_test_split(img_files, test_size=VAL_SIZE, random_state=42)
    
    # 5. 转换训练集
    print("正在处理训练集 (Train Set)...")
    to_coco(train_files, 
            TRAIN_DIR, 
            os.path.join(ANNOTATIONS_DIR, "instances_train2017.json"))
            
    # 6. 转换验证集
    print("正在处理验证集 (Val Set)...")
    to_coco(val_files, 
            VAL_DIR, 
            os.path.join(ANNOTATIONS_DIR, "instances_val2017.json"))

    print("\n" + "="*50)
    print("数据集转换完成！")
    print(f"训练集位置: {TRAIN_DIR}")
    print(f"验证集位置: {VAL_DIR}")
    print(f"标注文件位置: {ANNOTATIONS_DIR}")
    print("现在你可以运行 train.py 进行训练了。")
    print("="*50)

if __name__ == '__main__':
    main()