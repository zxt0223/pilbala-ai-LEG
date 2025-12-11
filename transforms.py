import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target
# === 将此代码追加到 transforms.py 文件末尾 ===

class RandomVerticalFlip(object):
    """随机垂直翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)  # 垂直翻转图片 (H维度)
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            # 垂直翻转只需改变 ymin 和 ymax
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]] 
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-2)
        return image, target
# [请追加到 transforms.py 末尾]
from torchvision.transforms import ColorJitter as TorchColorJitter
from torchvision.transforms import GaussianBlur as TorchGaussianBlur

class RandomColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, prob=0.5):
        self.transform = TorchColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.transform(image)
        return image, target

class RandomGaussianBlur(object):
    def __init__(self, kernel_size=(5, 9), sigma=(0.1, 5), prob=0.5):
        self.transform = TorchGaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.transform(image)
        return image, target