import torch
from backbone.legnet import LWEGNet

# 1. 实例化你的模型
model = LWEGNet(stem_dim=32, depths=(1, 4, 4, 2))
model_keys = list(model.state_dict().keys())

# 2. 加载权重文件
ckpt_path = "./LWEGNet_tiny.pth"  # 确保路径对
try:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # 有些权重包在 'state_dict' 或 'model' 字典里，有些直接是字典
    if 'state_dict' in checkpoint:
        ckpt_keys = list(checkpoint['state_dict'].keys())
    elif 'model' in checkpoint:
        ckpt_keys = list(checkpoint['model'].keys())
    else:
        ckpt_keys = list(checkpoint.keys())
        
    print(f"\n=== 侦探报告 ===")
    print(f"模型里的参数名示例 (前5个): {model_keys[:5]}")
    print(f"权重里的参数名示例 (前5个): {ckpt_keys[:5]}")
    
    print("\n=== 尝试匹配分析 ===")
    # 比如我们拿模型里的第一个卷积权重去权重文件里找
    sample_key = model_keys[0]
    print(f"正在寻找模型参数: {sample_key}")
    
    # 几种常见的改名策略测试
    candidates = [
        sample_key,
        "backbone." + sample_key,
        sample_key.replace("Stem", "stem"), # 大小写差异
        "backbone." + sample_key.replace("Stem", "stem")
    ]
    
    found = False
    for c in candidates:
        if c in ckpt_keys:
            print(f"  √ 在权重里找到了对应的名字: {c}")
            found = True
            break
            
    if not found:
        print(f"  × 没找到。这可能是前缀或者层级结构完全不同。")
        
except Exception as e:
    print(f"加载权重文件出错: {e}")