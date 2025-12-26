import time
import os
import datetime
import argparse

import torch
# [核心修改] 禁用 P2P 以防死锁
os.environ["NCCL_P2P_DISABLE"] = "1"

import transforms
from my_dataset_coco import CocoDetection
# [核心修改] 导入新的 backbone 接口
from backbone import resnet50_fpn_backbone, resnet18_fpn_backbone, legnet_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir

def create_model(num_classes, load_pretrain_weights=True, backbone_name="legnet", ablation_mode="full"):
    # [核心修改] 动态选择 backbone
    if backbone_name == "resnet50":
        # 如果你有 resnet50.pth，请确保路径正确，否则设为 None 或让其自动下载
        backbone = resnet50_fpn_backbone(pretrain_path="/group/chenjinming/wyy/pytorch-pilipala-LEG/resnet50.pth", trainable_layers=3)
        print("Using Backbone: ResNet50")
    elif backbone_name == "resnet18":
        # 需准备 resnet18.pth
        backbone = resnet18_fpn_backbone(pretrain_path="/group/chenjinming/wyy/pytorch-pilipala-LEG/resnet18-f37072fd.pth", trainable_layers=3)
        print("Using Backbone: ResNet18")
    elif backbone_name == "legnet":
        # 这里的 pretrain_path 可以指向你在 LEG Github 下载的权重，或者设为 ""
        backbone = legnet_fpn_backbone(pretrain_path="/group/chenjinming/wyy/pytorch-pilipala-LEG/LWEGNet_tiny.pth", ablation_mode=ablation_mode)
        print(f"Using Backbone: LEGNet (Ablation Mode: {ablation_mode})")
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    model = MaskRCNN(backbone, num_classes=num_classes, min_size=1000, max_size=1333)

    if load_pretrain_weights:
        # 仅针对 ResNet50 加载官方 COCO 权重
        if backbone_name == "resnet50":
            weights_dict = torch.load("maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
            for k in list(weights_dict.keys()):
                if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    return model

def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    print("Loading data")
    # [核心修改] 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomColorJitter(brightness=0.3, contrast=0.3, prob=0.5),
            transforms.RandomGaussianBlur(prob=0.3)
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    COCO_root = args.data_path
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # [核心修改] 传入新参数
    model = create_model(num_classes=args.num_classes + 1, 
                         load_pretrain_weights=args.pretrain,
                         backbone_name=args.backbone,
                         ablation_mode=args.ablation_mode)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)
        lr_scheduler.step()

        det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

        if args.rank in [-1, 0]:
            with open(det_results_file, "a") as f:
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                f.write("epoch:{} {}\n".format(epoch, '  '.join(result_info)))

            if seg_info is not None:
                with open(seg_results_file, "a") as f:
                    result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                    f.write("epoch:{} {}\n".format(epoch, '  '.join(result_info)))

        if args.output_dir:
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files, os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='wyy/pytorch-pilipala-LEG/coco2017', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--lr-step_size', default=8, type=int)
    parser.add_argument('--lr-steps', default=[35, 45], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--output-dir', default='multi_train_weights', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true")
    parser.add_argument('--world-size', default=4, type=int)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp")

    # [核心修改] 新增参数定义，这部分就是报错缺失的内容
    parser.add_argument('--backbone', default='legnet', help='backbone: resnet50, resnet18, legnet')
    parser.add_argument('--ablation-mode', default='full', 
                        help='Ablation mode: full, baseline, no_log, no_scharr, no_gaussian, no_lfea')

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
    """
    # 注意：对程序来说，它只看得到这4张，会自动重新编号为 0,1,2,3
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 /group/chenjinming/wyy/pytorch-pilipala-stone/train_multi_GPU.py
        CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 /group/chenjinming/wyy/pytorch-pilipala-stone/train_multi_GPU.py --batch-size 4 --lr 0.015 --epochs 50 --lr-steps 35 45
        screen -dmS training bash -c "CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 /group/chenjinming/wyy/pytorch-pilipala-LEG/train_multi_GPU.py --batch-size 4 --lr 0.015 --epochs 50 --lr-steps 35 45 2>&1 | tee training.log"
        screen -dmS training bash -c "CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 /group/chenjinming/wyy/pytorch-pilipala-LEG/train_multi_GPU.py --batch-size 4 --lr 0.015 --epochs 50 --lr-steps 35 45 2>&1 | tee training_no_log.log"
        
screen -dmS training bash -c "CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 /group/chenjinming/wyy/pytorch-pilipala-LEG/train_multi_GPU.py --batch-size 4 --lr 0.015 --epochs 50 --lr-steps 35 45 2>&1 | tee training_no_scharr.log"
        
screen -dmS training bash -c "CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 /group/chenjinming/wyy/pytorch-pilipala-LEG/train_multi_GPU.py --batch-size 4 --lr 0.015 --epochs 50 --lr-steps 35 45 2>&1 | tee training_no_gaussian.log"
screen -dmS training bash -c "CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 /group/chenjinming/wyy/pytorch-pilipala-LEG/train_multi_GPU.py --batch-size 4 --lr 0.015 --epochs 50 --lr-steps 35 45 2>&1 | tee training_no_lfea.log"       
    screen -dmS train_job bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_multi_GPU.py 2>&1 | tee train_log.txt; exec bash"
    配置方案,GPU 数量,每卡 Batch Size,总 Batch Size,推荐学习率 (--lr),下降节点 (--lr-steps),说明
    方案 A,     2 张,       4,          8,              0.008,          35 45,              学习率翻倍
    方案 B,     3 张,       4,          12,             0.012,          35 45,              学习率 x3
    方案 C,     4 张,       4,          16,             0.016,          35 45,              学习率 x4
如果发现 Loss 震荡不下降（NaN 或 忽高忽低），说明学习率太大了。这时请不要犹豫，直接把推荐的学习率除以 2（例如 4 卡用 0.008）。LEGNet 有时对大学习率比较敏感。

nvidia-smi topo -m
这会显示 GPU 之间的连接矩阵（PIX, PXB, PHB, SYS 等）。如果显示 SYS，说明跨组需要走系统内存，这正是 NCCL_P2P_DISABLE=1 发挥作用的地方。
    (base) chenjinming@CCT-8xRTX4090:/group/chenjinming$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity
GPU0     X      NODE    NODE    NODE    SYS     SYS     SYS     SYS     0-63,128-191    0
GPU1    NODE     X      NODE    NODE    SYS     SYS     SYS     SYS     0-63,128-191    0
GPU2    NODE    NODE     X      NODE    SYS     SYS     SYS     SYS     0-63,128-191    0
GPU3    NODE    NODE    NODE     X      SYS     SYS     SYS     SYS     0-63,128-191    0
GPU4    SYS     SYS     SYS     SYS      X      NODE    NODE    NODE    64-127,192-255  1
GPU5    SYS     SYS     SYS     SYS     NODE     X      NODE    NODE    64-127,192-255  1
GPU6    SYS     SYS     SYS     SYS     NODE    NODE     X      NODE    64-127,192-255  1
GPU7    SYS     SYS     SYS     SYS     NODE    NODE    NODE     X      64-127,192-255  1

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
    """