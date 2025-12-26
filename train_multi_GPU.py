import time
import os
import datetime
import torch
import shutil 

# 禁用 NCCL P2P
os.environ["NCCL_P2P_DISABLE"] = "1"

# =========================================================
# [核心修改] 你的绝对项目根路径
# 所有的输入(coco2017, 权重) 和 输出(logs, save_weights) 都会基于这个路径
# =========================================================
PROJECT_ROOT = "/group/chenjinming/wyy/pytorch-pilipala-LEG"

import transforms
from my_dataset_coco import CocoDetection
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir
from torch.utils.tensorboard import SummaryWriter

def create_model(num_classes, load_pretrain_weights=True, ablation_mode="full"):
    # 自动在根目录下找预训练权重
    weights_path = os.path.join(PROJECT_ROOT, "LWEGNet_tiny.pth")
    
    if not os.path.exists(weights_path):
        print(f"Warning: Pretrained weights not found at {weights_path}, using random init.")
        weights_path = "" 
        
    backbone = legnet_fpn_backbone(pretrain_path=weights_path, ablation_mode=ablation_mode)
    model = MaskRCNN(backbone, num_classes=num_classes, min_size=1000, max_size=1333)
    return model

def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # ================= [文件路径管理 - 绝对路径版] =================
    # 1. 确定本次实验的保存文件夹 (例如: /group/.../save_weights/no_scharr)
    experiment_dir = os.path.join(args.output_dir, args.ablation_mode)
    
    if args.rank in [-1, 0]:
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        print(f">> [Output] All results will be saved to: {experiment_dir}")

    # 2. 结果 txt 文件名自动生成
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    det_results_file = os.path.join(experiment_dir, f"det_results_{args.ablation_mode}_{now_str}.txt")
    seg_results_file = os.path.join(experiment_dir, f"seg_results_{args.ablation_mode}_{now_str}.txt")
    # ===========================================================

    # TensorBoard 日志保存到根目录下的 logs/
    writer = None
    if args.rank == 0:
        log_root = os.path.join(PROJECT_ROOT, "logs")
        log_dir = os.path.join(log_root, f"exp_{args.ablation_mode}_{now_str}")
        writer = SummaryWriter(log_dir=log_dir)

    print("Loading data")
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

    # 强制使用根目录下的 coco2017
    # 即使命令行不传 --data-path，这里也会覆盖为正确的绝对路径
    if args.data_path == 'coco2017': # 如果是默认值
        args.data_path = os.path.join(PROJECT_ROOT, 'coco2017')
    
    COCO_root = args.data_path
    if not os.path.exists(COCO_root):
        raise FileNotFoundError(f"Dataset path not found: {COCO_root}")

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

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_dataset.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=train_dataset.collate_fn)

    print(f"Creating model with ablation mode: {args.ablation_mode}")
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain, ablation_mode=args.ablation_mode)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, warmup=True, scaler=scaler)
        
        if writer is not None:
            writer.add_scalar('Train/Loss', mean_loss.item(), epoch)
            writer.add_scalar('Train/Learning_Rate', lr, epoch)

        lr_scheduler.step()

        # Evaluate
        det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

        if args.rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            val_map.append(det_info[1]) 

            if writer is not None:
                writer.add_scalar('Val/Det_mAP_0.5:0.95', det_info[0], epoch)
                writer.add_scalar('Val/Det_mAP_0.5', det_info[1], epoch)
                if seg_info is not None:
                    writer.add_scalar('Val/Seg_mAP_0.5:0.95', seg_info[0], epoch)

            # 写入带模式后缀的 TXT 文件
            with open(det_results_file, "a") as f:
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            if seg_info is not None:
                with open(seg_results_file, "a") as f:
                    result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")

            # 保存权重到专门的文件夹
            if args.output_dir:
                save_files = {'model': model_without_ddp.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'lr_scheduler': lr_scheduler.state_dict(),
                              'args': args,
                              'epoch': epoch}
                if args.amp:
                    save_files["scaler"] = scaler.state_dict()
                
                # 仅保存 model_0.pth, model_1.pth 到子目录
                save_on_master(save_files, os.path.join(experiment_dir, f'model_{epoch}.pth'))

    # 训练结束：自动画图并归档
    if args.rank in [-1, 0]:
        if writer is not None:
            writer.close()
        
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr, plot_map
            plot_loss_and_lr(train_loss, learning_rate)
            plot_map(val_map)
            
            # [新增] 将生成的图片移动到结果文件夹，并改名
            for img_name in ['loss_and_lr.png', 'mAP.png']:
                if os.path.exists(img_name):
                    new_name = f"{img_name.split('.')[0]}_{args.ablation_mode}.png"
                    dest_path = os.path.join(experiment_dir, new_name)
                    shutil.move(img_name, dest_path)
                    print(f"已归档图片: {dest_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    # 设置默认路径为根目录下的路径
    parser.add_argument('--data-path', default=os.path.join(PROJECT_ROOT, 'coco2017'), help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='images per gpu')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument('--lr-step_size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[35, 45], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    
    # 设置默认输出路径为根目录下的 save_weights
    parser.add_argument('--output-dir', default=os.path.join(PROJECT_ROOT, 'save_weights'), help='root path to save weights')
    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true", help="test only")
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    
    parser.add_argument('--ablation-mode', default='full', type=str, 
                        help='Mode: full, baseline, no_scharr, no_log, no_gaussian, no_lfea')

    args = parser.parse_args()

    # 创建根目录
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