import time
import os
import datetime

import torch
# from torchvision.ops.misc import FrozenBatchNorm2d 

# [核心修改 1] 强制禁用 NCCL P2P，解决服务器通讯卡死/报错问题，同时保持 NCCL 的高速度
os.environ["NCCL_P2P_DISABLE"] = "1"

import transforms
from my_dataset_coco import CocoDetection

# [核心修改 2] 导入 LEGNet backbone
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir
from torch.utils.tensorboard import SummaryWriter # [新增] TensorBoard

def create_model(num_classes, load_pretrain_weights=True):
    # [核心修改 3] 使用 LEGNet 构建 backbone
    # 这里的 pretrain_path 可以指向你在 LEG Github 下载的 LWEGNet_tiny.pth
    # 如果没有权重，传 "" 空字符串即可
    backbone = legnet_fpn_backbone(pretrain_path="LWEGNet_tiny.pth") 
    
    # [核心修改 4] 增加 min_size=1000 以提升小目标检测能力
    model = MaskRCNN(backbone, 
                     num_classes=num_classes,
                     min_size=1000, max_size=1333)

    if load_pretrain_weights:
        # LEGNet 结构不同，通常不建议加载基于 ResNet 的 COCO 预训练权重
        pass 

    return model

def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    # [新增] 初始化 TensorBoard，只在主进程初始化
    writer = None
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Data loading code
    print("Loading data")

    # [核心修改 5] 增加数据增强 (颜色抖动 + 高斯模糊)
    # 注意：请确保你的 transforms.py 文件末尾已经添加了这两个类的定义
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

    # load train data set
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    
    # load validation data set
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
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    # 兼容性设置：worker数量
    if args.workers > 0:
        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            timeout=30 if args.distributed else 0, # 分布式下建议设置timeout防止死锁
            collate_fn=train_dataset.collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=train_dataset.collate_fn)

    # 验证集 loader
    if args.workers > 0:
        data_loader_test = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            timeout=30 if args.distributed else 0,
            collate_fn=train_dataset.collate_fn)
    else:
        data_loader_test = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        # find_unused_parameters=True 有助于解决某些特定网络结构的死锁问题
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # [核心修改 6] 调整学习率策略：前35个epoch保持大学习率
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

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)
        
        # [新增] TensorBoard 记录 Loss
        if writer is not None:
            writer.add_scalar('Train/Loss', mean_loss.item(), epoch)
            writer.add_scalar('Train/Learning_Rate', lr, epoch)

        lr_scheduler.step()

        # evaluate after every epoch
        det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            val_map.append(det_info[1]) 

            # [新增] TensorBoard 记录 mAP
            if writer is not None:
                writer.add_scalar('Val/Det_mAP_0.5:0.95', det_info[0], epoch)
                writer.add_scalar('Val/Det_mAP_0.5', det_info[1], epoch)
                if seg_info is not None:
                    writer.add_scalar('Val/Seg_mAP_0.5:0.95', seg_info[0], epoch)
                    writer.add_scalar('Val/Seg_mAP_0.5', seg_info[1], epoch)

            with open(det_results_file, "a") as f:
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            if seg_info is not None:
                with open(seg_results_file, "a") as f:
                    result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")

        if args.output_dir:
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.rank in [-1, 0]:
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)
        if len(val_map) != 0:
            from plot_curve import plot_map
            plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path', default='coco2017', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    # [核心修改 7] 默认类别设为1（对应石头）
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # [核心修改 8] 单卡 Batch Size，根据你 24G 显存，设为 4 是安全的
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step_size', default=8, type=int, help='decrease lr every step-size epochs')
    # [核心修改 9] 学习率衰减策略调整为 [35, 45]，适应更强的数据增强
    parser.add_argument('--lr-steps', default=[35, 45], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='multi_train_weights', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true", help="test only")
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    # [核心修改 10] 默认不加载 COCO 预训练，因为使用了 LEGNet
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
    """
    # 注意：对程序来说，它只看得到这4张，会自动重新编号为 0,1,2,3
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_multi_GPU.py
    配置方案,GPU 数量,每卡 Batch Size,总 Batch Size,推荐学习率 (--lr),下降节点 (--lr-steps),说明
    方案 A,     2 张,       4,          8,              0.008,          35 45,              学习率翻倍
    方案 B,     3 张,       4,          12,             0.012,          35 45,              学习率 x3
    方案 C,     4 张,       4,          16,             0.016,          35 45,              学习率 x4
如果发现 Loss 震荡不下降（NaN 或 忽高忽低），说明学习率太大了。这时请不要犹豫，直接把推荐的学习率除以 2（例如 4 卡用 0.008）。LEGNet 有时对大学习率比较敏感。

nvidia-smi topo -m
这会显示 GPU 之间的连接矩阵（PIX, PXB, PHB, SYS 等）。如果显示 SYS，说明跨组需要走系统内存，这正是 NCCL_P2P_DISABLE=1 发挥作用的地方。
    
    """