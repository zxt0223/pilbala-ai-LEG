import os
import datetime

import torch
from torchvision.ops.misc import FrozenBatchNorm2d
from torch.utils.tensorboard import SummaryWriter
import transforms
from network_files import MaskRCNN
# from backbone import resnet50_fpn_backbone    # 原有的
from backbone.legnet import legnet_fpn_backbone # 新增的
from my_dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups


def create_model(num_classes, load_pretrain_weights=True):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    # backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)

    # model = MaskRCNN(backbone, num_classes=num_classes)

    # if load_pretrain_weights:
    #     # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    #     weights_dict = torch.load("maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
    #     for k in list(weights_dict.keys()):
    #         if ("box_predictor" in k) or ("mask_fcn_logits" in k):      # 删除与类别相关的权重
    #             del weights_dict[k]

    #     print(model.load_state_dict(weights_dict, strict=False))

    # return model
#-------------------------------------------------------------------------------↓修改后的
        # 选项 1: 使用原来的 ResNet50
    # backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
    
    # 选项 2: 使用新的 LEGNet (Tiny版配置)
    # 这里的 pretrain_path 可以指向你在 LEG Github 下载的 LWEGNet_tiny.pth
    # 如果没有权重，传 "" 空字符串即可
    backbone = legnet_fpn_backbone(pretrain_path="LWEGNet_tiny.pth") 
    
    model = MaskRCNN(backbone, num_classes=num_classes, min_size=1000, max_size=1333) #复写了min_size=1000, max_size=1333 

    if load_pretrain_weights:
        # 注意：如果你换了 backbone，原来的 'maskrcnn_resnet50_fpn_coco.pth' 
        # 的 backbone 部分权重就没法用了，只能加载 head 部分的权重，
        # 或者干脆不加载这一步，让模型多训练一会儿。
        # 简单的做法是：如果是换骨干实验，这里可以先设为 False，或者手动过滤权重。
        pass 

    return model
#-------------------------------------------------------------------------------↓修改后的

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    # 写入表头说明（编码格式：utf-8）
    det_header = """# 目标检测(bounding box)评估结果文件
# 每行格式: epoch:epoch_number bbox_AP50_95 bbox_AP50 bbox_AP75 bbox_AP_small bbox_AP_medium bbox_AP_large bbox_AR_1 bbox_AR_10 bbox_AR_100 loss learning_rate
# 列说明:
# 1. epoch: 训练轮次
# 2. bbox_AP50_95: COCO主要评估指标，IoU阈值从0.5到0.95的平均mAP
# 3. bbox_AP50: IoU阈值为0.5时的mAP（宽松指标）
# 4. bbox_AP75: IoU阈值为0.75时的mAP（严格指标）
# 5. bbox_AP_small: 小目标（面积<32²）的mAP
# 6. bbox_AP_medium: 中目标（32²<面积<96²）的mAP
# 7. bbox_AP_large: 大目标（面积>96²）的mAP
# 8. bbox_AR_1: 每张图最多检测1个目标时的平均召回率
# 9. bbox_AR_10: 每张图最多检测10个目标时的平均召回率
# 10. bbox_AR_100: 每张图最多检测100个目标时的平均召回率
# 11. loss: 当前epoch的平均训练损失
# 12. learning_rate: 当前学习率
#
# 训练开始时间: {}\n""".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    seg_header = """# 实例分割(segmentation mask)评估结果文件
# 每行格式: epoch:epoch_number seg_AP50_95 seg_AP50 seg_AP75 seg_AP_small seg_AP_medium seg_AP_large seg_AR_1 seg_AR_10 seg_AR_100 loss learning_rate
# 列说明:
# 1. epoch: 训练轮次
# 2. seg_AP50_95: 分割主要评估指标，IoU阈值从0.5到0.95的平均mAP
# 3. seg_AP50: IoU阈值为0.5时的分割mAP
# 4. seg_AP75: IoU阈值为0.75时的分割mAP
# 5. seg_AP_small: 小目标分割mAP
# 6. seg_AP_medium: 中目标分割mAP
# 7. seg_AP_large: 大目标分割mAP
# 8. seg_AR_1: 每张图最多检测1个目标时的分割召回率
# 9. seg_AR_10: 每张图最多检测10个目标时的分割召回率
# 10. seg_AR_100: 每张图最多检测100个目标时的分割召回率
# 11. loss: 当前epoch的平均训练损失
# 12. learning_rate: 当前学习率
#
# 训练开始时间: {}\n""".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 写入表头到文件（使用utf-8编码）
    with open(det_results_file, "w", encoding="utf-8") as f:
        f.write(det_header)
        f.write("# 数据开始:\n")
    
    with open(seg_results_file, "w", encoding="utf-8") as f:
        f.write(seg_header)
        f.write("# 数据开始:\n")

    # 在 train.py 中修改 data_transform
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # [新增] 在这里加入新的增强
            transforms.RandomColorJitter(brightness=0.3, contrast=0.3, prob=0.5), 
            transforms.RandomGaussianBlur(prob=0.3)
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(data_root, "train", data_transform["train"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:     #-1代表不使用
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> instances_val2017.json
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    # model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=False)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        # [新增] 写入 Loss 和 学习率
        writer.add_scalar('Loss/train', mean_loss.item(), epoch)
        writer.add_scalar('Learning_Rate', lr, epoch)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)
        # [新增] 写入 mAP (det_info[0] 是 mAP@0.5:0.95, det_info[1] 是 mAP@0.5)
        writer.add_scalar('mAP_bbox/0.5:0.95', det_info[0], epoch)
        writer.add_scalar('mAP_bbox/0.5', det_info[1], epoch)
        # 如果有分割结果
        if seg_info is not None:
            writer.add_scalar('mAP_segm/0.5:0.95', seg_info[0], epoch)
            writer.add_scalar('mAP_segm/0.5', seg_info[1], epoch)

        # write detection into txt（追加模式，使用utf-8编码）
        with open(det_results_file, "a", encoding="utf-8") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt（追加模式，使用utf-8编码）
        with open(seg_results_file, "a", encoding="utf-8") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "save_weights/model_{}.pth".format(epoch))
    # 循环结束后关闭
    writer.close()
    # 在文件末尾添加训练总结
    with open(det_results_file, "a", encoding="utf-8") as f:
        f.write(f"\n# 训练结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总训练轮次: {args.epochs}\n")
        if len(val_map) > 0:
            best_epoch = val_map.index(max(val_map))
            f.write(f"# 最佳bbox mAP@0.5:0.95: {max(val_map):.4f} (epoch {best_epoch})\n")
    
    with open(seg_results_file, "a", encoding="utf-8") as f:
        f.write(f"\n# 训练结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总训练轮次: {args.epochs}\n")

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    # 实例化 SummaryWriter
    writer = SummaryWriter(log_dir='logs')
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型               cuda:0 gpu 0 / cuda:1 gpu1 / cpu   
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='coco2017', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.004, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[30, 45], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练的batch size(如果内存/GPU显存充裕，建议设置更大)
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)