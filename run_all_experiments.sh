#!/bin/bash

# =================配置区域=================
# 定义你要跑的所有模式 (根据你的需求增减)
# 建议顺序：先跑 Full (满血版)，再跑 Baseline (最低版)，最后跑中间的
MODES=("full" "baseline" "no_scharr" "no_log" "no_gaussian" "no_lfea")

# 设置显卡 (请根据你的服务器实际情况修改)
export CUDA_VISIBLE_DEVICES=1,2,3
NUM_GPU=3

# 设置超参数 (所有实验必须保持一致，才公平)
BATCH_SIZE=4
LR=0.005      # 如果是从头训练建议 0.005 或 0.002；如果有权重可用 0.015
EPOCHS=50
DATA_PATH="/group/chenjinming/wyy/pytorch-pilipala-LEG/coco2017" # 请修改为你的真实路径
# =========================================

# 创建日志文件夹
mkdir -p logs_all

echo "========================================================"
echo "🚀 开始全自动消融实验流水线"
echo "待运行模式: ${MODES[*]}"
echo "========================================================"

for mode in "${MODES[@]}"
do
    echo ""
    echo "--------------------------------------------------------"
    echo ▶️  [$(date '+%Y-%m-%d %H:%M:%S')] 正在启动模式: $mode"
    echo "--------------------------------------------------------"

    # 构造日志文件名
    LOG_FILE="logs_all/training_${mode}.log"

    # 运行训练命令 (使用 torchrun)
    # 注意：这里调用的是 train_multi_GPU.py
    # 并且传入了 --ablation-mode "$mode"
    torchrun --nproc_per_node=$NUM_GPU train_multi_GPU.py \
        --data-path "$DATA_PATH" \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --lr-steps 35 45 \
        --ablation-mode "$mode" \
        --output-dir "save_weights" \
        2>&1 | tee "$LOG_FILE"

    # 检查上一条命令是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 模式 $mode 训练完成！日志已保存至 $LOG_FILE"
    else
        echo "❌ 模式 $mode 训练中途出错！请检查日志 $LOG_FILE"
        # 出错后可以选择 exit 1 停止，或者 continue 继续跑下一个
        # 这里选择继续跑下一个，以免一个挂了影响后面
    fi

    echo "💤 休息 10 秒，等待显存释放..."
    sleep 10
done

echo ""
echo "========================================================"
echo "🎉 所有消融实验已全部结束！"
echo "请去 save_weights/ 文件夹查看权重，去 logs_all/ 查看日志。"
echo "========================================================"