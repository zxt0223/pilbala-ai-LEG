#!/bin/bash
# 文件名: zxt_run_all.sh
# 作用: 自动调度 LEGNet 消融实验 (Full, Baseline, No Scharr...)

# ================= 配置区域 (请修改这里) =================
# 1. 项目根目录
ROOT_DIR="/group/chenjinming/wyy/pytorch-pilipala-LEG"
# 2. 数据集路径
DATA_PATH="${ROOT_DIR}/coco2017"

# 3. 指定使用的显卡 (例如使用 1,2,3 号卡)
export CUDA_VISIBLE_DEVICES=1,2,6
# 4. 显卡数量 (必须与上面的数量一致)
NUM_GPU=3

# 5. 训练超参数
BATCH_SIZE=4        # 每张卡的 batch_size (总 batch = 4 * 3 = 12)
LR=0.012            # 初始学习率 (建议: 0.004 * NUM_GPU)
EPOCHS=24           # 训练轮数

# 6. 定义实验列表 (这些就是传给 train_multi_GPU.py 的参数)
LEG_MODES=("full" "baseline" "no_scharr" "no_gaussian" "no_lfea" "no_log")

# 7. 重复实验次数 (跑3次取平均)
REPEAT_TIMES=3

# =======================================================

cd $ROOT_DIR
mkdir -p zxt_logs
mkdir -p zxt_checkpoints

echo "========================================================"
echo "?? [LEGNet] 全自动实验脚本启动"
echo "使用显卡: $CUDA_VISIBLE_DEVICES (共 $NUM_GPU 张)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# 定义任务函数
run_experiment() {
    BACKBONE=$1
    MODE=$2
    RUN_IDX=$3
    
    # 构造实验名称
    EXP_NAME="legnet_${MODE}_run${RUN_IDX}"
    OUTPUT_DIR="zxt_checkpoints/${EXP_NAME}"
    LOG_FILE="zxt_logs/${EXP_NAME}.log"
    
    # 随机端口 (防止端口占用报错)
    MASTER_PORT=$(($RANDOM + 20000))
    
    echo "--------------------------------------------------------"
    echo "??  正在运行: ${EXP_NAME}"
    echo "    配置: Backbone=$BACKBONE | Mode=$MODE"
    echo "    日志: $LOG_FILE"
    
    mkdir -p $OUTPUT_DIR
    
    # [核心命令] torchrun 启动多卡训练
    torchrun --nproc_per_node=$NUM_GPU --master_port=$MASTER_PORT train_multi_GPU.py \
        --data-path "$DATA_PATH" \
        --backbone "$BACKBONE" \
        --ablation-mode "$MODE" \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --lr-steps 16 22 \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
        
    # 检查上一条命令是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "? 成功完成: ${EXP_NAME}"
    else
        echo "? 训练失败: ${EXP_NAME}"
        sleep 5
    fi
    
    echo "?? 冷却 10 秒..."
    sleep 10
}

# --- 开始循环执行 ---
for r in $(seq 1 $REPEAT_TIMES)
do
    echo ">>>>>>> 开始第 $r 轮实验 <<<<<<<"
    for mode in "${LEG_MODES[@]}"
    do
        run_experiment "legnet" "$mode" "$r"
    done
done

echo "========================================================"
echo "?? 所有实验结束！"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"