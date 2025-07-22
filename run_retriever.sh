#!/bin/bash

# 这是一个自动化测试脚本，用于运行 retriever 模型的微调过程。
# 它使用较小的批处理大小、训练轮数等配置，以便进行快速测试。
#
# 使用方法:
# ./aaai/test_finetune_retriever.sh [MODEL_PATH] [TRAIN_DATA_PATH] [OUTPUT_DIR]
#但是文件里面有默认配置可以修改
# 参数说明:
#   MODEL_PATH: (可选) 预训练模型的路径。
#               默认: /data/yangcheng/bge-large-en-v1.5
#   TRAIN_DATA_PATH: (可选) 训练数据(.jsonl)文件路径。
#                    默认: /data/yangcheng/aaai/data/traindata/retrieverdata/retriever_training_data_test.jsonl
#   OUTPUT_DIR: (可选) 用于保存微调后模型的目录。
#               默认: /data/yangcheng/aaai/retriever_finetuned_test
#
BASE_DIR="/data/yangcheng"
# 如果任何命令执行失败，立即退出脚本
set -e

# --- 配置 ---
# Python训练脚本的路径
PYTHON_SCRIPT="${BASE_DIR}/aaai/model/3_finetune_retriever.py"

# 参数的默认值
DEFAULT_MODEL_PATH="${BASE_DIR}/bge-large-en-v1.5"
DEFAULT_TRAIN_DATA_PATH="${BASE_DIR}/aaai/data/traindata/retrieverdata/retriever_training_data.jsonl"
DEFAULT_OUTPUT_DIR="${BASE_DIR}/aaai/retriever_finetuned"

# 如果提供了命令行参数，则覆盖默认值
MODEL_PATH=${1:-$DEFAULT_MODEL_PATH}
TRAIN_DATA_PATH=${2:-$DEFAULT_TRAIN_DATA_PATH}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}
LOGGING_DIR="${OUTPUT_DIR}/logs"

# --- 运行前检查 ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 训练脚本未找到,路径: '$PYTHON_SCRIPT'"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录未找到,路径: '$MODEL_PATH'"
    exit 1
fi

# --- 关键修改 ---
# 检查训练数据文件是否存在。如果不存在，则报错并退出。
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "错误: 训练数据文件未找到,路径: '$TRAIN_DATA_PATH'"
    echo "请确认文件路径正确，或通过第二个参数指定其路径。"
    exit 1
fi

# 不再清理上一次运行的输出，而是确保目录存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "输出目录 '$OUTPUT_DIR' 不存在。正在创建..."
    mkdir -p "$OUTPUT_DIR"
fi
if [ ! -d "$LOGGING_DIR" ]; then
    echo "日志目录 '$LOGGING_DIR' 不存在。正在创建..."
    mkdir -p "$LOGGING_DIR"
fi


# --- 执行训练 ---
echo "--- 开始微调测试 ---"
echo "模型路径:           $MODEL_PATH"
echo "训练数据路径:   $TRAIN_DATA_PATH"
echo "输出目录:       $OUTPUT_DIR"
echo "日志目录:      $LOGGING_DIR"
echo "---------------------------------"

# 使用python3执行脚本，并传入测试所需的参数
python3 "$PYTHON_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR" \
    --num_train_epochs 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --max_seq_length 512 \
    --save_steps 100 \
    --logging_steps 100 \
    --fp16 False \

# --- 运行后检查 ---
# 检查模型文件是否已创建
if [ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -f "$OUTPUT_DIR/model.safetensors" ]; then
    echo "--- 测试成功结束 ---"
    echo "微调后的模型已保存在 '$OUTPUT_DIR'"
else
    echo "--- 测试失败 ---"
    echo "训练结束后，在输出目录 '$OUTPUT_DIR' 中未找到模型文件。"
    exit 1
fi
