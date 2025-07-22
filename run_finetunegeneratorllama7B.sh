#!/bin/bash

# 这是一个自动化测试脚本，用于运行 Llama-2-7b-chat-hf 生成器模型的微调过程。
#
# 使用方法:
# /data/yangcheng/run_finetunegeneratorllama7B.sh [BASE_MODEL_PATH] [INPUT_DATA_PATH] [OUTPUT_DIR]
#
# 参数说明:
#   BASE_MODEL_PATH: (可选) 基础模型路径。
#                    默认: /data/share_weight/Llama-2-7b-chat-hf
#   INPUT_DATA_PATH: (可选) 输入的训练数据路径。
#                    默认: /data/yangcheng/aaai/data/traindata/generatordata_test/generator_training_data_with_retrieval.jsonl
#   OUTPUT_DIR: (可选) 输出模型的保存目录。
#               默认: /data/yangcheng/aaai/generator_finetuned_test/Llama-2-7b-chat-hf
BASE_DIR="/data/yangcheng"
# 如果任何命令执行失败，立即退出脚本
set -e

# --- 配置 ---
PYTHON_SCRIPT="${BASE_DIR}/aaai/model/7_finetune_generatorllama2_7B.py"

# 参数的默认值
DEFAULT_BASE_MODEL_PATH="/data/share_weight/Llama-2-7b-chat-hf"
DEFAULT_INPUT_DATA_PATH="${BASE_DIR}/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl"
DEFAULT_OUTPUT_DIR="${BASE_DIR}/aaai/generator_finetuned/Llama-2-7b-chat-hf"

# 如果提供了命令行参数，则覆盖默认值
BASE_MODEL_PATH=${1:-$DEFAULT_BASE_MODEL_PATH}
INPUT_DATA_PATH=${2:-$DEFAULT_INPUT_DATA_PATH}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}

# --- 运行前检查 ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本未找到, 路径: '$PYTHON_SCRIPT'"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "错误: 基础模型目录未找到, 路径: '$BASE_MODEL_PATH'"
    exit 1
fi

if [ ! -f "$INPUT_DATA_PATH" ]; then
    echo "错误: 输入数据文件未找到, 路径: '$INPUT_DATA_PATH'"
    exit 1
fi

# 清理上一次运行的输出
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "输出目录 '$OUTPUT_DIR' 不存在。正在创建..."
    mkdir -p "$OUTPUT_DIR"
fi


# --- 执行微调 ---
echo "--- 开始微调 Llama-2-7b-chat-hf 生成器模型测试 ---"
echo "基础模型路径:   $BASE_MODEL_PATH"
echo "输入数据路径:   $INPUT_DATA_PATH"
echo "输出目录:       $OUTPUT_DIR"
echo "---------------------------------"

# 使用python3执行脚本，并传入测试所需的参数
python3 "$PYTHON_SCRIPT" \
    --base_model_path "$BASE_MODEL_PATH" \
    --input_data_path "$INPUT_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5

# --- 运行后检查 ---
# 检查模型文件是否已创建
if [ -d "$OUTPUT_DIR" ]; then
    echo "--- 测试成功结束 ---"
    echo "微调后的模型已保存在 '$OUTPUT_DIR'"
else
    echo "--- 测试失败 ---"
    echo "训练结束后，在输出目录 '$OUTPUT_DIR' 中未找到模型文件。"
    exit 1
fi