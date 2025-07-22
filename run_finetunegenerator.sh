#!/bin/bash

# 这是一个统一的自动化测试脚本，用于运行所有生成器模型的微调测试。
#
# 使用方法:
# /data/yangcheng/run_all_finetune_tests.sh [INPUT_DATA_PATH]
#
# 参数说明:
#   INPUT_DATA_PATH: (可选) 输入的训练数据路径。
#                    默认: /data/yangcheng/aaai/data/traindata/generatordata_test/generator_training_data_with_retrieval.jsonl

# 如果任何命令执行失败，立即退出脚本
BASE_DIR="/data/yangcheng"
set -e

# --- 配置 ---
DEFAULT_INPUT_DATA_PATH="${BASE_DIR}/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl"

# 如果提供了命令行参数，则覆盖默认值
INPUT_DATA_PATH=${1:-$DEFAULT_INPUT_DATA_PATH}

# --- 运行前检查 ---
if [ ! -f "$INPUT_DATA_PATH" ]; then
    echo "错误: 输入数据文件未找到, 路径: '$INPUT_DATA_PATH'"
    exit 1
fi

# 检查所有测试脚本是否存在
TEST_SCRIPTS=(
    "${BASE_DIR}/run_finetunegeneratorllama7B.sh"
    "${BASE_DIR}/run_finetunegeneratorllama8B.sh"
    "${BASE_DIR}/run_finetunegeneratorqwen3B.sh"
    "${BASE_DIR}/run_finetunegeneratorqwen7B.sh"
)

for SCRIPT in "${TEST_SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT" ]; then
        echo "错误: 测试脚本未找到, 路径: '$SCRIPT'"
        exit 1
    fi
    # 确保脚本有执行权限
    chmod +x "$SCRIPT"
done

echo "--- 开始运行所有生成器模型微调测试 ---"
echo "输入数据路径: $INPUT_DATA_PATH"
echo "---------------------------------"

# 按顺序运行每个测试脚本
for SCRIPT in "${TEST_SCRIPTS[@]}"; do
    echo "运行脚本: $SCRIPT"
    # 传递输入数据路径作为参数，其他参数使用默认值
    bash "$SCRIPT" "" "$INPUT_DATA_PATH" ""
    echo "脚本 $SCRIPT 运行完成。"
    echo "---------------------------------"
done

echo "--- 所有微调测试已完成 ---"