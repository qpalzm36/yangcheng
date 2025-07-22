#!/bin/bash

# 这是一个自动化测试脚本，用于运行生成器数据的构建过程。
# 它使用前续步骤（微调、建库）测试后产生的输出作为输入。
#
# 使用方法:
# ./aaai/test_build_generator_data.sh [FINETUNED_MODEL_PATH] [KB_DIR] [INPUT_DATA] [OUTPUT_DIR]
#
# 参数说明:
#   FINETUNED_MODEL_PATH: (可选) 微调后的检索器模型路径。
#                         默认: /data/yangcheng/aaai/retriever_finetuned_test
#   KB_DIR: (可选) 知识库目录，应包含 faiss_index.bin 和 knowledge_base_docs.jsonl。
#           默认: /data/yangcheng/aaai/knowledgebase_test
#   INPUT_DATA: (可选) 原始的生成器训练数据。
#               默认: /data/yangcheng/aaai/data/traindata/generatordata/generater_training_data_first.jsonl
#   OUTPUT_DIR: (可选) 保存处理后数据和图表的输出目录。
#               默认: /data/yangcheng/aaai/data/traindata/generatordata_test
BASE_DIR="/data/yangcheng"
# 如果任何命令执行失败，立即退出脚本
set -e

# --- 配置 ---
PYTHON_SCRIPT="${BASE_DIR}/aaai/model/6_build_generator_data.py"

# 参数的默认值
DEFAULT_MODEL_PATH="${BASE_DIR}/aaai/retriever_finetuned"
DEFAULT_KB_DIR="${BASE_DIR}/aaai/knowledgebase"
DEFAULT_INPUT_DATA="${BASE_DIR}/aaai/data/traindata/generatordata/generater_training_data_first.jsonl"
DEFAULT_OUTPUT_DIR="${BASE_DIR}/aaai/data/traindata/generatordata"

# 如果提供了命令行参数，则覆盖默认值
FINETUNED_MODEL_PATH=${1:-$DEFAULT_MODEL_PATH}
KB_DIR=${2:-$DEFAULT_KB_DIR}
INPUT_DATA=${3:-$DEFAULT_INPUT_DATA}
OUTPUT_DIR=${4:-$DEFAULT_OUTPUT_DIR}

# 从知识库目录动态生成文件路径
FAISS_INDEX_PATH="${KB_DIR}/faiss_index.bin"
KB_DOCS_PATH="${KB_DIR}/knowledge_base_docs.jsonl"

# 动态生成输出文件路径
OUTPUT_DATA_PATH="${OUTPUT_DIR}/generator_training_data_with_retrieval.jsonl"
OUTPUT_CHART_PATH="${OUTPUT_DIR}/retrieval_consistency_scores.png"

# --- 运行前检查 ---
# 移除 OpenAI API Key 检查，因为已在 Python 脚本中设置
# if [ -z "$OPENAI_API_KEY" ]; then
#     echo "错误: 环境变量 'OPENAI_API_KEY' 未设置。"
#     echo "请在运行脚本前设置您的API密钥: export OPENAI_API_KEY='your_key_here'"
#     exit 1
# fi

# 检查依赖文件和目录
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "错误: Python脚本未找到: '$PYTHON_SCRIPT'"; exit 1; fi
if [ ! -d "$FINETUNED_MODEL_PATH" ]; then echo "错误: 微调模型目录未找到: '$FINETUNED_MODEL_PATH'"; exit 1; fi
if [ ! -f "$FAISS_INDEX_PATH" ]; then echo "错误: FAISS索引未找到: '$FAISS_INDEX_PATH'"; exit 1; fi
if [ ! -f "$KB_DOCS_PATH" ]; then echo "错误: 知识库文档未找到: '$KB_DOCS_PATH'"; exit 1; fi
if [ ! -f "$INPUT_DATA" ]; then echo "错误: 输入数据文件未找到: '$INPUT_DATA'"; exit 1; fi

# 不再清理上一次运行的输出，而是确保目录存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "输出目录 '$OUTPUT_DIR' 不存在。正在创建..."
    mkdir -p "$OUTPUT_DIR"
fi

# --- 执行数据构建 ---
echo "--- 开始构建生成器数据测试 ---"
echo "模型路径:         $FINETUNED_MODEL_PATH"
echo "FAISS索引:       $FAISS_INDEX_PATH"
echo "知识库文档:     $KB_DOCS_PATH"
echo "输入数据:         $INPUT_DATA"
echo "输出数据路径:     $OUTPUT_DATA_PATH"
echo "输出图表路径:     $OUTPUT_CHART_PATH"
echo "---------------------------------"

python3 "$PYTHON_SCRIPT" \
    --model_path "$FINETUNED_MODEL_PATH" \
    --faiss_index_path "$FAISS_INDEX_PATH" \
    --knowledge_base_docs_path "$KB_DOCS_PATH" \
    --input_data_path "$INPUT_DATA" \
    --output_data_path "$OUTPUT_DATA_PATH" \
    --output_chart_path "$OUTPUT_CHART_PATH"
    

# --- 运行后检查 ---
if [ -f "$OUTPUT_DATA_PATH" ] && [ -f "$OUTPUT_CHART_PATH" ]; then
    echo "--- 测试成功结束 ---"
    echo "处理后的数据文件已保存在: '$OUTPUT_DATA_PATH'"
    echo "性能分析图表已保存在: '$OUTPUT_CHART_PATH'"
else
    echo "--- 测试失败 ---"
    if [ ! -f "$OUTPUT_DATA_PATH" ]; then echo "错误: 未找到输出数据文件: '$OUTPUT_DATA_PATH'"; fi
    if [ ! -f "$OUTPUT_CHART_PATH" ]; then echo "错误: 未找到输出图表文件: '$OUTPUT_CHART_PATH'"; fi
    exit 1
fi
