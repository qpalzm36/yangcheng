#!/bin/bash

# 这是一个自动化测试脚本，用于运行知识库构建过程。
#
# 使用方法:
# ./aaai/test_build_knowledge_base.sh [STRUCTURED_DATA_PATH] [FINETUNED_MODEL_PATH] [OUTPUT_KB_DIR]
#
# 参数说明:
#   STRUCTURED_DATA_PATH: (可选) 结构化数据 (.jsonl) 文件路径。
#                         默认: /data/yangcheng/aaai/data/traindata/structuredata/sampled_testbase_structured.jsonl
#   FINETUNED_MODEL_PATH: (可选) 微调后的检索器模型路径。
#                         默认: /data/yangcheng/aaai/retriever_finetuned
#   OUTPUT_KB_DIR: (可选) 用于保存知识库的输出目录。
#                  默认: /data/yangcheng/aaai/knowledgebase_test

# 如果任何命令执行失败，立即退出脚本
BASE_DIR="/data/yangcheng"
set -e

# --- 配置 ---
# Python脚本的路径
PYTHON_SCRIPT="${BASE_DIR}/aaai/model/4_build_knowledge_base.py"

# 参数的默认值
DEFAULT_STRUCTURED_DATA_PATH="${BASE_DIR}/aaai/data/traindata/structuredata/sampled_testbase_structured.jsonl"
DEFAULT_FINETUNED_MODEL_PATH="${BASE_DIR}/aaai/retriever_finetuned"
DEFAULT_OUTPUT_KB_DIR="${BASE_DIR}/aaai/knowledgebase"

# 如果提供了命令行参数，则覆盖默认值
STRUCTURED_DATA_PATH=${1:-$DEFAULT_STRUCTURED_DATA_PATH}
FINETUNED_MODEL_PATH=${2:-$DEFAULT_FINETUNED_MODEL_PATH}
OUTPUT_KB_DIR=${3:-$DEFAULT_OUTPUT_KB_DIR}

# --- 运行前检查 ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本未找到, 路径: '$PYTHON_SCRIPT'"
    exit 1
fi

if [ ! -f "$STRUCTURED_DATA_PATH" ]; then
    echo "错误: 结构化数据文件未找到, 路径: '$STRUCTURED_DATA_PATH'"
    exit 1
fi

if [ ! -d "$FINETUNED_MODEL_PATH" ]; then
    echo "错误: 微调后的模型目录未找到, 路径: '$FINETUNED_MODEL_PATH'"
    exit 1
fi

# 不再清理上一次运行的输出，而是确保目录存在
if [ ! -d "$OUTPUT_KB_DIR" ]; then
    echo "输出目录 '$OUTPUT_KB_DIR' 不存在。正在创建..."
    mkdir -p "$OUTPUT_KB_DIR"
fi

# --- 执行知识库构建 ---
echo "--- 开始构建知识库测试 ---"
echo "结构化数据路径: $STRUCTURED_DATA_PATH"
echo "微调模型路径:   $FINETUNED_MODEL_PATH"
echo "知识库输出目录: $OUTPUT_KB_DIR"
echo "---------------------------------"

# 使用python3执行脚本
python3 "$PYTHON_SCRIPT" \
    --structured_data_path "$STRUCTURED_DATA_PATH" \
    --finetuned_retriever_path "$FINETUNED_MODEL_PATH" \
    --output_kb_dir "$OUTPUT_KB_DIR"

# --- 运行后检查 ---
# 检查 Faiss 索引和文档文件是否已创建
FAISS_INDEX_FILE="$OUTPUT_KB_DIR/faiss_index.bin"
DOCS_FILE="$OUTPUT_KB_DIR/knowledge_base_docs.jsonl"

if [ -f "$FAISS_INDEX_FILE" ] && [ -f "$DOCS_FILE" ]; then
    echo "--- 测试成功结束 ---"
    echo "知识库已成功构建并保存在 '$OUTPUT_KB_DIR'"
else
    echo "--- 测试失败 ---"
    if [ ! -f "$DOCS_FILE" ]; then
        echo "错误: 未找到知识库文档文件: '$DOCS_FILE'"
    fi
    if [ ! -f "$FAISS_INDEX_FILE" ]; then
        echo "错误: 未找到 Faiss 索引文件: '$FAISS_INDEX_FILE'"
    fi
    exit 1
fi