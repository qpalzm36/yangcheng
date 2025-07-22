#!/bin/bash

# 这是一个统一的自动化测试脚本，用于运行所有推理任务。
#
# 使用方法:
# /data/yangcheng/run_all_inference_tests.sh [TASK_TYPE] [GENERATOR_MODEL_PATH] [RETRIEVER_MODEL_PATH] [KB_DOCS_PATH] [FAISS_INDEX_PATH] [TEST_SET_PATH] [OUTPUT_LOG_PATH]
#
# 参数说明:
#   TASK_TYPE: (必需) 任务类型，可选值: default, aime, gsm, math
#   GENERATOR_MODEL_PATH: (可选) 生成器模型路径。
#                         默认: /data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct
#   RETRIEVER_MODEL_PATH: (可选) 检索器模型路径。
#                         默认: /data/yangcheng/aaai/retriever_finetuned
#   KB_DOCS_PATH: (可选) 知识库文档路径。
#                 默认: /data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl
#   FAISS_INDEX_PATH: (可选) FAISS索引路径。
#                     默认: /data/yangcheng/aaai/knowledgebase/faiss_index.bin
#   TEST_SET_PATH: (可选) 测试集路径。会根据TASK_TYPE设置默认值。
#   OUTPUT_LOG_PATH: (可选) 输出日志路径。会根据TASK_TYPE设置默认值。
BASE_DIR="/data/yangcheng"
# 如果任何命令执行失败，立即退出脚本
set -e

# --- 配置 ---
DEFAULT_GENERATOR_MODEL_PATH="${BASE_DIR}/aaai/generator_finetuned/Qwen-2.5-3B-Instruct"
DEFAULT_RETRIEVER_MODEL_PATH="${BASE_DIR}/aaai/retriever_finetuned"
DEFAULT_KB_DOCS_PATH="${BASE_DIR}/aaai/knowledgebase/knowledge_base_docs.jsonl"
DEFAULT_FAISS_INDEX_PATH="${BASE_DIR}/aaai/knowledgebase/faiss_index.bin"

# 根据任务类型设置默认的测试集和输出日志路径
TASK_TYPE="$1"
case "$TASK_TYPE" in
    "default")
        PYTHON_SCRIPT="${BASE_DIR}/aaai/model/8_run_inference_final_fixed.py"
        DEFAULT_TEST_SET_PATH="${BASE_DIR}/aaai/test/test120/structured_test_set.jsonl"
        DEFAULT_OUTPUT_LOG_PATH="${BASE_DIR}/aaai/results/inference_log_test12_qwen3B.jsonl"
        ;;
    "aime")
        PYTHON_SCRIPT="${BASE_DIR}/aaai/model/8_run_inference_final_fixedaime.py"
        DEFAULT_TEST_SET_PATH="${BASE_DIR}/AIME/AIME_2020_2024_filtered.jsonl"
        DEFAULT_OUTPUT_LOG_PATH="${BASE_DIR}/aaai/resultsaime/inference_log_aimeqwen3B.jsonl"
        ;;
    "gsm")
        PYTHON_SCRIPT="${BASE_DIR}/aaai/model/8_run_inference_final_fixedGSM.py"
        DEFAULT_TEST_SET_PATH="${BASE_DIR}/gsm8k/main/sampled.jsonl"
        DEFAULT_OUTPUT_LOG_PATH="${BASE_DIR}/aaai/resultsGSM/inference_log_gsm8k.jsonl"
        ;;
    "math")
        PYTHON_SCRIPT="${BASE_DIR}/aaai/model/8_run_inference_final_fixedmath.py"
        DEFAULT_TEST_SET_PATH="${BASE_DIR}/MATH-500/test.jsonl"
        DEFAULT_OUTPUT_LOG_PATH="${BASE_DIR}/aaai/resultsmath/inference_log_vllm_math500.jsonl"
        ;;
    "all")
        echo "运行所有推理任务..."
        bash "$0" "default"
        bash "$0" "aime"
        bash "$0" "gsm"
        bash "$0" "math"
        exit 0
        ;;
    *)
        echo "错误: 无效的任务类型 '$TASK_TYPE'。可选值: default, aime, gsm, math, all"
        exit 1
        ;;
esac

# 如果提供了命令行参数，则覆盖默认值
GENERATOR_MODEL_PATH=${2:-$DEFAULT_GENERATOR_MODEL_PATH}
RETRIEVER_MODEL_PATH=${3:-$DEFAULT_RETRIEVER_MODEL_PATH}
KB_DOCS_PATH=${4:-$DEFAULT_KB_DOCS_PATH}
FAISS_INDEX_PATH=${5:-$DEFAULT_FAISS_INDEX_PATH}
TEST_SET_PATH=${6:-$DEFAULT_TEST_SET_PATH}
OUTPUT_LOG_PATH=${7:-$DEFAULT_OUTPUT_LOG_PATH}

# --- 运行前检查 ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本未找到, 路径: '$PYTHON_SCRIPT'"
    exit 1
fi

if [ ! -d "$GENERATOR_MODEL_PATH" ]; then
    echo "错误: 生成器模型目录未找到, 路径: '$GENERATOR_MODEL_PATH'"
    exit 1
fi

if [ ! -d "$RETRIEVER_MODEL_PATH" ]; then
    echo "错误: 检索器模型目录未找到, 路径: '$RETRIEVER_MODEL_PATH'"
    exit 1
fi

if [ ! -f "$KB_DOCS_PATH" ]; then
    echo "错误: 知识库文档文件未找到, 路径: '$KB_DOCS_PATH'"
    exit 1
fi

if [ ! -f "$FAISS_INDEX_PATH" ]; then
    echo "错误: FAISS索引文件未找到, 路径: '$FAISS_INDEX_PATH'"
    exit 1
fi

if [ ! -f "$TEST_SET_PATH" ]; then
    echo "错误: 测试集文件未找到, 路径: '$TEST_SET_PATH'"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_LOG_PATH")"

# --- 执行推理 ---
echo "--- 开始运行推理任务: $TASK_TYPE ---"
echo "生成器模型路径:   $GENERATOR_MODEL_PATH"
echo "检索器模型路径:   $RETRIEVER_MODEL_PATH"
echo "知识库文档路径:   $KB_DOCS_PATH"
echo "FAISS索引路径:     $FAISS_INDEX_PATH"
echo "测试集路径:         $TEST_SET_PATH"
echo "输出日志路径:       $OUTPUT_LOG_PATH"
echo "---------------------------------"

# 使用python3执行脚本，并传入参数
python3 "$PYTHON_SCRIPT" \
    --generator_model_path "$GENERATOR_MODEL_PATH" \
    --retriever_model_path "$RETRIEVER_MODEL_PATH" \
    --knowledge_base_docs_path "$KB_DOCS_PATH" \
    --faiss_index_path "$FAISS_INDEX_PATH" \
    --test_set_path "$TEST_SET_PATH" \
    --output_log_path "$OUTPUT_LOG_PATH"

echo "--- 推理任务 $TASK_TYPE 完成 ---"