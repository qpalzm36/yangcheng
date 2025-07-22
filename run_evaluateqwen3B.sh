#!/bin/bash

# 这是一个统一的自动化测试脚本，用于运行所有评估任务。
#
# 使用方法:
# ./run_all_evaluation_tests.sh [TEST_TYPE]
#
# 参数说明:
#   TEST_TYPE: (可选) 指定要运行的测试类型。选项包括：
#              - "all": 运行所有测试 (默认)
#              - "standard": 运行标准测试
#              - "aime": 运行AIME测试
#              - "gsm": 运行GSM8K测试
#              - "math": 运行MATH-500测试
#

# 默认测试类型为"all"
TEST_TYPE=${1:-all}

# 基础路径配置
BASE_DIR="/data/yangcheng/aaai"
PYTHON_SCRIPT_DIR="${BASE_DIR}/model"

# 函数：运行单个评估任务
run_evaluation_task() {
    local script_name=$1
    local inference_log_path=$2
    local evaluation_report_path=$3
    local task_name=$4

    echo "开始运行 ${task_name} 评估任务..."
    echo "使用的脚本: ${script_name}"
    echo "推理日志路径: ${inference_log_path}"
    echo "评估报告路径: ${evaluation_report_path}"
    echo "----------------------------------------"

    python3 "${PYTHON_SCRIPT_DIR}/${script_name}" \
        --inference_log_path "${inference_log_path}" \
        --evaluation_report_path "${evaluation_report_path}"

    if [ $? -eq 0 ]; then
        echo "✓ ${task_name} 评估任务成功完成！"
    else
        echo "✗ ${task_name} 评估任务失败！请检查错误信息。"
    fi
    echo "----------------------------------------"
    echo ""
}

# 根据TEST_TYPE决定运行哪些测试
case $TEST_TYPE in
    "all")
        echo "运行所有评估测试..."
        run_evaluation_task "9_evaluate_results.py" "${BASE_DIR}/results/inference_log_test120_qwen3B.jsonl" "${BASE_DIR}/results/evaluation_report_test120_qwen3B.json" "标准测试"
        run_evaluation_task "9_evaluate_resultsaime.py" "${BASE_DIR}/resultsaime/inference_log_aimeqwen3B.jsonl" "${BASE_DIR}/resultsaime/evaluation_report_aimeqwen3B.json" "AIME测试"
        run_evaluation_task "9_evaluate_resultsGSM.py" "${BASE_DIR}/resultsGSM/inference_log_gsm8k.jsonl" "${BASE_DIR}/resultsGSM/evaluation_report_gsm8kqwen3B.json" "GSM8K测试"
        run_evaluation_task "9_evaluate_resultsmath.py" "${BASE_DIR}/resultsmath/inference_log_vllm_math500.jsonl" "${BASE_DIR}/resultsmath/evaluation_report_math500qwen3B.json" "MATH-500测试"
        ;;
    "standard")
        echo "运行标准评估测试..."
        run_evaluation_task "9_evaluate_results.py" "${BASE_DIR}/results/inference_log_test120_qwen3B.jsonl" "${BASE_DIR}/results/evaluation_report_test120_qwen3B.json" "标准测试"
        ;;
    "aime")
        echo "运行AIME评估测试..."
        run_evaluation_task "9_evaluate_resultsaime.py" "${BASE_DIR}/resultsaime/inference_log_aimeqwen3B.jsonl" "${BASE_DIR}/resultsaime/evaluation_report_aimeqwen3B.json" "AIME测试"
        ;;
    "gsm")
        echo "运行GSM8K评估测试..."
        run_evaluation_task "9_evaluate_resultsGSM.py" "${BASE_DIR}/resultsGSM/inference_log_gsm8k.jsonl" "${BASE_DIR}/resultsGSM/evaluation_report_gsm8kqwen3B.json" "GSM8K测试"
        ;;
    "math")
        echo "运行MATH-500评估测试..."
        run_evaluation_task "9_evaluate_resultsmath.py" "${BASE_DIR}/resultsmath/inference_log_vllm_math500.jsonl" "${BASE_DIR}/resultsmath/evaluation_report_math500qwen3B.json" "MATH-500测试"
        ;;
    *)
        echo "错误：未知的测试类型 '${TEST_TYPE}'"
        echo "有效选项为: all, standard, aime, gsm, math"
        exit 1
        ;;
esac

echo "所有指定测试已完成！"