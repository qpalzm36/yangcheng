#!/bin/bash

# 这是一个统一的自动化测试脚本，用于运行所有的推理任务（基于Llama-3-8B）。
#
# 使用方法:
# ./run_all_inference_tests_llama3.sh [TASK]
#
# 参数说明:
#   TASK: (可选) 指定要运行的任务。选项包括:
#         - "test120": 运行test120数据集推理
#         - "aime": 运行AIME数据集推理
#         - "gsm8k": 运行GSM8K数据集推理
#         - "math500": 运行MATH-500数据集推理
#         - 如果不提供参数，将运行所有任务
#
BASE_DIR="/data/yangcheng"
TASK="$1"

# 函数：运行单个推理任务
run_inference_task() {
    local task_name="$1"
    local python_script="$2"
    local test_set_path="$3"
    local output_log_path="$4"
    local cuda_devices="$5"
    local extra_params="$6"

    echo "开始运行任务: $task_name"
    echo "使用的Python脚本: $python_script"
    echo "测试集路径: $test_set_path"
    echo "输出日志路径: $output_log_path"
    echo "CUDA设备: $cuda_devices"
    echo "额外参数: $extra_params"
    echo "----------------------------------------"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES="$cuda_devices"

    # 运行Python脚本并传递参数
    python3 "$python_script" \
        --generator_model_path "${BASE_DIR}/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct" \
        --retriever_model_path "${BASE_DIR}/aaai/retriever_finetuned" \
        --knowledge_base_docs_path "${BASE_DIR}/aaai/knowledgebase/knowledge_base_docs.jsonl" \
        --faiss_index_path "${BASE_DIR}/aaai/knowledgebase/faiss_index.bin" \
        --test_set_path "$test_set_path" \
        --output_log_path "$output_log_path" \
        $extra_params

    if [ $? -eq 0 ]; then
        echo "任务 $task_name 成功完成。"
    else
        echo "任务 $task_name 失败。请检查错误信息。"
    fi
    echo "----------------------------------------"
}

# 根据参数决定运行哪些任务
if [ -z "$TASK" ] || [ "$TASK" == "all" ]; then
    echo "将运行所有推理任务..."
    run_inference_task "Test120" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8B.py" "${BASE_DIR}/aaai/test/test120/structured_test_set.jsonl" "${BASE_DIR}/aaai/results/inference_log_test120_llama3.jsonl" "1" ""
    run_inference_task "AIME" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8Baime.py" "${BASE_DIR}/AIME/AIME_2020_2024_filtered.jsonl" "${BASE_DIR}/aaai/resultsaime/inference_log_vllm_llama3_8B_aime.jsonl" "1" ""
    run_inference_task "GSM8K" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8BGSM.py" "${BASE_DIR}/gsm8k/main/sampled.jsonl" "${BASE_DIR}/aaai/resultsGSM/gsm8k_inference_log_llama3.jsonl" "1" ""
    run_inference_task "MATH-500" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8Bmath.py" "${BASE_DIR}/MATH-500/test.jsonl" "${BASE_DIR}/aaai/resultsmath/inference_log_vllm_llama3_math500.jsonl" "1" ""
elif [ "$TASK" == "test120" ]; then
    run_inference_task "Test120" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8B.py" "${BASE_DIR}/aaai/test/test120/structured_test_set.jsonl" "${BASE_DIR}/aaai/results/inference_log_test120_llama3.jsonl" "1" ""
elif [ "$TASK" == "aime" ]; then
    run_inference_task "AIME" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8Baime.py" "${BASE_DIR}/AIME/AIME_2020_2024_filtered.jsonl" "${BASE_DIR}/aaai/resultsaime/inference_log_vllm_llama3_8B_aime.jsonl" "1" ""
elif [ "$TASK" == "gsm8k" ]; then
    run_inference_task "GSM8K" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8BGSM.py" "${BASE_DIR}/gsm8k/main/sampled.jsonl" "${BASE_DIR}/aaai/resultsGSM/gsm8k_inference_log_llama3.jsonl" "1" ""
elif [ "$TASK" == "math500" ]; then
    run_inference_task "MATH-500" "${BASE_DIR}/aaai/model/8_run_inference_final_fixedllama3_8Bmath.py" "${BASE_DIR}/MATH-500/test.jsonl" "${BASE_DIR}/aaai/resultsmath/inference_log_vllm_llama3_math500.jsonl" "1" ""
else
    echo "无效的任务参数: $TASK"
    echo "有效选项: test120, aime, gsm8k, math500, all (或不提供参数运行所有任务)"
    exit 1
fi

echo "所有指定任务已完成。"