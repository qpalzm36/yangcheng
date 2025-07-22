#!/bin/bash

# 这是一个统一的自动化脚本，用于按顺序运行整个AAAI项目的所有阶段。
# 包括检索器微调、知识库构建、生成器数据构建、生成器模型微调、推理和评估。
#
# 使用方法:
# /data/yangcheng/run_all_pipeline.sh [STAGE]
#
# 参数说明:
#   STAGE: (可选) 指定要开始运行的阶段。选项包括：
#          - 如果不提供参数，将运行所有阶段
#          - "retriever": 从检索器微调开始
#          - "build_knowledge": 从知识库构建开始
#          - "build_generator_data": 从生成器数据构建开始
#          - "finetune_generator": 从生成器模型微调开始
#          - "inference": 从推理开始
#          - "evaluation": 从评估开始
#
# 注意：脚本会从指定阶段开始运行，并继续执行后续所有阶段。
BASE_DIR="/data/yangcheng"
# 如果任何命令执行失败，立即退出脚本
set -e

# 获取参数，默认为空（运行所有阶段）
STAGE=${1:-all}

# 函数：运行单个脚本并检查结果
run_script() {
    local script_path="$1"
    local stage_name="$2"
    
    echo "----------------------------------------"
    echo "开始运行阶段：${stage_name}"
    echo "执行脚本：${script_path}"
    echo "----------------------------------------"
    
    bash "${script_path}"
    
    if [ $? -eq 0 ]; then
        echo "✓ 阶段 ${stage_name} 成功完成！"
    else
        echo "✗ 阶段 ${stage_name} 失败！请检查错误信息。"
        exit 1
    fi
    echo "----------------------------------------"
    echo ""
}

# 函数：运行带有参数的脚本
run_script_with_arg() {
    local script_path="$1"
    local stage_name="$2"
    local arg="$3"
    
    echo "----------------------------------------"
    echo "开始运行阶段：${stage_name}"
    echo "执行脚本：${script_path} ${arg}"
    echo "----------------------------------------"
    
    bash "${script_path}" "${arg}"
    
    if [ $? -eq 0 ]; then
        echo "✓ 阶段 ${stage_name} 成功完成！"
    else
        echo "✗ 阶段 ${stage_name} 失败！请检查错误信息。"
        exit 1
    fi
    echo "----------------------------------------"
    echo ""
}

# 根据STAGE参数决定从哪个阶段开始运行
case $STAGE in
    "all"|"retriever")
        run_script "${BASE_DIR}/run_retriever.sh" "检索器微调"
        ;;&
    "all"|"build_knowledge")
        run_script "${BASE_DIR}/run_buildknowlege.sh" "知识库构建"
        ;;&
    "all"|"build_generator_data")
        run_script "${BASE_DIR}/run_buildgeneratordata.sh" "生成器数据构建"
        ;;&
    "all"|"finetune_generator")
        run_script "${BASE_DIR}/run_finetunegeneratorqwen3B.sh" "Qwen3B生成器微调"
        run_script "${BASE_DIR}/run_finetunegeneratorqwen7B.sh" "Qwen7B生成器微调"
        run_script "${BASE_DIR}/run_finetunegeneratorllama7B.sh" "Llama7B生成器微调"
        run_script "${BASE_DIR}/run_finetunegeneratorllama8B.sh" "Llama8B生成器微调"
        ;;&
    "all"|"inference")
        run_script_with_arg "${BASE_DIR}/run_inferenceqwen3B.sh" "Qwen3B推理" "all"
        run_script_with_arg "${BASE_DIR}/run_inferenceqwen7B.sh" "Qwen7B推理" "all"
        run_script_with_arg "${BASE_DIR}/run_inferencellama7B.sh" "Llama7B推理" "all"
        run_script_with_arg "${BASE_DIR}/run_inferencellama8B.sh" "Llama8B推理" "all"
        ;;&
    "all"|"evaluation")
        run_script_with_arg "${BASE_DIR}/run_evaluateqwen3B.sh" "Qwen3B评估" "all"
        run_script_with_arg "${BASE_DIR}/run_evaluateqwen7B.sh" "Qwen7B评估" "all"
        run_script_with_arg "${BASE_DIR}/run_evaluateqwenllama7B.sh" "Llama7B评估" "all"
        run_script_with_arg "${BASE_DIR}/run_evaluateqwenllama8B.sh" "Llama8B评估" "all"
        ;;
    *)
        echo "错误：未知的阶段 '${STAGE}'"
        echo "有效选项为: all, retriever, build_knowledge, build_generator_data, finetune_generator, inference, evaluation"
        exit 1
        ;;
esac

echo "所有指定阶段已完成！"