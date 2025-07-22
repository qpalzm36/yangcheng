import os
import json
import re
from tqdm import tqdm
import torch
import torch.multiprocessing as mp 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from vllm import LLM, SamplingParams
import argparse  # <--- 新增导入

def protect_math_expressions(content: str):
    """【新】保护数学表达式，返回一个从占位符到表达式的映射。"""
    math_map = {}
    protected_content = content
    
    math_patterns = [
        (r'\\\\?frac\{[^}]*\}\{[^}]*\}', 'FRAC'),
        (r'\\\\?sqrt\{[^}]*\}', 'SQRT'),
        (r'\\\\?boxed\{[^}]*\}', 'BOXED'),
        (r'\\\\?text\{[^}]*\}', 'TEXT'),
        (r'\\\\?left\([^)]*\\\\?right\)', 'PAREN'),
        (r'\$[^$]*\$', 'DOLLAR'), # 保护美元符号包围的公式
    ]
    
    # 使用一个统一的占位符格式
    placeholder_template = "__MATH_EXPR_{}__"
    
    # 从后往前查找和替换，避免索引问题
    for pattern, _ in math_patterns:
        matches = list(re.finditer(pattern, protected_content))
        for match in reversed(matches):
            placeholder = placeholder_template.format(len(math_map))
            math_map[placeholder] = match.group(0)
            protected_content = protected_content[:match.start()] + placeholder + protected_content[match.end():]
            
    return protected_content, math_map

def restore_math_expressions(obj, math_map):
    """【新】使用映射恢复被保护的数学表达式。"""
    if isinstance(obj, dict):
        return {k: restore_math_expressions(v, math_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_math_expressions(item, math_map) for item in obj]
    elif isinstance(obj, str):
        result = obj
        # 多次迭代以处理嵌套的占位符
        for _ in range(3): # 迭代3次足以处理大多数嵌套
            for placeholder, expr in math_map.items():
                if placeholder in result:
                    result = result.replace(placeholder, expr)
        return result
    else:
        return obj

def escape_latex_for_json(s: str) -> str:
    """
    【全新 V2】在送入JSON解析器前，专门转义LaTeX命令中的反斜杠。
    例如，将`\frac`变为`\\frac`，但会忽略已经转义的 `\\frac`。
    """
    # 使用负向先行断言 `(?<!\\)` 来匹配前面不是反斜杠的单个反斜杠。
    # 这可以防止将 `\\frac` 错误地变成 `\\\frac`。
    return re.sub(r'(?<!\\)\\([a-zA-Z]+)', r'\\\\\1', s)

def parse_json_robust(raw_content: str, problem_id: int) -> dict:
    """
    【全新 V3】一个极其健壮的JSON解析器，这是净化环节的核心。
    它只负责一件事：将LLM的原始输出字符串，尽最大努力变成一个干净的Python字典。
    【关键修复】采用“先尝试，后修复”策略。
    """
    if not isinstance(raw_content, str):
        return {"error": "parsing_failed: input is not a string", "original_content": raw_content or ""}

    content = raw_content.strip()
    
    # 策略 1: 直接解析。对于绝大多数情况，这应该是可行的。
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 只有在因为无效转义序列（典型的LaTeX问题）失败时，才进入修复流程。
        if "Invalid \\escape" not in str(e):
            # 对于其他JSON错误（如括号不匹配），尝试寻找最后一个 '}'
            try:
                last_brace_index = content.rfind('}')
                if last_brace_index != -1:
                    return json.loads(content[:last_brace_index + 1])
            except json.JSONDecodeError:
                pass # 如果还是失败，就放弃，返回最终错误
            return {"error": f"parsing_failed: {str(e)}", "original_content": raw_content}

    # 策略 2: 修复LaTeX转义后重试。
    # 这个策略专门用来解决 `Invalid \escape` 错误
    try:
        content_escaped = escape_latex_for_json(content)
        return json.loads(content_escaped)
    except json.JSONDecodeError as e_escaped:
        # 如果修复后仍然失败，返回详细的错误信息
        return {"error": f"parsing_failed_after_escape: {str(e_escaped)}", "original_content": raw_content}


def _extract_answer_from_text(text: str) -> str:
    """
    【新 V3】从给定的文本中提取答案的辅助函数。
    【新增能力】可以从 [PROCESS] 块中提取最后的数值。
    """
    if not text:
        return None
    
    # 策略 1: 查找 \boxed{} (最高优先级)
    # 【关键修复】修复正则表达式以正确处理嵌套的 LaTeX，例如 \frac{a}{b}
    boxed_match = re.search(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 策略 2: 查找 [CONCLUSION]
    conclusion_match = re.search(r'\[CONCLUSION\](.*)', text, re.DOTALL)
    if conclusion_match:
        # 对从CONCLUSION提取的内容，再次尝试提取\boxed
        conclusion_content = conclusion_match.group(1).strip().strip('"')
        boxed_in_conclusion = re.search(r'\\boxed\{(.+?)\}', conclusion_content)
        if boxed_in_conclusion:
            return boxed_in_conclusion.group(1).strip()
        return conclusion_content
        
    # 策略 3: 查找 The final answer is
    result_match = re.search(r'The final answer is\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if result_match:
        return result_match.group(1).strip().strip('"')

    # 【备用策略】策略 4: 从 [PROCESS] 块中寻找最后的结果
    # 寻找 `... = <number>` 这种模式
    process_match = re.findall(r'\[PROCESS\].*?=\s*([-\d\.]+(?:/[-\d\.]+)?)\s*', text, re.DOTALL | re.IGNORECASE)
    if process_match:
        return process_match[-1].strip() # 返回最后一个匹配到的数字

    return None

def extract_final_answer_robust(generated_steps: list) -> str:
    """
    【重构】一个更简单、更可靠的答案提取器。
    它现在接收一个干净的Python对象列表，不再进行任何字符串清理或JSON解析。
    【关键修复】如果最后一步解析失败，它会尝试从原始输出中提取答案。
    """
    if not generated_steps:
        return None

    last_step_obj = generated_steps[-1]
    
    if not isinstance(last_step_obj, dict):
        return None

    # 【关键修复】检查最后一步是否为解析错误
    if 'error' in last_step_obj and 'original_content' in last_step_obj:
        # 如果解析失败，从原始文本中提取答案
        return _extract_answer_from_text(last_step_obj['original_content'])

    # 从成功解析的字典中提取文本内容
    step_text = ""
    for value in last_step_obj.values():
        if isinstance(value, str):
            step_text = value
            break
    
    return _extract_answer_from_text(step_text)


# --- 配置 ---
# 保留原有的GPU固定设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
MAX_NEW_TOKENS = 512  
MAX_STEPS = 10        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_decision_prompt(problem: str, previous_step: str = None) -> str:
    """为决策阶段创建提示"""
    instruction = (
        "You are a planner for a math solving agent. Your task is to decide if you need to retrieve an example for the next step. "
        "Based on the problem and the previous step, your response MUST be one of two tags and nothing else: `<retrieval>` or `<no retrieval>`."
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {previous_step}")

    input_content = f"{{{', '.join(input_parts)}}}"
    prompt = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_content}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def create_content_prompt(problem: str, previous_step=None, retrieval_context: str = None) -> str:
    """为内容生成阶段创建提示"""
    instruction = (
        "You are an expert math solver. Your goal is to generate the next step to solve a math problem. "
        "Your response MUST be a single JSON object with a key like `\"Step N\"` and the explanation as the value. "
        "Use clear, concise language. When you reach the final answer, include it in the CONCLUSION section with \\boxed{answer} format. "
        "If the problem is fully solved, add [END_OF_SOLUTION] *after* the JSON object." # <--- 修改在这里
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    
    if previous_step:
        if isinstance(previous_step, str):
            try:
                previous_step_dict = json.loads(previous_step)
            except json.JSONDecodeError:
                previous_step_dict = None
        else:
            previous_step_dict = previous_step
            
        if previous_step_dict:
            input_parts.append(f"\"previous_step\": {json.dumps(previous_step_dict)}")
    
    if retrieval_context:
        retrieval_str = json.dumps(retrieval_context)
        input_parts.append(f"\"retrieval_context\": {retrieval_str}")
    
    input_content = f"{{{', '.join(input_parts)}}}"
    prompt = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_content}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def run_retrieval(query, retriever_model, faiss_index, knowledge_base, k=1):
    """编码查询并从知识库中检索前k个文档"""
    query_embedding = retriever_model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    query_embedding_np = query_embedding.cpu().numpy().astype('float32')
    if query_embedding_np.ndim == 1:
        query_embedding_np = np.expand_dims(query_embedding_np, axis=0)
    faiss.normalize_L2(query_embedding_np)
    _, top_k_indices = faiss_index.search(query_embedding_np, k)
    retrieved_docs = [knowledge_base[i] for i in top_k_indices[0]]
    return retrieved_docs[0] if retrieved_docs else None

def run_inference_vllm(generator_model_path, retriever_model_path, knowledge_base_docs_path, faiss_index_path, test_set_path, output_log_path):
    # 先清理可能存在的Ray会话
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except:
        pass
    
    # 清理Ray临时文件
    import subprocess
    try:
        subprocess.run(["ray", "stop"], capture_output=True, timeout=10)
    except:
        pass
    
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    if os.path.exists(output_log_path):
        os.remove(output_log_path)

    print("正在加载 vLLM 生成器...")
    llm = LLM(
        model=generator_model_path,
        tensor_parallel_size=1,  # 改为1，因为只用一个GPU
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
        enforce_eager=True  # 添加这个参数，避免CUDA图编译问题
    )
    tokenizer = llm.get_tokenizer()

    print(f"正在加载检索器模型: {retriever_model_path}")
    retriever = SentenceTransformer(retriever_model_path, device=DEVICE)
    
    print(f"正在加载知识库: {knowledge_base_docs_path}")
    kb_docs = []
    with open(knowledge_base_docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text_parts = []
            if "Problem" in data:
                text_parts.append(f"Problem: {data['Problem']}")
            step_keys = sorted(
                [k for k in data.keys() if 'Step' in k],
                key=lambda k: int(re.search(r'Step (\d+)', k).group(1))
            )
            for key in step_keys:
                text_parts.append(f"{key}: {data[key]}")
            kb_docs.append("\n\n".join(text_parts))
    
    print(f"正在加载 FAISS 索引: {faiss_index_path}")
    index = faiss.read_index(faiss_index_path)
    print("所有模型和数据加载成功。")
    
    # 采样参数
    sampling_params_decision = SamplingParams(
        max_tokens=5,
        temperature=0.0,
        stop=["<retrieval>", "<no retrieval>"],
        include_stop_str_in_output=True,
        ignore_eos=True,
        skip_special_tokens=False
    )
    
    sampling_params_content = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.1,
        stop=["<|im_end|>", "[END_OF_SOLUTION]"],
        include_stop_str_in_output=True
    )

    with open(test_set_path, 'r', encoding='utf-8') as f_in:
        test_problems = [json.loads(line) for line in f_in]

    request_states = [
        {
            "problem_id": i,
            "problem_data": p,
            "generated_steps": [],
            "inference_trace": [],
            "finished": False
        } for i, p in enumerate(test_problems)
    ]

    # 两个阶段的推理循环
    for step in range(MAX_STEPS):
        active_requests = [r for r in request_states if not r["finished"]]
        if not active_requests:
            print("所有问题都已解决。")
            break
        
        print(f"\n--- 第 {step+1} 步, 处理 {len(active_requests)} 个请求 ---")

        # 阶段 1: 决策
        print("  - 阶段 1: 批量进行检索决策...")
        decision_prompts = [
            create_decision_prompt(
                r["problem_data"]["problem"],
                r["generated_steps"][-1] if r["generated_steps"] else None
            ) for r in active_requests
        ]
        decision_outputs = llm.generate(decision_prompts, sampling_params_decision)

        # 准备阶段 2 的输入
        content_prompts = []
        for i, request in enumerate(tqdm(active_requests, desc="  - 处理决策并准备执行")):
            decision_raw = decision_outputs[i].outputs[0].text.strip()
            needs_retrieval = "<retrieval>" in decision_raw

            retrieved_context = None
            retrieval_query = None
            if needs_retrieval:
                problem_text = request["problem_data"]["problem"]
                query_context = request['generated_steps'][-1] if request['generated_steps'] else ""
                query = f"{problem_text}\n{query_context}" if query_context else problem_text
                retrieval_query = query
                retrieved_context = run_retrieval(query, retriever, index, kb_docs)

            request["inference_trace"].append({
                "step_number": f"{step+1}.1_decision",
                "generator_prompt": decision_prompts[i],
                "raw_generator_output": decision_raw,
                "needs_retrieval_for_next_step": needs_retrieval,
                "retrieval_query": retrieval_query,
                "retrieved_context_for_next_step": retrieved_context
            })
            
            content_prompts.append(create_content_prompt(
                request["problem_data"]["problem"],
                request["generated_steps"][-1] if request["generated_steps"] else None,
                retrieved_context
            ))

        # 阶段 2: 内容生成
        print("  - 阶段 2: 批量生成步骤内容...")
        content_outputs = llm.generate(content_prompts, sampling_params_content)

        # 处理阶段 2 的结果并更新状态
        for i, request in enumerate(tqdm(active_requests, desc="  - 处理生成内容")):
            output_obj = content_outputs[i].outputs[0]
            raw_content = output_obj.text.strip()
            
            # 1. 检查我们约定的结束标记
            is_end_of_solution = raw_content.endswith("[END_OF_SOLUTION]")

            # 2. 从原始文本中移除结束标记（如果存在），得到需要被解析的纯JSON内容
            content_to_parse = raw_content.removesuffix("[END_OF_SOLUTION]").strip() if is_end_of_solution else raw_content
            
            # 3. 解析纯净的JSON内容
            parsed_step_obj = parse_json_robust(content_to_parse, request['problem_id'])
            request["generated_steps"].append(parsed_step_obj)
            
            # 检查是否出现解析错误。这种错误是代码层面的，应该终止。
            is_parsing_error = "error" in parsed_step_obj
            
            request["inference_trace"][-1].update({
                "execution_prompt": content_prompts[i],
                "execution_output_raw": raw_content,
                "parsed_step_content": parsed_step_obj,
                "is_end_of_solution": is_end_of_solution,
                "finish_reason": output_obj.finish_reason
            })
            
            # 4. 只有在三种情况下才终止对这个问题的推理：
            #    a) 发生 *解析错误*
            #    b) 我们约定的结束标记出现了
            #    c) 到达了最大步数限制
            if is_parsing_error or is_end_of_solution or len(request["generated_steps"]) >= MAX_STEPS:
                request["finished"] = True

    print("\n所有推理步骤完成或达到最大步数。")
    
    # 写入日志文件
    with open(output_log_path, 'w', encoding='utf-8') as f_out:
        for request in tqdm(request_states, desc="写入日志文件"):
            # 使用健壮的答案提取函数
            final_generated_answer = extract_final_answer_robust(
                request["generated_steps"]
            ) if request["generated_steps"] else None
            
            gold_steps_data = request["problem_data"]
            gold_steps = sorted(
                [{"key": k, "value": v} for k, v in gold_steps_data.items() if k.startswith("Step")],
                key=lambda x: int(re.search(r'(\d+)', x['key']).group(1))
            )
            
            log_entry = {
                "problem_id": request["problem_id"],
                "problem_text": request["problem_data"]["problem"],
                "gold_answer": request["problem_data"].get("answer"),
                "final_generated_answer": final_generated_answer,
                "generated_steps": request["generated_steps"],
                "gold_steps": gold_steps,
                "inference_trace": request["inference_trace"],
            }
            f_out.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    print(f"\n推理完成。结果已记录到: {output_log_path}")

def main():
    parser = argparse.ArgumentParser(description="运行推理任务。")
    parser.add_argument("--generator_model_path", type=str, default="/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct", help="生成器模型路径。")
    parser.add_argument("--retriever_model_path", type=str, default="/data/yangcheng/aaai/retriever_finetuned", help="检索器模型路径。")
    parser.add_argument("--knowledge_base_docs_path", type=str, default="/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl", help="知识库文档路径。")
    parser.add_argument("--faiss_index_path", type=str, default="/data/yangcheng/aaai/knowledgebase/faiss_index.bin", help="FAISS索引路径。")
    parser.add_argument("--test_set_path", type=str, default="/data/yangcheng/aaai/test/test120/structured_test_set.jsonl", help="测试集路径。")
    parser.add_argument("--output_log_path", type=str, default="/data/yangcheng/aaai/results/inference_log_vllm_2stage_final_fixed_new.jsonl", help="输出日志路径。")
    
    args = parser.parse_args()
    
    run_inference_vllm(
        args.generator_model_path,
        args.retriever_model_path,
        args.knowledge_base_docs_path,
        args.faiss_index_path,
        args.test_set_path,
        args.output_log_path
    )

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise e
        print("Start method already set.")
    main()