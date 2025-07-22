import openai
import os
import json
import time
from tqdm import tqdm
import re
from typing import Optional, Tuple
from collections import Counter

# --- 配置 ---
# 请确保您的API密钥已设置为环境变量，或者在此处直接指定
os.environ["OPENAI_API_KEY"] = "sk-DRNtKTg3hJLuU6J28jaasoxTgqKvKmqweXSViHhVAbcuEmSG"
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech/v1")
except TypeError:
    print("错误：OPENAI_API_KEY 未设置。请在您的环境中设置该变量。")
    exit(1)

MODEL = "gpt-4o-mini"
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# --- 文件路径配置 ---
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/sampled_testgenerate_structured.jsonl"
OUTPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generater_training_data_first.jsonl"

# --- Prompt 设计 (已根据您的反馈进行优化) ---
SYSTEM_PROMPT = """
You are an expert in mathematical reasoning and pedagogy. Your task is to analyze the reasoning leap from a `previous_step` to a `current_step` within the context of a given `problem`. Based on this analysis, you will decide if a student would need to retrieve a guiding example to understand or perform this specific step.

**Part 1: Your Evaluation Criteria**

Your entire judgment is based on the transition from the `previous_step` (or the problem description) to the `current_step`.

-   **You should recommend `<retrieval>` if the transition to the `current_step` represents a significant intellectual challenge or a common pitfall. Examples include**:
    1.  **Non-obvious Strategy or Knowledge**: The step requires a theorem, formula, or problem-solving trick that is not immediately apparent from the preceding context.
    2.  **Significant Logical Jump**: The step represents a major leap in reasoning that combines multiple smaller, unstated steps.
    3.  **Conceptual Flaw / Error-Prone Step**: The step involves a process where a conceptual misunderstanding is likely or has occurred. This is a form of **flawed logic**. Examples:
        - Using an incorrect formula for the situation.
        - Misapplying a complex procedure (e.g., integration by parts).
        - Incorrectly using a mathematical property (e.g., distributing a function like `f(a+b) = f(a)+f(b)`).

-   **You should recommend `<no retrieval>` if the `current_step` is a direct and simple consequence of the `previous_step`**, such as:
    1.  A straightforward calculation or algebraic manipulation.
    2.  The direct application of a very common and obvious formula.
    3.  A simple arithmetic slip (e.g., `2+3=6`) or a transcription error. These are considered minor execution errors, not conceptual gaps requiring retrieval.

**Part 2: Your Output Format**

Your response MUST be a JSON object with two keys: "tag" and "category".

1.  **"tag"**: Your decision from Part 1 (`<retrieval>` or `<no retrieval>`).

2.  **"category"**:
    *   If "tag" is `<no retrieval>`, the value for "category" MUST be "N/A".
    *   If "tag" is `<retrieval>`, you MUST choose one of the following categories that best describes your reasoning:
        - "Non-obvious Theorem/Formula"
        - "Clever Trick/Strategy"
        - "Significant Logical Jump"
        - "Conceptual Calculation Error"
        - "Other"
"""

def get_tag_and_category(problem: str, previous_step: Optional[str], current_step: str) -> Tuple[str, str]:
    """
    调用LLM API获取标签和分类。
    """
    prompt_data = {
        "problem": problem,
        "previous_step": previous_step if previous_step else "This is the first step.",
        "current_step": current_step
    }
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt_data, indent=2)}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=60,
                response_format={"type": "json_object"}
            )
            response_data = json.loads(response.choices[0].message.content)
            tag = response_data.get("tag")
            category = response_data.get("category")
            
            if tag in ["<retrieval>", "<no retrieval>"] and isinstance(category, str):
                return tag, category
            else:
                print(f"警告：收到无效数据 Tag: {tag}, Category: {category}。正在重试...")

        except json.JSONDecodeError as e:
            print(f"JSON解析失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    print("错误：达到最大重试次数，返回默认值。")
    return "<no retrieval>", "Error/Default"

def process_data():
    """
    主处理函数：读取、调用API、写入结果并收集统计信息。
    """
    print(f"正在从 '{INPUT_DATA_PATH}' 读取数据...")
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    
    all_problem_ratios = []
    retrieval_category_counts = Counter()

    with open(OUTPUT_DATA_PATH, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(tqdm(lines, desc="正在处理问题")):
            if not line.strip():
                continue
            
            data = json.loads(line)
            new_data = {"problem": data["problem"]}
            
            step_keys = sorted(
                [key for key in data if re.match(r'Step \d+', key)], 
                key=lambda x: int(x.split(' ')[1])
            )
            
            retrieval_steps_in_problem = 0
            
            previous_step_text = None
            for step_key in step_keys:
                current_step_text = data[step_key]
                
                print(f"\n[问题 {i+1}/{len(lines)}] 正在评估: {step_key}...")
                tag, category = get_tag_and_category(data["problem"], previous_step_text, current_step_text)
                print(f"  -> 获得标签: {tag} | 分类: {category}")
                
                if tag == "<retrieval>":
                    retrieval_steps_in_problem += 1
                    retrieval_category_counts[category] += 1

                new_key = f"{tag}{step_key}"
                new_data[new_key] = current_step_text
                previous_step_text = current_step_text
            
            if step_keys:
                problem_ratio = retrieval_steps_in_problem / len(step_keys)
                all_problem_ratios.append(problem_ratio)
                print(f"--- 问题 {i+1} 完成. 检索步骤占比: {problem_ratio:.2%} ({retrieval_steps_in_problem}/{len(step_keys)}) ---")

            f_out.write(json.dumps(new_data) + '\n')

    # --- 打印最终统计报告 ---
    print("\n\n" + "="*56)
    print(" " * 18 + "最终统计报告")
    print("="*56)
    
    if all_problem_ratios:
        average_ratio = sum(all_problem_ratios) / len(all_problem_ratios)
        print(f"\n📊 平均需要检索的步骤占比: {average_ratio:.2%}")
    else:
        print("\n📊 未处理任何问题，无法计算平均占比。")

    print("\n\n🔍 Retrieval 原因分类统计:")
    total_retrievals = sum(retrieval_category_counts.values())
    if total_retrievals > 0:
        print(f"{'分类':<35} {'数量':<10} {'占比':<10}")
        print("-" * 56)
        for category, count in retrieval_category_counts.most_common():
            percentage = (count / total_retrievals) * 100
            print(f"{category:<35} {count:<10} {f'{percentage:.2f}%':<10}")
    else:
        print("  - 未记录任何retrieval事件。")
    print("\n" + "="*56)

    print(f"\n处理完成！带标签的数据已保存到 '{OUTPUT_DATA_PATH}'")

if __name__ == "__main__":
    process_data()