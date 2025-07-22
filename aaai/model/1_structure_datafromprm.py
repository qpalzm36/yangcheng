import openai
import os
import json
from tqdm import tqdm
import re

# --- 配置 ---
# 请确保您的API密钥和代理设置正确
os.environ["OPENAI_API_KEY"] = "sk-DRNtKTg3hJLuU6J28jaasoxTgqKvKmqweXSViHhVAbcuEmSG"
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech/v1")
except TypeError:
    print("错误：OPENAI_API_KEY 未设置。请在您的环境中设置该变量。")
    exit(1)

MODEL = "gpt-4o-mini"

# --- 文件路径配置 ---
PRM_INPUT_PATH = "/data/yangcheng/aaai/data/traindata/retrieverdata/sample_prm.jsonl"
STRUCTURED_OUTPUT_PATH = "/data/yangcheng/aaai/data/traindata/retrieverdata/sampled_testretrival_structured.jsonl"
ERROR_LOG_FILE = "/data/yangcheng/aaai/data/traindata/retrieverdata/prm_structure_error_log.txt"

# --- 用于GPT-4的系统Prompt (与1_structure_data.py完全相同) ---
SYSTEM_PROMPT = """
You are an expert in mathematical reasoning. Your task is to break down a solution to a math problem into a series of logical steps. Each step must follow this strict format: "[CONDITION] ... [PROCESS] ... [CONCLUSION] ..."

**Format Definition:**
- [CONDITION]: What is known at the beginning of this step? Include initial problem info and conclusions from previous steps.
- [PROCESS]: The core reasoning or calculation performed. Be concise but maintain logical and mathematical rigor.
- [CONCLUSION]: The result or new information derived at the end of this step.

**Language Requirement:**
- ALL responses must be in English, including all text descriptions
- Only mathematical expressions should remain in their original format (LaTeX)

**Mathematical Notation Requirements:**
- All mathematical formulas, equations, variables, and expressions MUST be formatted in LaTeX
- Use \\( ... \\) for inline math (e.g., \\( x^2 + y^2 = 10 \\))
- Use \\[ ... \\] for display math when needed
- Include ALL mathematical symbols, variables, numbers in mathematical context within LaTeX formatting
- Examples: \\( x = 2 \\), \\( y = -3x + 2 \\), \\( \\frac{a}{b} \\), \\( \\sqrt{c} \\)

**Conciseness Requirement:**
- Use minimal natural language while preserving essential meaning and logical coherence
- Focus on mathematical content rather than verbose explanations
- Eliminate unnecessary descriptive words but keep mathematical precision

**Example:**
Input: {"problem": "求直线 \\( y = -3x + 2 \\) 与圆 \\( x^2 + y^2 = 10 \\) 的交点坐标。", "solution": "将直线方程代入圆方程...展开整理...求根公式..."}

Output: {"problem": "求直线 \\( y = -3x + 2 \\) 与圆 \\( x^2 + y^2 = 10 \\) 的交点坐标。", "Step 1": "[CONDITION] 直线 \\( y = -3x + 2 \\)，圆 \\( x^2 + y^2 = 10 \\) [PROCESS] 代入直线方程到圆方程 [CONCLUSION] \\( x^2 + (-3x + 2)^2 = 10 \\)", "Step 2": "[CONDITION] \\( x^2 + (-3x + 2)^2 = 10 \\) [PROCESS] 展开并整理 [CONCLUSION] \\( 5x^2 - 6x - 3 = 0 \\)", "Step 3": "[CONDITION] \\( 5x^2 - 6x - 3 = 0 \\) [PROCESS] 应用求根公式 \\( x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\) [CONCLUSION] \\( x = \\frac{3 \\pm 2\\sqrt{6}}{5} \\)", "Step 4": "[CONDITION] \\( x = \\frac{3 + 2\\sqrt{6}}{5} \\)，\\( y = -3x + 2 \\) [PROCESS] 代入求 \\( y \\) [CONCLUSION] \\( y = \\frac{1 - 6\\sqrt{6}}{5} \\)", "Step 5": "[CONDITION] \\( x = \\frac{3 - 2\\sqrt{6}}{5} \\)，\\( y = -3x + 2 \\) [PROCESS] 代入求 \\( y \\) [CONCLUSION] \\( y = \\frac{1 + 6\\sqrt{6}}{5} \\)"}

**Instructions:**
1. You will receive a JSON object with "problem" and "solution" fields
2. Break down the "solution" into logical steps following the [CONDITION][PROCESS][CONCLUSION] format
3. Return a JSON object with "problem" field and step fields like "Step 1", "Step 2", etc.
4. Each step value should be a concise string in the format "[CONDITION] ... [PROCESS] ... [CONCLUSION] ..."
5. ALL mathematical content must be in LaTeX format using \\( ... \\) or \\[ ... \\]
6. Minimize natural language while preserving mathematical meaning and logical flow
7. Ensure all JSON strings are properly escaped
"""

def call_gpt4_for_structuring(problem, solution):
    """调用GPT-4 API进行结构化处理。"""
    user_input = {
        "problem": problem,
        "solution": solution
    }

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=prompt_messages,
            temperature=0.1,
            max_tokens=2048,
            top_p=0.9,
            stream=False,
            response_format={"type": "json_object"}
        )
        gpt_response_content = response.choices[0].message.content
        return json.loads(gpt_response_content), None
    except json.JSONDecodeError as e:
        error_info = {
            "error_type": "JSONDecodeError",
            "message": str(e),
            "gpt_response": gpt_response_content,
            "problem": problem,
            "solution": solution
        }
        return None, error_info
    except Exception as e:
        error_info = {
            "error_type": "UnknownError", 
            "message": str(e),
            "problem": problem,
            "solution": solution
        }
        return None, error_info

def process_prm_line(line):
    """处理 prm.jsonl 文件中的单行，将其转换为 problem 和 solution 字符串。"""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print(f"跳过无效的JSON行: {line.strip()}")
        return None, None

    problem = data.get("problem")
    if not problem:
        return None, None

    # 提取所有 "Step N" 键并按数字排序
    step_keys = sorted(
        [k for k in data.keys() if k.startswith('Step')], 
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    
    if not step_keys:
        return None, None

    # 将所有步骤内容合并成一个 "solution" 字符串
    solution_parts = [data[key].strip() for key in step_keys]
    solution = "\n".join(solution_parts)

    structured_result, error = call_gpt4_for_structuring(problem, solution)
    return structured_result, error

def main():
    """主函数，处理 prm 文件并追加到现有结构化文件中。"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("错误：OPENAI_API_KEY 环境变量未设置。")
        return

    if not os.path.exists(PRM_INPUT_PATH):
        print(f"错误：找不到输入文件 {PRM_INPUT_PATH}。")
        return
        
    print(f"开始处理文件: {PRM_INPUT_PATH}")
    
    # 确保输出和错误日志的目录存在
    os.makedirs(os.path.dirname(STRUCTURED_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)

    with open(PRM_INPUT_PATH, 'r', encoding='utf-8') as infile, \
         open(STRUCTURED_OUTPUT_PATH, 'a', encoding='utf-8') as outfile, \
         open(ERROR_LOG_FILE, 'a', encoding='utf-8') as error_file:

        lines = infile.readlines()
        success_count = 0
        error_count = 0
        
        for line in tqdm(lines, desc=f"结构化 {os.path.basename(PRM_INPUT_PATH)}"):
            result, error = process_prm_line(line)
            if result:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                outfile.flush()
                success_count += 1
            elif error:
                error_file.write(f"文件: {PRM_INPUT_PATH}\n")
                error_file.write(json.dumps(error, ensure_ascii=False) + '\n')
                error_file.flush()
                error_count += 1

    print("\n处理完成!")
    print(f"成功: {success_count}, 错误: {error_count}")
    print(f"结构化数据已追加到: {STRUCTURED_OUTPUT_PATH}")
    if error_count > 0:
        print(f"错误详情请查看日志: {ERROR_LOG_FILE}")

if __name__ == "__main__":
    main()