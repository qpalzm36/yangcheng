import json
import os
import random
from tqdm import tqdm

# --- 文件路径配置 ---
STRUCTURED_RETRIEVER_DATA_PATH = "/data/yangcheng/aaai/data/traindata/retrieverdata/sampled_testretrival_structuredsamll.jsonl"
RETRIEVER_TRAINING_DATA_PATH = "/data/yangcheng/aaai/data/traindata/retrieverdata/retriever_training_data.jsonl"

def load_structured_data(file_path):
    """从jsonl文件加载结构化数据。"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line.strip()}")
    return data

def extract_steps_from_item(item):
    """从结构化数据项中提取(步骤键, 步骤值)的元组列表。"""
    steps = []
    step_num = 1
    while f"Step {step_num}" in item:
        key = f"Step {step_num}"
        value = item[key]
        steps.append((key, value))
        step_num += 1
    return steps

def create_retriever_training_data(structured_data):
    """为检索器创建对比学习训练数据，使用元素替换法构造负例。"""
    print("开始构建检索器训练数据 (元素替换法)...")
    
    if len(structured_data) < 2:
        print("错误: 至少需要两个不同的问题来构造跨问题负例。")
        return []

    training_samples = []
    
    # 遍历每个问题 P，将其作为正例的来源
    for i, p_item in enumerate(tqdm(structured_data, desc="生成训练样本")):
        p_problem = p_item.get("problem", "")
        p_steps = extract_steps_from_item(p_item)
        
        if not p_problem or not p_steps:
            continue
            
        # --- 新增: 处理 "问题 -> 步骤1" 的情况 ---
        p_step_1_key, p_step_1_value = p_steps[0]
        
        query_dict_s1 = {"Problem": p_problem}
        query_s1 = json.dumps(query_dict_s1, ensure_ascii=False)

        pos_dict_s1 = {"Problem": p_problem, p_step_1_key: p_step_1_value}
        positive_passage_s1 = json.dumps(pos_dict_s1, ensure_ascii=False)

        negatives_s1 = []

        # 负例 (顺序错误): 使用此问题后面的步骤
        if len(p_steps) > 1:
            for future_step_idx in range(1, len(p_steps)):
                future_step_key, future_step_val = p_steps[future_step_idx]
                neg_dict = {"Problem": p_problem, future_step_key: future_step_val}
                negatives_s1.append(json.dumps(neg_dict, ensure_ascii=False))

        # 负例 (内容错误): 使用另一个问题的步骤1
        q_idx = i
        while q_idx == i:
            q_idx = random.randint(0, len(structured_data) - 1)
        q_item = structured_data[q_idx]
        q_steps = extract_steps_from_item(q_item)
        if q_steps:
            q_step_1_value = q_steps[0][1]
            neg_dict = {"Problem": p_problem, p_step_1_key: q_step_1_value}
            negatives_s1.append(json.dumps(neg_dict, ensure_ascii=False))

        if negatives_s1:
            training_samples.append({
                "query": query_s1,
                "pos": [positive_passage_s1],
                "neg": negatives_s1
            })

        # --- 现有逻辑: 处理 "步骤i -> 步骤i+1" 的情况 ---
        if len(p_steps) < 2:
            continue # 如果少于2个步骤，无法构成(i, i+1)对，跳过下方循环

        for p_step_idx in range(len(p_steps) - 1):
            p_current_step_key, p_current_step_value = p_steps[p_step_idx]
            p_next_step_key, p_next_step_value = p_steps[p_step_idx + 1]

            # --- 构造查询和正例 ---
            query_dict = {"Problem": p_problem, p_current_step_key: p_current_step_value}
            query = json.dumps(query_dict, ensure_ascii=False)
            
            pos_dict = {
                "Problem": p_problem, 
                p_current_step_key: p_current_step_value, 
                p_next_step_key: p_next_step_value
            }
            positive_passage = json.dumps(pos_dict, ensure_ascii=False)
            
            negatives = []

            # --- 构造逻辑顺序负例 ---
            if len(p_steps) > p_step_idx + 2:
                for future_step_idx in range(p_step_idx + 2, len(p_steps)):
                    p_future_step_key, p_future_step_value = p_steps[future_step_idx]
                    neg_dict = {
                        "Problem": p_problem,
                        p_current_step_key: p_current_step_value,
                        p_future_step_key: p_future_step_value,
                    }
                    negatives.append(json.dumps(neg_dict, ensure_ascii=False))

            # --- 构造元素替换负例 ---
            # 随机选择一个不同的问题 Q
            q_idx = i
            while q_idx == i:
                q_idx = random.randint(0, len(structured_data) - 1)
            
            q_item = structured_data[q_idx]
            q_problem = q_item.get("problem", "")
            q_steps = extract_steps_from_item(q_item)

            if not q_problem or not q_steps:
                continue

            # 从 Q 中随机选择一个步骤
            q_step_idx = random.randint(0, len(q_steps) - 1)
            q_rand_step_key, q_rand_step_value = q_steps[q_step_idx]
            
            q_rand_next_step_value = ""
            if len(q_steps) > q_step_idx + 1:
                _, q_rand_next_step_value = q_steps[q_step_idx+1]
                
            # 负例1: {"Problem": "Q", "Step i": "B", "Step i+1": "C"} (上下文失配)
            neg_1 = {"Problem": q_problem, p_current_step_key: p_current_step_value, p_next_step_key: p_next_step_value}
            negatives.append(json.dumps(neg_1, ensure_ascii=False))

            # 负例2: {"Problem": "P", "Step i": "Y", "Step i+1": "C"} (前置步骤错误)
            neg_2 = {"Problem": p_problem, p_current_step_key: q_rand_step_value, p_next_step_key: p_next_step_value}
            negatives.append(json.dumps(neg_2, ensure_ascii=False))

            # 负例3: {"Problem": "P", "Step i": "B", "Step i+1": "Z"} (后续步骤错误)
            if q_rand_next_step_value:
                neg_3 = {"Problem": p_problem, p_current_step_key: p_current_step_value, p_next_step_key: q_rand_next_step_value}
                negatives.append(json.dumps(neg_3, ensure_ascii=False))

            # 负例4: {"Problem": "P", "Step i": "Y", "Step i+1": "Z"} (完全借用)
            if q_rand_next_step_value:
                neg_4 = {"Problem": p_problem, p_current_step_key: q_rand_step_value, p_next_step_key: q_rand_next_step_value}
                negatives.append(json.dumps(neg_4, ensure_ascii=False))

            training_samples.append({
                "query": query,
                "pos": [positive_passage],
                "neg": negatives
            })
            
    print(f"构建完成，共生成 {len(training_samples)} 个训练样本。")
    return training_samples

def save_training_data(training_data, output_path):
    """保存训练数据到文件。"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"检索器训练数据已保存到: {output_path}")

def main():
    """主函数"""
    structured_data = load_structured_data(STRUCTURED_RETRIEVER_DATA_PATH)
    if not structured_data:
        print("无法加载结构化数据，请先运行 1_structure_data.py")
        return
    print(f"成功加载 {len(structured_data)} 个结构化问题")
    
    training_data = create_retriever_training_data(structured_data)
    if not training_data:
        print("无法生成训练数据")
        return
    
    save_training_data(training_data, RETRIEVER_TRAINING_DATA_PATH)
    
    print("\n=== 数据样本示例 ===")
    if training_data:
        sample = training_data[-1]
        print("--- QUERY ---")
        print(json.dumps(json.loads(sample["query"]), indent=2, ensure_ascii=False))
        print("\n--- POSITIVE ---")
        print(json.dumps(json.loads(sample["pos"][0]), indent=2, ensure_ascii=False))
        print(f"\n--- NEGATIVES (showing first 4 of {len(sample['neg'])}) ---")
        for i, neg_str in enumerate(sample['neg'][:4]):
             print(f"  负例 {i+1}: \n{json.dumps(json.loads(neg_str), indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()