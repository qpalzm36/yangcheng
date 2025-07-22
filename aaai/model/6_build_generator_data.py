import json
import os
import torch
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import argparse

# --- Configuration ---
# 请在这里填入您的OpenAI API密钥
# IMPORTANT: Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-DRNtKTg3hJLuU6J28jaasoxTgqKvKmqweXSViHhVAbcuEmSG"


openai.api_key = os.environ["OPENAI_API_KEY"]
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech/v1")
except TypeError:
    print("错误：OPENAI_API_KEY 未设置。请在您的环境中设置该变量。")
    exit(1)

# 默认路径
MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generater_training_data_first.jsonl"
OUTPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl"
OUTPUT_CHART_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/retrieval_consistency_scores.png"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# --- Helper Functions ---

def get_consistency_score(gold_step_content: str, retrieved_step_content: str) -> int:
    """
    Calls GPT-4o-mini to evaluate the logical consistency between the gold and retrieved steps.
    """
    prompt = f"""
    You are an expert in mathematical problem-solving and logical reasoning. Your task is to evaluate if a "Retrieved Step" provides a logically sound example or guide for deriving a "Gold Answer Step".

    **Gold Answer Step:** This is the correct, ground-truth step in a problem's solution.
    **Retrieved Step:** This is a step retrieved from a knowledge base, intended to help generate the Gold Answer Step.

    You must evaluate the logical consistency between them on a scale of 1 to 5. "Logical consistency" means the reasoning, method, or formula used in the Retrieved Step is applicable and helpful for reaching the Gold Answer Step, even if the specific numbers, variables, or context are different.

    **Scoring Rubric:**
    - **5 (Excellent):** The retrieved step uses the exact same logical reasoning or formula needed for the gold step. It's a perfect example.
    - **4 (Good):** The retrieved step uses a very similar and highly relevant logical process. It's a strong and helpful guide.
    - **3 (Fair):** The retrieved step demonstrates a somewhat related concept or a more general version of the required logic. It might provide a hint but isn't a direct guide.
    - **2 (Poor):** The retrieved step is on the same general topic but the specific logic is incorrect or not applicable to the gold step. It's more likely to confuse than help.
    - **1 (Irrelevant):** The retrieved step is completely unrelated to the logic required for the gold step.

    **Your Task:**
    Analyze the two steps below and provide a single integer score from 1 to 5 based on their logical consistency. Your response MUST be only the integer, with no other text or explanation.

    ---
    **Gold Answer Step:**
    {gold_step_content}
    ---
    **Retrieved Step:**
    {retrieved_step_content}
    ---

    **Your Score (1-5):**
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in mathematical problem-solving and logical reasoning. Your task is to evaluate the logical consistency between two solution steps and provide a single integer score from 1 to 5."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5,
        )
        score_text = response.choices[0].message.content.strip()
        # Extract the first integer found in the response
        match = re.search(r'\d+', score_text)
        if match:
            return int(match.group(0))
        else:
            print(f"Warning: Could not parse score from response: '{score_text}'. Defaulting to 1.")
            return 1
    except Exception as e:
        print(f"An error occurred during OpenAI API call: {e}")
        return 1 # Return a default low score on error

def find_previous_step_content(data: dict, current_step_num: int) -> str:
    """Finds the content of the step immediately preceding the current one."""
    prev_step_num = current_step_num - 1
    
    # Check for both <retrieval> and <no retrieval> tags
    possible_keys = [
        f"<retrieval>Step {prev_step_num}",
        f"<no retrieval>Step {prev_step_num}"
    ]
    
    for key in possible_keys:
        if key in data:
            return data[key]
            
    # Fallback for keys that might not have tags (e.g., "Step 1")
    if f"Step {prev_step_num}" in data:
        return data[f"Step {prev_step_num}"]
        
    return ""

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="Build generator data with retrieval.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the finetuned retriever model")
    parser.add_argument("--faiss_index_path", type=str, default=FAISS_INDEX_PATH, help="Path to the FAISS index file")
    parser.add_argument("--knowledge_base_docs_path", type=str, default=KNOWLEDGE_BASE_DOCS_PATH, help="Path to the knowledge base documents file")
    parser.add_argument("--input_data_path", type=str, default=INPUT_DATA_PATH, help="Path to the input data file")
    parser.add_argument("--output_data_path", type=str, default=OUTPUT_DATA_PATH, help="Path to save the processed data file")
    parser.add_argument("--output_chart_path", type=str, default=OUTPUT_CHART_PATH, help="Path to save the performance chart")
    args = parser.parse_args()

    # 使用命令行参数更新路径
    model_path = args.model_path
    faiss_index_path = args.faiss_index_path
    knowledge_base_docs_path = args.knowledge_base_docs_path
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    output_chart_path = args.output_chart_path

    print("Step 1: Loading resources...")
    # Load Retriever Model
    retriever_model = SentenceTransformer(model_path, device=DEVICE)

    # Load FAISS Index
    faiss_index = faiss.read_index(faiss_index_path)
    
    # Load Knowledge Base Documents
    with open(knowledge_base_docs_path, 'r', encoding='utf-8') as f:
        knowledge_base_docs = [line.strip() for line in f]
    
    print("Step 2: Processing data and performing retrieval...")
    consistency_scores = []
    processed_data = []

    with open(input_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Problems"):
            data = json.loads(line)
            new_data_item = data.copy()

            # Sort keys to ensure step order if not already guaranteed
            sorted_keys = sorted(data.keys(), key=lambda k: int(re.search(r'Step (\d+)', k).group(1)) if 'Step' in k else -1)

            for key in sorted_keys:
                if key.startswith("<retrieval>Step"):
                    match = re.search(r'Step (\d+)', key)
                    step_num = int(match.group(1))
                    
                    # --- Construct Query ---
                    problem_text = data['problem']
                    query_obj = {"Problem": problem_text}

                    if step_num > 1:
                        prev_step_content = find_previous_step_content(data, step_num)
                        if prev_step_content:
                            query_obj[f"Step {step_num - 1}"] = prev_step_content
                    
                    query_str = json.dumps(query_obj)
                    
                    # --- Encode and Search ---
                    query_embedding = retriever_model.encode(query_str, convert_to_tensor=True, device=DEVICE)
                    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
                    
                    D, I = faiss_index.search(query_embedding_np, k=1)
                    retrieved_doc_index = I[0][0]
                    
                    # --- Inject Retrieved Document ---
                    retrieved_doc_json_str = knowledge_base_docs[retrieved_doc_index]
                    retrieved_paragraph = f"<p>{retrieved_doc_json_str}</p>"
                    new_data_item[key] += retrieved_paragraph # Append to the original content
                    
                    # --- Evaluate Consistency ---
                    gold_step_content = data[key]
                    retrieved_doc = json.loads(retrieved_doc_json_str)
                    
                    # The content to compare is the value of the last key in the retrieved dict
                    retrieved_step_content = list(retrieved_doc.values())[-1]
                    
                    score = get_consistency_score(gold_step_content, retrieved_step_content)
                    consistency_scores.append(score)
            
            # --- Find the last step and append the END token ---
            last_step_key = None
            # Iterate backwards through sorted keys to find the last actual step
            for key in reversed(sorted_keys):
                if 'Step' in key:
                    last_step_key = key
                    break

            if last_step_key:
                new_data_item[last_step_key] += " [END_OF_SOLUTION]"

                
            processed_data.append(new_data_item)

    # --- Save Processed Data ---
    print(f"\nStep 3: Saving processed data to {output_data_path}...")
    with open(output_data_path, 'w', encoding='utf-8') as f_out:
        for item in processed_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # --- Report and Visualize Results ---
    if not consistency_scores:
        print("No <retrieval> tags found. No scores to report.")
        return

    print("\nStep 4: Analyzing and visualizing retrieval performance...")
    average_score = np.mean(consistency_scores)
    
    print(f"\n--- Retrieval Performance Report ---")
    print(f"Total Retrievals Performed: {len(consistency_scores)}")
    print(f"Average Logical Consistency Score: {average_score:.2f} / 5.0")
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    # Use a bar chart for discrete scores 1-5
    score_counts = {i: consistency_scores.count(i) for i in range(1, 6)}
    plt.bar(score_counts.keys(), score_counts.values(), color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Retrieval Logical Consistency Scores')
    plt.xlabel('Consistency Score (1-5)')
    plt.ylabel('Number of Retrievals')
    plt.xticks(range(1, 6)) # Ensure all score labels are shown
    plt.grid(axis='y', linestyle='--')

    # Add text labels on top of each bar
    for score, count in score_counts.items():
        if count > 0:
            plt.text(score, count + 0.1, str(count), ha='center', va='bottom')

    plt.savefig(output_chart_path)
    print(f"Performance chart saved to {output_chart_path}")
    # plt.show() # Uncomment to display the plot directly if in an interactive environment

if __name__ == "__main__":
    main()
