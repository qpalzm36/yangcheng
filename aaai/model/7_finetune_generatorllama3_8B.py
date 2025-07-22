import json
import os
import re
import math
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
import pandas as pd
import argparse  # <--- 新增导入

# 保留原有的GPU固定设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# --- Configuration ---
BASE_MODEL_PATH = "/data/share_weight/Meta-Llama-3-8B-Instruct"
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl"
OUTPUT_DIR = "/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct"

# --- Data Preparation ---

def create_decision_prompt(problem: str, previous_step: str = None) -> str:
    """
    【Llama-3 版】为决策阶段创建高度聚焦的提示。
    使用 Llama-3 Instruct 的官方聊天模板。
    """
    instruction = (
        "You are a planner for a math solving agent. Your task is to decide if you need to retrieve an example for the next step. "
        "Based on the problem and the previous step, your response MUST be one of two tags and nothing else: `<retrieval>` or `<no retrieval>`."
    )

    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {json.dumps(previous_step)}")
    
    input_content = f"{{{', '.join(input_parts)}}}"

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{input_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def create_content_prompt(problem: str, previous_step: Dict = None, retrieval_context: str = None) -> str:
    """
    【Llama-3 版】为内容生成阶段创建高度聚焦的提示。
    使用 Llama-3 Instruct 的官方聊天模板。
    """
    instruction = (
        "You are an expert math solver. Your goal is to generate the next step to solve a math problem. "
        "Your response MUST be a single JSON object with a key like `\"Step N\"` and the explanation as the value. "
        "Use clear, concise language. When you reach the final answer, include it in the CONCLUSION section with \\boxed{answer} format. "
        "If the problem is fully solved, add [END_OF_SOLUTION] *after* the JSON object."
    )

    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {json.dumps(previous_step)}")

    if retrieval_context:
        retrieval_str = json.dumps(retrieval_context)
        input_parts.append(f"\"retrieval_context\": {retrieval_str}")
    
    input_content = f"{{{', '.join(input_parts)}}}"

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{input_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def process_data(input_data_path: str) -> Dataset:
    """Loads and processes the JSONL data to create training samples."""
    print("Processing data file...")
    training_texts = []
    
    with open(input_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing problems"):
            data = json.loads(line)
            problem_text = data.get('problem', '')
            
            step_map = {
                int(re.search(r'Step (\d+)', k).group(1)): k 
                for k in data.keys() if 'Step' in k
            }
            if not step_map: continue
                
            sorted_step_nums = sorted(step_map.keys())

            for step_num in sorted_step_nums:
                key = step_map[step_num]
                step_content_raw = data[key]
                
                previous_step_dict = None
                if step_num > 1:
                    prev_step_num = step_num - 1
                    if prev_step_num in step_map:
                        prev_key = step_map[prev_step_num]
                        previous_step_dict = {prev_key: data[prev_key]}

                is_solution_end = "[END_OF_SOLUTION]" in step_content_raw
                
                retrieval_context = None
                p_match = re.search(r'(<p>.*</p>)', step_content_raw)
                if p_match:
                    retrieval_context = p_match.group(1)
                
                step_content_clean = re.sub(r'<p>.*</p>', '', step_content_raw).strip()
                step_content_clean = step_content_clean.replace("[END_OF_SOLUTION]", "").strip()

                decision_tag = ""
                tag_match = re.search(r'(<retrieval>|<no retrieval>)', key)
                if tag_match:
                    decision_tag = tag_match.group(1)
                
                previous_step_for_decision_str = None
                if previous_step_dict:
                    prev_key, prev_value = list(previous_step_dict.items())[0]
                    previous_step_for_decision_str = f"{json.dumps(prev_key)}: {json.dumps(prev_value)}"

                prompt1 = create_decision_prompt(problem_text, previous_step_for_decision_str)
                full_text1 = f"{prompt1}{decision_tag}<|eot_id|>"
                training_texts.append(full_text1)

                key_clean = key.replace(decision_tag, "").strip()
                target_json_str = json.dumps({key_clean: step_content_clean})

                prompt2 = create_content_prompt(problem_text, previous_step_dict, retrieval_context)
                
                assistant_output = target_json_str
                if is_solution_end:
                    assistant_output += " [END_OF_SOLUTION]"

                full_text2 = f"{prompt2}{assistant_output}<|eot_id|>"
                training_texts.append(full_text2)

    print(f"Created {len(training_texts)} training samples.")
    if training_texts:
        print("\n--- Example Training Samples ---")
        print("Sample 1 (Decision Learning):")
        print(training_texts[0])
        print("\nSample 2 (Content Generation Learning):")
        print(training_texts[1])
        if len(training_texts) > 2:
            print("\nSample 3 (Decision Learning):")
            print(training_texts[2])
            print("\nSample 4 (Content Generation Learning):")
            print(training_texts[3])
        print("----------------------------\n")
        
    return Dataset.from_dict({"text": training_texts})

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="微调Meta-Llama-3-8B-Instruct生成器模型。")
    parser.add_argument("--base_model_path", type=str, default="/data/share_weight/Meta-Llama-3-8B-Instruct", help="基础模型路径。")
    parser.add_argument("--input_data_path", type=str, default="/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl", help="输入的训练数据路径。")
    parser.add_argument("--output_dir", type=str, default="/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct", help="输出模型的保存目录。")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--batch_size", type=int, default=16, help="每个设备的批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数。")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率。")
    
    args = parser.parse_args()

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading base model and tokenizer from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add the new special tokens
    print("Adding special tokens: <retrieval>, <no retrieval>, [END_OF_SOLUTION]")
    new_special_tokens = {
        "additional_special_tokens": [
            "<retrieval>", 
            "<no retrieval>", 
            "[END_OF_SOLUTION]"
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(new_special_tokens)
    print(f"Added {num_added_toks} new special tokens.")
        
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize the model's token embeddings to accommodate the new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # --- 2. Prepare Dataset ---
    train_dataset = process_data(args.input_data_path)
    print(f"Total training samples: {len(train_dataset)}")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # --- 3. Configure Training (Simplified) ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        tf32=True,
        save_total_limit=3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- 4. Start Training ---
    print("\nStarting fine-tuning...")
    trainer.train()
    print("Training complete.")

    # --- 5. Save Final Model ---
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()