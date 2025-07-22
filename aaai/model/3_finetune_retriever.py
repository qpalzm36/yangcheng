import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

# 固定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 1. 定义参数 ---

@dataclass
class ModelArguments:
    """与模型相关的参数"""
    model_name_or_path: str = field(
        default="/data/yangcheng/bge-large-en-v1.5",
        metadata={"help": "预训练模型的本地路径"}
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": "对比学习损失函数中的温度系数"}
    )

@dataclass
class DataArguments:
    """与数据相关的参数"""
    train_file: str = field(
        default="/data/yangcheng/aaai/data/traindata/retrieverdata/retriever_training_data.jsonl",
        metadata={"help": "训练数据文件路径 (.jsonl格式)"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "分词后的最大序列长度。BGE模型原生支持的最大长度为512。"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """自定义训练参数"""
    output_dir: str = field(
        default="/data/yangcheng/aaai/retriever_finetuned",
        metadata={"help": "模型checkpoint和最终模型的输出目录"}
    )
    num_train_epochs: float = field(default=3.0, metadata={"help": "训练的总轮数"})
    per_device_train_batch_size: int = field(default=32, metadata={"help": "每个设备的训练批量大小"})
    gradient_accumulation_steps: int = field(default=2, metadata={"help": "梯度累积步数"})
    learning_rate: float = field(default=1e-5, metadata={"help": "初始学习率"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "学习率预热的比例"})
    logging_dir: str = field(default='/data/yangcheng/aaai/trainlogs/retriever_logs', metadata={"help": "TensorBoard日志目录"})
    logging_steps: int = field(default=100, metadata={"help": "每隔多少步记录一次日志"})
    save_steps: int = field(default=100, metadata={"help": "每隔多少步保存一次模型"})
    fp16: bool = field(default=False, metadata={"help": "是否使用FP16混合精度训练"})
    bf16: bool = field(
        default=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        metadata={"help": "是否使用BF16混合精度训练 (需要Ampere或更新架构的GPU)"}
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "是否使用 torch.compile 加速模型 (需要PyTorch 2.0+)"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "保留自定义列(query, pos, neg)，交由DataCollator处理"}
    )


# --- 2. 自定义数据整理器 (Data Collator) ---

class ContrastiveDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        texts = []
        group_sizes = []  # <--- 新增: 用于记录每个样本的元素数量
        for feature in features:
            # 计算当前样本的总元素数 (1个query + N个pos + M个neg)
            num_elements = 1 + len(feature['pos']) + len(feature['neg'])
            group_sizes.append(num_elements)

            texts.append(feature['query'])
            texts.extend(feature['pos'])
            texts.extend(feature['neg'])

        batch = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 将group_sizes列表也放入batch中，以便Trainer可以访问
        batch['group_sizes'] = group_sizes
        return batch

# --- 3. 自定义训练器 (Trainer) ---

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 从inputs字典中弹出group_sizes，因为它不是模型本身的参数
        group_sizes = inputs.pop("group_sizes")

        outputs = model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # --- 关键修改: 使用torch.split代替buggy的view操作 ---
        # per_sample_embeddings是一个元组，每个元素是对应样本的所有向量
        per_sample_embeddings = torch.split(embeddings, group_sizes, dim=0)

        temp = self.model.config.temperature
        losses = []
        
        # 遍历批次中的每一个样本来独立计算损失
        for sample_embeddings in per_sample_embeddings:
            # sample_embeddings的形状是 (1+1+num_neg, D)
            query_embedding = sample_embeddings[0:1]      # Shape: (1, D)
            positive_embedding = sample_embeddings[1:2]   # Shape: (1, D)
            negative_embeddings = sample_embeddings[2:]   # Shape: (num_neg, D)

            # 如果一个样本因为某些原因没有负例，则跳过
            if negative_embeddings.shape[0] == 0:
                continue
            
            # 为了使用einsum，我们需要(B, C, D)的形状，这里B=1, C=num_neg
            negative_embeddings_reshaped = negative_embeddings.unsqueeze(0) # Shape: (1, num_neg, D)

            positive_scores = torch.einsum('bd,bd->b', query_embedding, positive_embedding).unsqueeze(-1)
            negative_scores = torch.einsum('bd,bcd->bc', query_embedding, negative_embeddings_reshaped)

            all_scores = torch.cat([positive_scores, negative_scores], dim=1)
            all_scores /= temp
            
            # 修改：使用 embeddings 的设备，而不是 model.device
            labels = torch.zeros(1, dtype=torch.long, device=embeddings.device)
            
            loss = F.cross_entropy(all_scores, labels)
            losses.append(loss)
        
        # 将批次中所有样本的损失求平均值
        final_loss = torch.stack(losses).mean()
        
        return (final_loss, {"outputs": outputs}) if return_outputs else final_loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.logging_dir:
        os.makedirs(training_args.logging_dir, exist_ok=True)
    
    print(f"从 {data_args.train_file} 加载数据...")
    train_dataset = load_dataset('json', data_files={'train': data_args.train_file})['train']
    print(f"数据集加载完成，共 {len(train_dataset)} 条样本。")

    print(f"从本地路径加载模型: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # 根据训练参数确定加载模型的数据类型
    torch_dtype = (
        torch.bfloat16 if training_args.bf16 else (
            torch.float16 if training_args.fp16 else torch.float32
        )
    )
    
    # --- 关键改动: 移除Flash Attention ---
    # BGE模型 (基于BertModel) 不支持Flash Attention 2，所以我们不再尝试启用它。
    # 我们仍然会从 BF16/FP16 的使用中获益。
    print(f"模型加载配置: torch_dtype={torch_dtype}, torch_compile={training_args.torch_compile}")

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype
        # 注意: 已移除 attn_implementation 参数，让 transformers 库为 BertModel 使用其默认的注意力机制
    )
    model.config.temperature = model_args.temperature
    
    if torch.cuda.is_available():
        print(f"CUDA可见设备: {os.getenv('CUDA_VISIBLE_DEVICES')}. 将使用GPU。")
    else:
        print("未发现CUDA设备，将在CPU上运行。")
    
    data_collator = ContrastiveDataCollator(tokenizer, data_args.max_seq_length)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("开始训练...")
    trainer.train()

    print(f"训练完成。正在将最终模型保存到 {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("模型保存完毕。")

if __name__ == "__main__":
    main()