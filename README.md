# Logic-Step
code for AAAI </br>
**Config** </br>
python=3.10 </br>
pip install openai pandas jsonlines scikit-learn numpy </br>
pip install accelerate matplotlib seaborn tqdm pyarrow </br>
pip install torch transformers sentence-transformers </br>
pip install einops transformers_stream_generator </br>
pip install tiktoken faiss-cpu datasets peft </br>
pip install vllm </br>

**retriever**:fine-tune bge-en-v1.5 </br>
finetuned model can be stored in "/data/yangcheng/aaai/retriever_finetuned" </br>

**generator**:fine-tune Qwen-2.5-3B-Instruct,Qwen2.5-Math-7B-Instruct,,Meta-Llama3-8B-Instruct,Llama-2-7b-chat-hf </br>
fintuned model can be stored in: </br>
"/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct" </br>
"/data/yangcheng/aaai/generator_finetuned/Qwen2.5-Math-7B-Instruct" </br>
"/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct" </br>
"/data/yangcheng/aaai/generator_finetuned/Llama-2-7b-chat-hf" </br>

**Command**</br>
nohup python 3_finetune_retriever.py > 3_finetune_retriever32.log 2>&1 & //训练编码器 </br>
nohup python 4_build_knowledge_base.py > 4_build_knowledge_base.log 2>&1 & //构建知识库 </br>
nohup python 6_build_generator_data.py > 6_build_generator_data.log 2>&1 & //构建训练数据 </br>
nohup python 7_finetune_generator2.py > 7_finetune_generator2.log 2>&1 & //训练生成器 </br>
nohup python 7_finetune_generatorllama2_7B.py > 7_finetune_generatorllama2_7B.log 2>&1 & //训练生成器 </br>
nohup python 7_finetune_generatorllama3_8B.py > 7_finetune_generatorllama3_8B.log 2>&1 & //训练生成器 </br>
nohup python 7_finetune_generatorqwne7B.py > 7_finetune_generatorqwen7B.log 2>&1 & //训练生成器 </br>

**The file path that may need to be modified** </br>
**3_finetune_retriever.py**: </br>
default="/data/yangcheng/aaai/data/traindata/retrieverdata/retriever_training_data.jsonl" //在代码的第37行  但是训练文件数据就在/aaai/data/traindata/retrieverdata/retriever_training_data.jsonl里面是没错的 </br>
default="/data/yangcheng/aaai/retriever_finetuned" //在代码的第49行 在aaai文件夹下已经新建好了一个空的retriever_finetuned文件夹 </br>

**4_build_knowledge_base.py**:</br>
STRUCTURED_DATA_PATH = "/data/yangcheng/aaai/data/traindata/structuredata/sampled_testbase_structured.jsonl" //输入的文件 在代码的11行 就在/aaai/data/traindata/structuredata/sampled_testbase_structured.jsonl </br>
FINETUNED_RETRIEVER_PATH = "/data/yangcheng/aaai/retriever_finetuned" //微调好的编码器文件存放位置 在代码的第14行 </br>
KB_DIR = "/data/yangcheng/aaai/knowledgebase" //知识库的保存文件夹 在代码的第17行 </br>
KNOWLEDGE_BASE_DOCS_PATH = os.path.join(KB_DIR, "knowledge_base_docs.jsonl") //存储知识库所有文档的jsonl文件位置 </br>
FAISS_INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin") // 存储Faiss索引的文件 </br>

**6_build_generator_data.py**:</br>
MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned" //微调好的编码器文件存放位置 在代码的第25行 </br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin" //知识库索引的文件位置 在代码的第26行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl" //存储知识库所有文档的jsonl文件位置 在代码的第27行</br>
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generater_training_data_first.jsonl" //存放第一次训练生成器代码的文件（带有检索标记的文档） 在代码的第28行</br>
OUTPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl" // 输出文件的路径 在代码的第29行</br>
OUTPUT_CHART_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/retrieval_consistency_scores.png" //输出图片的路径 在代码的第30行</br>

**7_finetune_generator2.py**:</br>
BASE_MODEL_PATH = "/data/share_weight/Qwen2.5-3B-Instruct"//要微调的模型的路径 第23行</br>
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl"// 之前构造生成器数据的输出路径 第22行</br>
OUTPUT_DIR = "/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct" //微调好的模型的位置 第23行</br>

**7_finetune_generatorllama2_7B.py**:</br>
BASE_MODEL_PATH = "/data/share_weight/Llama-2-7b-chat-hf" </br>
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl" </br>
OUTPUT_DIR = "/data/yangcheng/aaai/generator_finetuned/Llama-2-7b-chat-hf" </br>

**7_finetune_generatorllama3_8B.py**:</br>
BASE_MODEL_PATH = "/data/share_weight/Meta-Llama-3-8B-Instruct" </br>
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl" </br>
OUTPUT_DIR = "/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct" </br>

**7_finetune_generatorqwen7B.py**:</br>
BASE_MODEL_PATH = "/data/share_weight/Qwen2.5-Math-7B-Instruct" </br>
INPUT_DATA_PATH = "/data/yangcheng/aaai/data/traindata/generatordata/generator_training_data_with_retrieval.jsonl" </br>
OUTPUT_DIR = "/data/yangcheng/aaai/generator_finetuned/Qwen2.5-Math-7B-Instruct" </br>

**8_run_inference_final_fixed.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct" //微调好的生成器存放位置 第168行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned" //微调好的编码器位置 第169行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl" //存储知识库所有文档的jsonl文件位置 第170行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第171行</br>
TEST_SET_PATH = "/data/yangcheng/aaai/test/test120/structured_test_set.jsonl"//测试文件的存放位置 第172行</br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/results/inference_log_vllm_2stage_final_fixed_new_8.jsonl"//文件输出的位置 第173行</br>

**8_run_inference_final_fixedaime.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct" //微调好的生成器存放位置 第150行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned" //微调好的编码器位置 第151行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl" //存储知识库所有文档的jsonl文件位置 第152行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第153行</br>
TEST_SET_PATH = "/data/yangcheng/AIME/AIME_2020_2024_filtered.jsonl"//测试文件的存放位置 第154行</br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsaime/inference_log_aimeqwen3B.jsonl"//文件输出的位置 第155行</br>

**8_run_inference_final_fixedGSM.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct" //微调好的生成器存放位置 第137行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned" //微调好的编码器位置 第138行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl" //存储知识库所有文档的jsonl文件位置 第139行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第140行</br>
TEST_SET_PATH = "/data/yangcheng/gsm8k/main/sampled.jsonl" //测试文件的存放位置 第141行</br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsGSM/inference_log_gsm8k.jsonl" //文件输出的位置 第142行</br>

**8_run_inference_final_fixedmath.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen-2.5-3B-Instruct" //微调好的生成器存放位置 第168行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned" //微调好的编码器位置 第169行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl" //存储知识库所有文档的jsonl文件位置 第170行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第171行</br>
TEST_SET_PATH = "/data/yangcheng/MATH-500/test.jsonl" //测试文件的存放位置 第172行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsmath/inference_log_vllm_math500.jsonl" //文件输出的位置 第173行</br>

**8_run_inference_final_fixedllama2_7B.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Llama-2-7b-chat-hf"//微调好的生成器存放位置 第98行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第99行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第100行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第101行</br>
TEST_SET_PATH = "/data/yangcheng/aaai/test/test120/structured_test_set.jsonl"//测试文件的存放位置 第102行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/results/inference_log_vllm_llama2.jsonl"//文件输出的位置 第103行</br>
**8_run_inference_final_fixedllama2_7Baime.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Llama-2-7b-chat-hf"//微调好的生成器存放位置 第101行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第102行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第103行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第104行</br>
TEST_SET_PATH = "/data/yangcheng/AIME/AIME_2020_2024_filtered.jsonl"//测试文件的存放位置 第105行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsaime/inference_log_aime_llama2.jsonl"//文件输出的位置 第106行</br>
**8_run_inference_final_fixedllama2_7BGSM.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Llama-2-7b-chat-hf"//微调好的生成器存放位置 第115行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第116行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第117行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第118行</br>
TEST_SET_PATH = "/data/yangcheng/gsm8k/main/sampled.jsonl" //测试文件的存放位置 第119行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsGSM/inference_log_vllm_llama2_gsm8k.jsonl"//文件输出的位置 第120行</br>
**8_run_inference_final_fixedllama2_7Bmath.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Llama-2-7b-chat-hf"//微调好的生成器存放位置 第98行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第99行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第100行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第101行</br>
TEST_SET_PATH = "/data/yangcheng/MATH-500/test.jsonl" //测试文件的存放位置 第102行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsmath/inference_log_vllm_llama2_math500.jsonl"  //文件输出的位置 第103行</br>

**8_run_inference_final_fixedllama3_8B.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct"//微调好的生成器存放位置 第98行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第99行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第100行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第101行</br>
TEST_SET_PATH = "/data/yangcheng/aaai/test/test120/structured_test_set.jsonl"//测试文件的存放位置 第102行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/results/inference_log_vllm_llama3.jsonl"//文件输出的位置 第103行</br>
**8_run_inference_final_fixedllama3_8Baime.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct"//微调好的生成器存放位置 第101行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第102行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第103行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第104行</br>
TEST_SET_PATH = "/data/yangcheng/AIME/AIME_2020_2024_filtered.jsonl"//测试文件的存放位置 第105行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsaime/inference_log_vllm_llama3_8B_aime.jsonl"//文件输出的位置 第106行</br>
**8_run_inference_final_fixedllama3_8BGSM.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct"//微调好的生成器存放位置 第108行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第109行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第110行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第111行</br>
TEST_SET_PATH = "/data/yangcheng/gsm8k/main/sampled.jsonl" //测试文件的存放位置 第112行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsGSM/gsm8k_inference_log_llama3.jsonl"//文件输出的位置 第113行</br>
**8_run_inference_final_fixedllama3_8Bmath.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Meta-Llama-3-8B-Instruct"//微调好的生成器存放位置 第104行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第105行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第106行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第107行</br>
TEST_SET_PATH = "/data/yangcheng/MATH-500/test.jsonl" //测试文件的存放位置 第108行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsmath/inference_log_vllm_llama3_math500.jsonl"  //文件输出的位置 第109行</br>

**8_run_inference_final_fixedqwen7B.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen2.5-Math-7B-Instruct"//微调好的生成器存放位置 第168行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第169行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第170行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第171行</br>
TEST_SET_PATH = "/data/yangcheng/aaai/test/test120/structured_test_set.jsonl"//测试文件的存放位置 第172行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/results/inference_log_vllm_math7b.jsonl"//文件输出的位置 第173行</br>
**8_run_inference_final_fixedqwen7Baime.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen2.5-Math-7B-Instruct"//微调好的生成器存放位置 第136行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第137行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第138行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第139行</br>
TEST_SET_PATH = "/data/yangcheng/AIME/AIME_2020_2024_filtered.jsonl"//测试文件的存放位置 第140行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsaime/inference_log_aimeqwen7b.jsonl"//文件输出的位置 第141行</br>
**8_run_inference_final_fixedqwen7BGSM.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen2.5-Math-7B-Instruct"//微调好的生成器存放位置 第124行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第125行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第126行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第127行</br>
TEST_SET_PATH = "/data/yangcheng/gsm8k/main/sampled.jsonl" //测试文件的存放位置 第128行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsGSM/inference_log_gsm8k_qwen7b.jsonl"//文件输出的位置 第129行</br>
**8_run_inference_final_fixedqwen7Bmath.py**:</br>
GENERATOR_MODEL_PATH = "/data/yangcheng/aaai/generator_finetuned/Qwen2.5-Math-7B-Instruct"//微调好的生成器存放位置 第150行</br>
RETRIEVER_MODEL_PATH = "/data/yangcheng/aaai/retriever_finetuned"//微调好的编码器位置 第151行</br>
KNOWLEDGE_BASE_DOCS_PATH = "/data/yangcheng/aaai/knowledgebase/knowledge_base_docs.jsonl"//存储知识库所有文档的jsonl文件位置 第152行</br>
FAISS_INDEX_PATH = "/data/yangcheng/aaai/knowledgebase/faiss_index.bin"//存储Faiss索引的文件 第153行</br>
TEST_SET_PATH = "/data/yangcheng/MATH-500/test.jsonl" //测试文件的存放位置 第154行 </br>
OUTPUT_LOG_PATH = "/data/yangcheng/aaai/resultsmath/inference_log_math500_qwen7b.jsonl"  //文件输出的位置 第155行</br>
