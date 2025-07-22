import pandas as pd
import os

def extract_aime_data(input_csv_path, output_jsonl_path, start_year, end_year):
    """
    从 AIME CSV 数据集中提取指定年份范围内的题目，并保存为 JSONL 文件。
    会过滤掉包含 '[asy]' 或 '[/asy]' 标签的题目。

    Args:
        input_csv_path (str): 输入的 AIME CSV 文件路径。
        output_jsonl_path (str): 输出的 JSONL 文件路径。
        start_year (int): 筛选的起始年份。
        end_year (int): 筛选的结束年份。
    """
    try:
        # 为CSV文件指定列名，因为原文件没有表头
        column_names = ['id', 'year', 'problem_num', 'question', 'answer', 'series']
        df = pd.read_csv(input_csv_path, header=None, names=column_names)

        # 将 'year' 列转换为数值类型，对于无法转换的值，将其设置为 NaN，然后删除这些行
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df.dropna(subset=['year'], inplace=True)
        df['year'] = df['year'].astype(int)

        # 筛选年份范围
        df_filtered_by_year = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

        # 排除包含 '[asy]' 或 '[/asy]' 的题目
        # 使用正则表达式 `\[/?asy\]` 来匹配这两种标签
        df_final = df_filtered_by_year[~df_filtered_by_year['question'].str.contains(r'\[/?asy\]', regex=True, na=False)]

        # 选择最终需要的列
        output_df = df_final[['year', 'question', 'answer']]

        # 确保输出目录存在
        output_dir = os.path.dirname(output_jsonl_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 以 JSON Lines 格式保存，覆盖旧文件
        output_df.to_json(output_jsonl_path, mode='w', orient='records', lines=True, force_ascii=False)
        
        print(f"成功从 '{input_csv_path}' 提取了 {len(output_df)} 条题目并保存到 '{output_jsonl_path}'。")

    except FileNotFoundError:
        print(f"错误: 文件 '{input_csv_path}' 未找到。")
    except Exception as e:
        print(f"发生一个错误: {e}")


if __name__ == "__main__":
    # 新功能：从 AIME 数据集提取题目
    input_csv = "/data/yangcheng/AIME/AIME_Dataset_1983_2024.csv"
    output_jsonl = "/data/yangcheng/AIME/AIME_2020_2024_filtered.jsonl"
    start_year = 2020
    end_year = 2024
    extract_aime_data(input_csv, output_jsonl, start_year, end_year)