DATA_ROOT: str =  "/data1/gcq/dataset/"

import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def convert_lists_to_json(df):
    """Convert lists in DataFrame to JSON strings."""
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df


def save_parquet(file_path, data):
    if isinstance(data, list):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame or a list of lists")
    # Convert lists to JSON strings before saving to Parquet
    data = convert_lists_to_json(data)
    pq.write_table(pa.Table.from_pandas(data), file_path)
    print(f'Save {file_path} is ok!')

# for i in range(10):
#     file_path = DATA_ROOT + f'2020-40_zh_head_000{i}.jsonl'
#     save_path = DATA_ROOT + f'2020-40_zh_head_000{i}.parquet'
#     data = read_jsonl_file(file_path)
#     df = pd.DataFrame(data)
#     save_parquet(save_path, df)
    # print('i is saved successfully!')

file_path1 = DATA_ROOT + 'sft_data/train_1M_CN.json'
file_path2 = DATA_ROOT + 'sft_data/train_2M_CN.json'
save_path = DATA_ROOT + 'sft_data/belle_3M_cn.parquet'
data = read_jsonl_file(file_path1)
data1 = read_jsonl_file(file_path2)
data.extend(data1)

df = pd.DataFrame(data)
save_parquet(save_path, df)