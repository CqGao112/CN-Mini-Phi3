import json

DATA_ROOT: str =  "/data1/gcq/dataset/"




data_path = DATA_ROOT + 'wikipedia-cn-20230720-filtered.json'
output_path = DATA_ROOT + 'tokenizer_wiki.txt'


with open(data_path,"r",encoding='utf-8') as f:
    data = json.load(f)
    with open(output_path,"w",encoding='utf-8') as f1:
        for item in data:
            iitem = item['completion']
            f1.write(iitem + '\n')
    print('------------')