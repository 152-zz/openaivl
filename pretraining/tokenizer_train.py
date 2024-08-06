import sys

sys.path.append('../model')
sys.path.append('../tokenizer')
import pickle
import os
import json
from multiprocessing import Pool, cpu_count

folder_path = "../processed_data/WuDaoCorpus2.0_base_200G"
vocab_size = 60000


def process_file(filename):
    content_list = []
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            content_list.append(item['content'])
    print('data loaded')
    return content_list


if __name__ == "__main__":
    # 获取文件列表
    file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.json')]

    # 限制处理的文件数量
    file_list = file_list[:10]

    # 使用多进程处理文件
    with Pool(cpu_count()) as pool:
        content_lists = pool.map(process_file, file_list)

    # 合并所有 content_list
    content_list = [item for sublist in content_lists for item in sublist]

    # 打印或处理 content_list
    texts = content_list

    from bpe import Tokenizer

    tokenizer = Tokenizer(vocab_size)
    tokenizer.train(texts)

    # 保存 tokenizer 实例到文件
    with open('../output/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
