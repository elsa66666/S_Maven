import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from datetime import datetime
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from random import sample
import argparse
import pickle

import utils
from utils import get_llama_response, get_prompt, get_start_date, check_answer, get_qualified_retrieval_list, \
    get_similar_retrieval_list

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
setattr(utils, 'get_llama_response', get_llama_response)
setattr(utils, 'get_prompt', get_prompt)
setattr(utils, 'get_start_date', get_start_date)
setattr(utils, 'check_answer', check_answer)
setattr(utils, 'get_qualified_retrieval_list', get_qualified_retrieval_list)
setattr(utils, 'get_similar_retrieval_list', get_similar_retrieval_list)


def test_llm_with_retrieve_result(test_dataset1, retrieve_model, retrieve_number, model_path1=''):
    # 加载大模型
    tokenizer = LlamaTokenizer.from_pretrained(model_path1)
    model = LlamaForCausalLM.from_pretrained(model_path1, device_map='auto')

    data = []
    test_directory = '../../data/processed_data/test/' + test_dataset1 + '_test_list.json'
    with open(test_directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    with open(('similar_candidates/test/test_' + test_dataset1 + '_' + retrieve_model + '.pkl'), 'rb') as f:
        all_retrieve_result = pickle.load(f)

    start_index = -1
    retrieve_index = -1
    for i in range(len(data)):
        date1 = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 如果date在测试范围内，开始测试
        if date1 >= get_start_date(test_dataset1):
            if start_index == -1:
                start_index = i
            # 序列的id
            query_sequence_id = data[i]['sequence_id']
            retrieve_index += 1
            print('index: ', query_sequence_id)

            # 输入llm的序列
            query_sequence = {"date_list": data[i]['date_list'],
                              "open_list": data[i]['open_list'],
                              "high_list": data[i]['high_list'],
                              "low_list": data[i]['low_list'],
                              "close_list": data[i]['close_list'],
                              "adj_close_list": data[i]['adj_close_list'],
                              }

            # all_retrieve_result[i]； 定位到这个query sequence,包含三列：index, query sequence, retrieve result
            # [0:retrieve_number]: 定位到前k条
            for item in all_retrieve_result:
                if item['index'][0] == query_sequence_id:
                    retrieve_result = item['retrieve_result'][0][0:retrieve_number]
                    break
            query = get_prompt(retrieve_number=retrieve_number,
                               query_sequence=query_sequence,
                               example_sequence_list=retrieve_result,
                               is_similar=True)
            if i == start_index:
                print("query: \n", query)
            # 股价预测的标答
            reference_answer = data[i]['movement']
            # llm返回的答案
            response = get_llama_response(index=query_sequence_id, query=query, model=model, tokenizer=tokenizer)
            print('response: ', response)
            # check llm是否回答正确
            generated_label, check = check_answer(generated_answer=response, reference_label=reference_answer)
            print('generated label: ', generated_label)
            print('reference label: ', reference_answer)
            print('check: ', check)
            print('\n')
            # 结果保存到csv中
            df1 = pd.DataFrame({'index': [query_sequence_id],
                                'query': [query],
                                'generated_answer': [response],
                                'generated_label': [generated_label],
                                'reference_label': [reference_answer],
                                'check': [check]})
            llm_output_directory = (
                    '../../retrieve_result/5.2.3_similar_retrieve/' + retrieve_model + '[llm_output]'
                    + test_dataset1
                    + '_similar_retrieve_'
                    + str(retrieve_number)
                    + '.csv')
            if i == start_index:  # 首行写入时加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
            else:  # 后面写入时不用加header
                df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
    print("Output directory: ", os.path.abspath(llm_output_directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='cikm18', choices=['acl18', 'bigdata22', 'cikm18'], type=str)
    parser.add_argument('--llm_path', default='/data/xiaomengxi/stock_rag/src/model/LLMs/merged_1310')
    parser.add_argument('--retrieve_number', default=5)
    parser.add_argument('--retrieve_model', default='instructor', choices=['dtw', 'ddtw', 'instructor',
                                                                           'bge', 'llm_embedder'])
    parser.add_argument('--to_string', default=False)
    args = parser.parse_args()

    test_llm_with_retrieve_result(test_dataset1=args.test_dataset,
                                  retrieve_model=args.retrieve_model,
                                  model_path1=args.llm_path,
                                  retrieve_number=args.retrieve_number)
