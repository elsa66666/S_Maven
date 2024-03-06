import json
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
import argparse
import pickle

import utils
from utils import get_llama_response, get_prompt, check_answer, get_qualified_retrieval_list, \
    get_similar_retrieval_list

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
setattr(utils, 'get_llama_response', get_llama_response)
setattr(utils, 'get_prompt', get_prompt)
setattr(utils, 'check_answer', check_answer)
setattr(utils, 'get_qualified_retrieval_list', get_qualified_retrieval_list)
setattr(utils, 'get_similar_retrieval_list', get_similar_retrieval_list)


def get_test_data(dataset):
    data = []
    directory = '../../data/processed_data/test/' + dataset + '_test_list.json'
    with open(directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # 在整个数据集中，query序列为起始日期的一年后，以确保每一条query都能检索近一年的序列
    query_start_date = datetime.strptime(data[0]['date_list'][0], format("%Y-%m-%d")) + relativedelta(years=1)
    return data, query_start_date


def predict_with_llm(test_dataset1, retrieve_model, retrieve_number, model_path1=''):
    # 加载大模型
    tokenizer = LlamaTokenizer.from_pretrained(model_path1)
    model = LlamaForCausalLM.from_pretrained(model_path1, device_map='auto')

    data, query_start_date = get_test_data(dataset=test_dataset1)

    with open(('similar_candidates/test/test_' + test_dataset1 + '_' + retrieve_model + '.pkl'), 'rb') as f:
        all_retrieve_result = pickle.load(f)

    start_index = -1
    retrieve_index = -1
    for i in range(len(data)):
        date1 = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 如果date在测试范围内，开始测试
        if date1 >= query_start_date:
            if start_index == -1:
                start_index = i
            # 序列的id
            query_sequence_id = data[i]['sequence_id']
            retrieve_index += 1
            print('index: ', query_sequence_id)

            # 输入llm的序列
            query_sequence = {
                'date_list': data[i]['date_list'],
                'open_list': data[i]['open_list'],
                'high_list': data[i]['high_list'],
                'low_list': data[i]['low_list'],
                'close_list': data[i]['close_list'],
                'adj_close_list': data[i]['adj_close_list'],
                'volume_list': data[i]['volume_list']
            }

            # all_retrieve_result[i]； 定位到这个query sequence,包含三列：index, query sequence, retrieve result
            # [0:retrieve_number]: 定位到前k条
            for item in all_retrieve_result:
                if item['index'][0] == query_sequence_id:
                    retrieve_result = item['retrieve_result'][0][0:retrieve_number]
                    break

            # 统计candidate的涨跌分布
            movement_count = [0, 0]  # [rise, fall]
            for candidate_item in retrieve_result:
                if candidate_item['candidate_data']['movement'] == 'rise':
                    movement_count[0] += 1
                elif candidate_item['candidate_data']['movement'] == 'fall':
                    movement_count[1] += 1
            print(movement_count)

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
            df1 = pd.DataFrame({'index': [str(query_sequence_id)],
                                'query': [str(query)],
                                'movement_count': [str(movement_count)],
                                'generated_answer': [str(response)],
                                'generated_label': [str(generated_label)],
                                'reference_label': [str(reference_answer)],
                                'check': [str(check)]})
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


def predict_without_llm(test_dataset1, retrieve_model, retrieve_number=5):
    data, query_start_date = get_test_data(dataset=test_dataset1)

    with open(('similar_candidates/test/test_' + test_dataset1 + '_' + retrieve_model + '.pkl'), 'rb') as f:
        all_retrieve_result = pickle.load(f)

    start_index = -1
    retrieve_index = -1
    for i in range(len(data)):
        date1 = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 如果date在测试范围内，开始测试
        if date1 >= query_start_date:
            if start_index == -1:
                start_index = i
            # 序列的id
            query_sequence_id = data[i]['sequence_id']
            retrieve_index += 1
            print('index: ', query_sequence_id)
            reference_answer = data[i]['movement']

            # all_retrieve_result[i]； 定位到这个query sequence,包含三列：index, query sequence, retrieve result
            # [0:retrieve_number]: 定位到前k条
            for item in all_retrieve_result:
                if item['index'][0] == query_sequence_id:
                    retrieve_result_list = item['retrieve_result'][0][0:retrieve_number]
                    break

            movement_count = [0, 0]  # [rise, fall]
            for candidate_item in retrieve_result_list:
                if candidate_item['candidate_data']['movement'] == 'rise':
                    movement_count[0] += 1
                elif candidate_item['candidate_data']['movement'] == 'fall':
                    movement_count[1] += 1

            predict_movement = ''
            if movement_count == [5, 0]:
                predict_movement == 'rise'
            elif movement_count == [0, 5]:
                predict_movement = 'fall'
            else:
                predict_movement = 'manual_check'

            query = ('query sequence: \n' + str(data[i]) + '\n' + 'candidate sequences: \n' + str(retrieve_result_list))

            if predict_movement == reference_answer:
                check = 1
            else:
                check = 0

            df1 = pd.DataFrame({'index': [query_sequence_id],
                                'query': [query],
                                'movement_count': movement_count,
                                'generated_label': [predict_movement],
                                'reference_label': [reference_answer],
                                'check': [check]})
            llm_output_directory = (
                    '../../retrieve_result/5.2.3_similar_retrieve/' + retrieve_model + '[non_llm_output]'
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
    parser.add_argument('--test_dataset', default='stock23', type=str)
    parser.add_argument('--llm_path', default='/data/xiaomengxi/stock_rag/src/model/LLMs/merged_7205')
    parser.add_argument('--retrieve_number', default=5)
    parser.add_argument('--retrieve_model', default='llm_embedder', choices=['dtw', 'ddtw', 'instructor',
                                                                             'bge', 'llm_embedder'])
    parser.add_argument('--to_string', default=False)
    args = parser.parse_args()

    predict_with_llm(test_dataset1=args.test_dataset,
                     retrieve_model=args.retrieve_model,
                     model_path1=args.llm_path,
                     retrieve_number=args.retrieve_number)

    '''
    predict_without_llm(test_dataset1=args.test_dataset,
                        retrieve_model=args.retrieve_model,
                        retrieve_number=5)
    '''
