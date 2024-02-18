import argparse
import json
import pandas as pd
import pickle
import os
from datetime import datetime
import model_utils
from model_utils import get_llama_response, get_prompt_for_training, get_start_date, check_answer, \
    get_qualified_retrieval_list, \
    get_similar_retrieval_list

setattr(model_utils, 'get_llama_response', get_llama_response)
setattr(model_utils, 'get_prompt_for_training', get_prompt_for_training)
setattr(model_utils, 'get_start_date', get_start_date)
setattr(model_utils, 'check_answer', check_answer)
setattr(model_utils, 'get_qualified_retrieval_list', get_qualified_retrieval_list)
setattr(model_utils, 'get_similar_retrieval_list', get_similar_retrieval_list)


def convert_data_for_lm_score(dataset, retrieve_number=15):
    test_directory = '../../data/processed_data/train/' + dataset + '_train_list.json'
    data = []
    with open(test_directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    with open('../baseline/similar_candidates/llm_embedder/train_'+dataset+'_llm_embedder.pkl', 'rb') as f:
        all_retrieve_result = pickle.load(f)

    id_list = []
    index_list = []

    for sequence in data:
        id_list.append(sequence['sequence_id'])
        index_list.append(sequence['sequence_index'])

    start_index = -1
    retrieve_store_index = -1
    query_list = []
    for i in range(len(data)):
        date1 = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 如果date在测试范围内，开始测试
        if date1 >= get_start_date(dataset):
            if start_index == -1:
                start_index = i
            retrieve_store_index += 1
            # 序列的id
            # print(i)
            query_sequence_id = data[i]['sequence_id']
            print('index: ', query_sequence_id)

            # 输入llm的序列
            query_sequence = {"date_list": data[i]['date_list'],
                              "open_list": data[i]['open_list'],
                              "high_list": data[i]['high_list'],
                              "low_list": data[i]['low_list'],
                              "close_list": data[i]['close_list'],
                              "adj_close_list": data[i]['adj_close_list'],
                              "volume_list": data[i]['volume_list'],
                              }
            answer = data[i]["movement"]
            # all_retrieve_result[retrieve_store_index]； 定位到这个query sequence,包含三列：index, query sequence, retrieve result
            # [0:retrieve_number]: 定位到前k条
            for item in all_retrieve_result:
                if item['index'][0] == query_sequence_id:
                    retrieve_result = item['retrieve_result'][0][0:retrieve_number]
                    break
            positive_retrieve_result = retrieve_result[0:5]
            negative_retrieve_result = retrieve_result[5:15]
            query = get_prompt_for_training(query_id=query_sequence_id,
                                            positive_retrieve=positive_retrieve_result,
                                            negative_retrieve=negative_retrieve_result,
                                            query_sequence=query_sequence,
                                            answer=answer,
                                            id_list=id_list,
                                            index_list=index_list)
            query_list.append(query)
    return query_list


if __name__ == "__main__":
    all_query_list = []
    # regenerate_index()
    for dataset in ['acl18', 'bigdata22', 'cikm18']:
        dataset_query_list = convert_data_for_lm_score(dataset)
        all_query_list += dataset_query_list

    with open('all_embedder_training.json', "w") as file:
        for obj in all_query_list:
            # 将JSON对象转换为字符串
            json_str = json.dumps(obj)
            # 将字符串写入文件，并添加换行符
            file.write(json_str + "\n")
