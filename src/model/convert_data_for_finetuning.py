import argparse
import json
import pandas as pd
import pickle
import os
from datetime import datetime
import model_utils
from model_utils import get_llama_response, get_prompt_for_finetune, get_start_date, check_answer, \
    get_qualified_retrieval_list, \
    get_similar_retrieval_list

setattr(model_utils, 'get_llama_response', get_llama_response)
setattr(model_utils, 'get_prompt_for_finetune', get_prompt_for_finetune)
setattr(model_utils, 'get_start_date', get_start_date)
setattr(model_utils, 'check_answer', check_answer)
setattr(model_utils, 'get_qualified_retrieval_list', get_qualified_retrieval_list)
setattr(model_utils, 'get_similar_retrieval_list', get_similar_retrieval_list)

'''
{
    "question":"",
    "cop":1,
    "opa":"rise",
    "opb":"fall",
    "id":"",
    "choice_type":"single"
}
'''


def convert_to_fine_tune(retrieve_model, flag, retrieve_number=5):
    f_write = open('finetune_stock_' + flag + '.json', "w")

    for dataset in ['acl18', 'bigdata22', 'cikm18']:
        data = []
        directory = '../../data/processed_data/' + flag + '/' + dataset + '_' + flag + '_list.json'

        with open(directory, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        if flag == ('train' or 'valid'):
            similar_directory = (
                '../baseline/similar_candidates/' + retrieve_model + '/' + flag + '_' + dataset + '_' + retrieve_model + '.pkl')
        elif flag == 'test':
            similar_directory = ('../baseline/similar_candidates/test/'+ dataset + '_' + retrieve_model + '_k=10.pkl')
        with open(similar_directory, 'rb') as f:
            all_retrieve_result = pickle.load(f)

        start_index = -1
        retrieve_store_index = -1

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
                                  "volume_list": data[i]['volume_list']
                                  }

                # item三列：index, query sequence, retrieve result
                # [0:retrieve_number]: 定位到前k条
                for item in all_retrieve_result:
                    if item['index'][0] == query_sequence_id:
                        retrieve_result = item['retrieve_result'][0][0:retrieve_number]
                        break
                positive_retrieve_result = retrieve_result[0:5]
                # negative_retrieve_result = retrieve_result[5:15]
                query = get_prompt_for_finetune(retrieve_number=retrieve_number,
                                                query_sequence=query_sequence,
                                                example_sequence_list=positive_retrieve_result,
                                                is_similar=True)
                # print("query: \n", query)
                # 股价预测的标答
                reference_answer = data[i]['movement']
                if reference_answer == 'rise':
                    cop = 1
                elif reference_answer == 'fall':
                    cop = 2
                item = {"question": query,
                        "cop": cop,
                        "opa": "rise",
                        "opb": "fall",
                        "id": query_sequence_id,
                        "choice_type": "single"
                        }
                f_write.write(json.dumps(item, ensure_ascii=False) + "\n")
    f_write.close()


if __name__ == "__main__":
    # convert_to_fine_tune(retrieve_model='llm_embedder', flag='train')
    # convert_to_fine_tune(retrieve_model='llm_embedder', flag='valid')
    convert_to_fine_tune(retrieve_model='llm_embedder', flag='test')
