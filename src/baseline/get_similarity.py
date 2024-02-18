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


def get_similarity(dataset, retrieve_model, flag='test'):
    '''
    :param dataset: acl18, bigdata22, cikm18
    :param to_string: retrieve时是否将sequence转为string
    :param retrieve_model:
    :param retrieve_number: 最相似的10条。先多抽一些，就只用运行一次
    :return:
    '''
    data = []
    directory = '../../data/processed_data/' + flag + '/' + dataset + '_' + flag + '_list.json'
    with open(directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # start_index用于判定起始index，因为起始行写入csv时加header
    start_index = -1
    retrieve_result_list = []
    for i in range(len(data)):
        date1 = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 注意测试的时间范围
        if date1 >= get_start_date(dataset):
            if start_index == -1:
                start_index = i
            # 序列的index
            query_sequence_id = data[i]['sequence_id']
            print('index: ', query_sequence_id)
            # 抽取最相似的k个
            retrieve_result = get_similar_retrieval_list(query_date=date1,
                                                         query_sequence_id=query_sequence_id,
                                                         retrieve_model=retrieve_model,
                                                         flag=flag)
            # print(retrieve_result)
            # print('\n')
            retrieve_result_list.append({'index': [query_sequence_id],
                                         'query_sequence': [data[i]],
                                         'retrieve_result': [retrieve_result]})
    if flag != 'test':
        with open(('similar_candidates/' + flag + '_' + dataset + '_' + retrieve_model + '.pkl'), 'wb') as f:
            pickle.dump(retrieve_result_list, f)
    else:
        with open(('similar_candidates/test/' + flag + '_' + dataset + '_' + retrieve_model + '.pkl'), 'wb') as f:
            pickle.dump(retrieve_result_list, f)
    return 0


if __name__ == "__main__":
    for dataset in ['bigdata22', 'cikm18', 'acl18']:  # 'acl18'
        for retrieve_model in ['stock_maven']:  # 'bge', 'instructor', 'llm_embedder'
            # 这一步只是获取相似度
            # get_similarity(dataset=dataset, retrieve_model=retrieve_model, flag='train')
            # get_similarity(dataset=dataset, retrieve_model=retrieve_model,  flag='valid')
            get_similarity(dataset=dataset, retrieve_model=retrieve_model, flag='test')
