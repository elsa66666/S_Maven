import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from random import sample
import argparse
import pickle

from sentence_transformers import SentenceTransformer, util

import utils
from utils import get_llama_response, get_prompt, check_answer, get_qualified_retrieval_list, \
    get_similar_retrieval_list, get_test_data

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
setattr(utils, 'get_llama_response', get_llama_response)
setattr(utils, 'get_prompt', get_prompt)
setattr(utils, 'check_answer', check_answer)
setattr(utils, 'get_qualified_retrieval_list', get_qualified_retrieval_list)
setattr(utils, 'get_similar_retrieval_list', get_similar_retrieval_list)
setattr(utils, 'get_test_data', get_test_data)


def get_similarity(dataset, retrieve_model, flag='test'):
    '''
    :param flag:
    :param dataset: acl18, bigdata22, cikm18
    :param to_string: retrieve时是否将sequence转为string
    :param retrieve_model:
    :param retrieve_number: 最相似的10条。先多抽一些，就只用运行一次
    :return:
    '''
    data, query_start_date = get_test_data(dataset=dataset)

    # start_index用于判定起始index，因为起始行写入csv时加header
    start_index = -1
    retrieve_result_list = []

    for i in range(len(data)):
        query_date = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 注意测试的时间范围
        if query_date >= query_start_date:
            if start_index == -1:
                start_index = i
            # 序列的id
            query_sequence_id = data[i]['sequence_id']
            print('index: ', query_sequence_id)
            # 抽取最相似的k个
            retrieve_result = get_similar_retrieval_list(query_date=query_date,
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
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='acl18', type=str)
    parser.add_argument('--retrieve_model', default='e5')
    args = parser.parse_args()

    get_similarity(dataset=args.test_dataset, retrieve_model=args.retrieve_model, flag='test')
