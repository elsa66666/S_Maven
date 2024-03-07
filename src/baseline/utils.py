import re
from datetime import datetime
import random
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import json
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def toggle_llama_query(tokenizer, model, query):
    try:
        inputs = tokenizer(query, return_tensors="pt")
        # Generate
        generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=4096)
        answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer
    except Exception as exc:
        print(exc)
        return 'broken'


def cut_pattern(count_index, string, pattern):
    count = 0
    for m in re.finditer(pattern, string):
        count += 1
        if count == count_index:
            return string[m.end():]
    return string


def get_llama_response(index, query, tokenizer, model):
    response_task = toggle_llama_query(tokenizer=tokenizer, model=model, query=query)
    while response_task == "broken":
        response_task = toggle_llama_query(tokenizer=tokenizer, model=model, query=query)
    response_task = cut_pattern(count_index=2,
                                string=response_task,
                                pattern='INST]').lstrip('\t').lstrip(' ')
    return response_task


def check_answer(generated_answer, reference_label):
    if ('rise' or 'Rise' or 'A') in generated_answer:
        generated_label = 'rise'
    elif ('fall' or 'Fall' or 'B') in generated_answer:
        generated_label = 'fall'
    elif generated_answer == 'A':
        generated_label = 'rise'
    elif generated_answer == 'B':
        generated_label = 'fall'
    else:
        generated_label = 'manual_check'

    if reference_label == generated_label:
        flag = 1
    elif generated_label != 'manual_check':
        flag = 0
    else:
        flag = 'manual_check'
    return generated_label, flag


def get_prompt(retrieve_number, query_sequence, example_sequence_list, is_similar=False):
    '''
    :param retrieve_number: retrieve条数
    :param query_sequence: 原始序列
    :param example_sequence_list: candidate原始序列
    :param is_similar: 是否是最相似的k条，默认是随机的k条
    :return:
    '''
    prompt00 = (
        "Given a stock context and a multiple choice question related to it, select the correct answer from the two options.\n            "
        "Question: ")
    prompt_task0 = "This is a JSON format stock price sequence."
    prompt_task1 = "This is a JSON format stock price sequence and a previous sequence of the same stock for reference."
    prompt_task_k = ("This is a JSON format stock price sequence and "
                     + str(retrieve_number)
                     + " previous sequences of the same stock for reference.")
    prompt000 = "\nPlease predict the fluctuation of adjusted close price on the next trading day. \nThe query sequence:\n"
    prompt_parameter = (
        "\nYour Choice:\n\n            Options: A: rise, B: fall.\n            Please answer with A or B only.\n            Answer:\n            ")

    str_query = str(query_sequence)
    example_str = ''
    for sequence in example_sequence_list:
        if is_similar:
            sequence = {"date_list": sequence['candidate_data']['date_list'],
                        "open_list": sequence['candidate_data']['open_list'],
                        "high_list": sequence['candidate_data']['high_list'],
                        "low_list": sequence['candidate_data']['low_list'],
                        "close_list": sequence['candidate_data']['close_list'],
                        "adj_close_list": sequence['candidate_data']['adj_close_list'],
                        "movement": sequence['candidate_data']['movement'],
                        "similarity_to_query_sequence": sequence['score']}
        else:
            sequence = {"date_list": sequence['date_list'],
                        "open_list": sequence['open_list'],
                        "high_list": sequence['high_list'],
                        "low_list": sequence['low_list'],
                        "close_list": sequence['close_list'],
                        "adj_close_list": sequence['adj_close_list'],
                        "movement": sequence['movement']}
        example_str += str(sequence) + '\n'
    # no retrieval
    if retrieve_number == 0:
        prompt_k = prompt00 + prompt_task0 + prompt000 + str_query + prompt_parameter
    else:
        if retrieve_number == 1:  # retrieve one example
            prompt_k = prompt00 + prompt_task1 + prompt000 + str_query + "\nReference sequences: \n" + example_str + prompt_parameter
        elif retrieve_number > 1:
            prompt_k = prompt00 + prompt_task_k + prompt000 + str_query + "\nReference sequences: \n" + example_str + prompt_parameter

    query_llama = ("<s>[INST] <<SYS>> \nYou are a economist. \n<</SYS>>"
                   + prompt_k + '[/INST]')
    return query_llama


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def get_test_data(dataset):
    data = []
    directory = '../../data/processed_data/test/' + dataset + '_test_list.json'
    with open(directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # 在整个数据集中，query序列为起始日期的一年后，以确保每一条query都能检索近一年的序列
    query_start_date = datetime.strptime(data[0]['date_list'][0], format("%Y-%m-%d")) + relativedelta(years=1)
    return data, query_start_date


def get_qualified_retrieval_list(query_date, query_sequence_id, all_data_list):
    start_retrieval_date = query_date - relativedelta(years=1)
    query_company = query_sequence_id.split('_')[1]
    qualified_data_list = []
    for i in range(len(all_data_list)):
        data_date = all_data_list[i]['date_list'][0]
        data_date = datetime.strptime(data_date, format("%Y-%m-%d"))
        data_company = all_data_list[i]['sequence_id'].split('_')[1]
        if (start_retrieval_date <= data_date < query_date) and (data_company == query_company):
            candidate = all_data_list[i]
            candidate_str = str({
                'date_list': candidate['date_list'],
                'open_list': candidate['open_list'],
                'high_list': candidate['high_list'],
                'low_list': candidate['low_list'],
                'close_list': candidate['close_list'],
                'adj_close_list': candidate['adj_close_list'],
                'volume_list': candidate['volume_list']
            })
            candidate_str = (candidate_str + ', \'movement\': '+ candidate['movement'])
            qualified_data_list.append(candidate_str)
    # retrieve_result = sample(qualified_data_list, retrieve_number)
    return qualified_data_list


def get_embeddings(retrieve_model, dataset, flag='test'):
    if flag == 'test':
        # query和candidate的编码不同
        if (retrieve_model == 'e5') or (retrieve_model == 'instructor') or (retrieve_model == 'uae'):
            with open('embeddings/test/' + flag + '_' + dataset + '_' + retrieve_model + '_embeddings_query.pkl',
                      'rb') as f:
                query_embedding_list = pickle.load(f)
            with open('embeddings/test/' + flag + '_' + dataset + '_' + retrieve_model + '_embeddings_candidate.pkl',
                      'rb') as f:
                candidate_embedding_list = pickle.load(f)
        # query和candidate的编码相同
        elif retrieve_model == 'bge' or retrieve_model == 'llm_embedder':
            with open('embeddings/test/' + flag + '_' + dataset + '_' + retrieve_model + '_embeddings.pkl',
                      'rb') as f:
                query_embedding_list = pickle.load(f)
                candidate_embedding_list = pickle.load(f)
    return query_embedding_list, candidate_embedding_list


def get_query_embedding(query_sequence_id, query_embedding_list):
    for i in range(len(query_embedding_list)):
        data1_id = query_embedding_list[i]['data']['sequence_id']
        if data1_id == query_sequence_id:
            query_embedding = query_embedding_list[i]['embedding']
            break
    return query_embedding


def get_similar_retrieval_list(query_date, query_sequence_id, retrieve_model, flag='test'):
    start_retrieval_date = query_date - relativedelta(years=1)
    query_company = query_sequence_id.split('_')[1]  # company name
    dataset = query_sequence_id.split('_')[0]  # dataset name

    query_embedding_list, candidate_embedding_list = get_embeddings(retrieve_model=retrieve_model,
                                                                    dataset=dataset,
                                                                    flag=flag)
    query_embedding = get_query_embedding(query_sequence_id=query_sequence_id,
                                          query_embedding_list=query_embedding_list)

    # {'data': data[i], 'embedding': embeddings_a}

    qualified_data_list = []
    for i in range(len(candidate_embedding_list)):
        candidate_data = candidate_embedding_list[i]['data']
        candidate_embedding = candidate_embedding_list[i]['embedding']
        candidate_date = candidate_data['date_list'][0]
        candidate_date = datetime.strptime(candidate_date, format("%Y-%m-%d"))
        candidate_company = candidate_data['sequence_id'].split('_')[1]
        if (start_retrieval_date <= candidate_date < query_date) and (candidate_company == query_company):
            if retrieve_model == 'instructor':
                score = cosine_similarity(query_embedding, candidate_embedding)
                score = float(score[0][0])
            elif (retrieve_model == 'bge') or (retrieve_model == 'llm_embedder'):
                score = query_embedding @ candidate_embedding.T
                score = float(score[0][0])
            elif retrieve_model == 'e5':
                score = query_embedding @ candidate_embedding.T
                score = float(score[0][0])
            # print('score: ', score)
            qualified_data_list.append({'candidate_data': candidate_data, 'score': score})
    print('降序')
    similarity_list = sorted(qualified_data_list, key=lambda x: x['score'], reverse=True)
    return similarity_list
