import re
from datetime import datetime
import random
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json


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


def get_start_date(dataset1):
    if dataset1 == "acl18":
        start_date1 = datetime.strptime("2015-06-03", format("%Y-%m-%d"))
    elif dataset1 == "bigdata22":
        start_date1 = datetime.strptime("2020-04-09", format("%Y-%m-%d"))
    else:  # cikm18
        start_date1 = datetime.strptime("2018-01-03", format("%Y-%m-%d"))
    return start_date1


def check_answer(generated_answer, reference_label):
    if ('rise' or 'Rise') in generated_answer:
        generated_label = 'rise'
    elif ('fall' or 'Fall') in generated_answer:
        generated_label = 'fall'
    elif ('freeze' or 'Freeze') in generated_answer:
        generated_label = 'freeze'
    else:
        generated_label = 'manual_check'

    if reference_label == generated_label:
        flag = 1
    elif generated_label != 'manual_check':
        flag = 0
    else:
        flag = 'manual_check'
    return generated_label, flag


def get_index_from_id(sequence_id, id_list, index_list):
    index_in_index_list = id_list.index(sequence_id)
    index = index_list[index_in_index_list]
    return index


def get_prompt_for_training(query_id,
                            positive_retrieve, negative_retrieve,
                            query_sequence, answer,
                            id_list, index_list):

    prompt00 = (
        "Given a stock context and a multiple choice question related to it, select the correct answer from the two options.\n            "
        "Question: ")
    prompt_task0 = "This is a JSON format stock price sequence,"

    prompt_parameter = (
        "\nYour Choice:\n\n            Options: A: rise, B: fall.\n            Please answer with A or B only.\n            Answer:\n            ")

    query_prompt = prompt00 + prompt_task0 + str(query_sequence) + prompt_parameter
    positive_prompt_list = []
    negative_prompt_list = []
    teacher_scores = []
    positive_index_list = []
    negative_index_list = []
    for data_list in [positive_retrieve, negative_retrieve]:
        for sequence in data_list:
            candidate_data = sequence['candidate_data']
            candidate_score = sequence['score']
            teacher_scores.append(float(candidate_score))
            prompt_sequence = {"date_list": candidate_data['date_list'],
                               "open_list": candidate_data['open_list'],
                               "high_list": candidate_data['high_list'],
                               "low_list": candidate_data['low_list'],
                               "close_list": candidate_data['close_list'],
                               "adj_close_list": candidate_data['adj_close_list'],
                               "volume_list": candidate_data['volume_list']}
            movement = candidate_data['movement']
            if movement == 'rise':
                answer = 'A: rise'
            elif movement == 'fall':
                answer = 'B: fall'

            candidate_prompt = prompt00 + prompt_task0 + str(prompt_sequence) + prompt_parameter + answer
            candidate_index = get_index_from_id(sequence_id=candidate_data['sequence_id'], id_list=id_list, index_list=index_list)
            if sequence in positive_retrieve:
                positive_prompt_list.append(candidate_prompt)
                positive_index_list.append(candidate_index)
            else:
                negative_prompt_list.append(candidate_prompt)
                negative_index_list.append(candidate_index)

    query_index = get_index_from_id(sequence_id=query_id, id_list=id_list, index_list=index_list)
    query_with_candidates = {
        "query_id": query_index,
        "query": query_prompt,
        "pos": positive_prompt_list,
        "neg": negative_prompt_list,
        # "teacher_scores": teacher_scores,
        "pos_index": positive_index_list,
        "neg_index": negative_index_list,
        "answers": [answer],
        "task": "icl"
    }
    return query_with_candidates


def get_qualified_retrieval_list(query_date, query_sequence_id, all_data_list):
    start_retrieval_date = query_date - relativedelta(years=1)
    query_company = query_sequence_id.split('_')[1]
    qualified_data_list = []
    for i in range(len(all_data_list)):
        data_date = all_data_list[i]['date_list'][0]
        data_date = datetime.strptime(data_date, format("%Y-%m-%d"))
        data_company = all_data_list[i]['sequence_id'].split('_')[1]
        if start_retrieval_date <= data_date < query_date and data_company == query_company:
            qualified_data_list.append(all_data_list[i])
    # retrieve_result = sample(qualified_data_list, retrieve_number)
    return qualified_data_list


def get_similar_retrieval_list(query_date, query_sequence_id, retrieve_model):
    start_retrieval_date = query_date - relativedelta(years=1)
    query_company = query_sequence_id.split('_')[1]  # company name
    dataset = query_sequence_id.split('_')[0]  # dataset name

    with open(('embeddings/' + dataset + '_' + retrieve_model + '_embeddings.pkl'), 'rb') as f:
        all_embedding_list = pickle.load(f)

    # {'data': data[i], 'embedding': embeddings_a}

    for i in range(len(all_embedding_list)):
        data1_id = all_embedding_list[i]['data']['sequence_id']
        if data1_id == query_sequence_id:
            query_embedding = all_embedding_list[i]['embedding']
            break

    qualified_data_list = []
    for i in range(len(all_embedding_list)):
        data1 = all_embedding_list[i]['data']
        data1_embedding = all_embedding_list[i]['embedding']
        data_date = data1['date_list'][0]
        data_date = datetime.strptime(data_date, format("%Y-%m-%d"))
        data_company = data1['sequence_id'].split('_')[1]
        if (start_retrieval_date <= data_date < query_date) and (data_company == query_company):
            if retrieve_model == 'instructor':
                score = cosine_similarity(query_embedding, data1_embedding)
            elif (retrieve_model == 'bge') or (retrieve_model == 'llm_embedder'):
                score = query_embedding @ data1_embedding.T
            score = score[0][0]
            # print('score: ', score)
            qualified_data_list.append({'candidate_data': data1, 'score': score})
    print('降序')
    similarity_list = sorted(qualified_data_list, key=lambda x: x['score'], reverse=True)
    return similarity_list


def get_prompt_for_finetune(retrieve_number, query_sequence, example_sequence_list, is_similar=False):
    '''
    :param retrieve_number: retrieve条数
    :param query_sequence: 原始序列
    :param example_sequence_list: candidate原始序列
    :param is_similar: 是否是最相似的k条，默认是随机的k条
    :return:
    '''
    options = ['rise', 'fall']
    random.shuffle(options)
    options_str = str(options)
    prompt_task0 = "This is a JSON format stock price sequence,"
    prompt_task1 = "This is a JSON format stock price sequence and a previous sequence of the same stock for reference,"
    prompt_task_k = ("This is a JSON format stock price sequence and "
                     + str(retrieve_number)
                     + " previous sequences of the same stock for reference.")
    prompt_parameter = ("\nPlease predict the fluctuation of adjusted close price on the next trading day. "
                        "\nThe query sequence:\n")

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
                        "volume_list": sequence['candidate_data']['volume_list'],
                        "movement": sequence['candidate_data']['movement'],
                        "similarity_to_query_sequence": sequence['score']}
        else:
            sequence = {"date_list": sequence['date_list'],
                        "open_list": sequence['open_list'],
                        "high_list": sequence['high_list'],
                        "low_list": sequence['low_list'],
                        "close_list": sequence['close_list'],
                        "adj_close_list": sequence['adj_close_list'],
                        "volume_list": sequence['volume_list'],
                        "movement": sequence['movement']}
        example_str += str(sequence) + '\n'
    # no retrieval
    if retrieve_number == 0:
        prompt0 = prompt_task0 + prompt_parameter
        query = prompt0 + str_query + "Your Choice:\n"
    # retrieve one example
    else:
        if retrieve_number == 1:
            prompt_k = prompt_task1 + prompt_parameter
        elif retrieve_number > 1:
            prompt_k = prompt_task_k + prompt_parameter
        query = prompt_k + str_query + "\nReference sequences: \n" + example_str + "Your Choice:\n"

    return query
