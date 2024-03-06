from InstructorEmbedding import INSTRUCTOR
import pickle
from sentence_transformers import SentenceTransformer
from angle_emb import AnglE, Prompts
import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import utils
from utils import get_llama_response, get_prompt, check_answer, get_qualified_retrieval_list, \
    get_similar_retrieval_list, last_token_pool, get_detailed_instruct, get_test_data

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
setattr(utils, 'get_llama_response', get_llama_response)
setattr(utils, 'get_prompt', get_prompt)
setattr(utils, 'check_answer', check_answer)
setattr(utils, 'get_qualified_retrieval_list', get_qualified_retrieval_list)
setattr(utils, 'get_similar_retrieval_list', get_similar_retrieval_list)
setattr(utils, 'last_token_pool', last_token_pool)
setattr(utils, 'get_detailed_instruct', get_detailed_instruct)
setattr(utils, 'get_test_data', get_test_data)


def get_embedding_sequence_str(sequence):
    seq1 = {
        'date_list': sequence['date_list'],
        'open_list': sequence['open_list'],
        'high_list': sequence['high_list'],
        'low_list': sequence['low_list'],
        'close_list': sequence['close_list'],
        'adj_close_list': sequence['adj_close_list'],
        'volume_list': sequence['volume_list']
    }
    return str(seq1)


# https://github.com/xlang-ai/instructor-embedding
def get_instructor_embeddings(test_dataset1):
    data, query_start_date = get_test_data(test_dataset1)
    if os.path.exists('baseline_models/instructor-large'):
        print('Loading local model ...')
        instructor = INSTRUCTOR('baseline_models/instructor-large')
    else:
        print('No local models, downloading ...')
        instructor = INSTRUCTOR('hkunlp/instructor-large')

    query_embedding_list = []
    for i in range(len(data)):
        query_date = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 注意query的时间范围
        if query_date >= query_start_date:
            # 序列的id
            query_sequence_id = data[i]['sequence_id'] + '_query'
            print('index: ', query_sequence_id)
            seq_str = get_embedding_sequence_str(data[i])
            sentences_a = [['Represent a stock sequence for retrieving similar sequences: ', seq_str]]
            embeddings_a = instructor.encode(sentences_a)
            query_embedding_list.append({'data': data[i], 'embedding': embeddings_a})
            print('finish embedding ', i, 'in ', len(data))

    with open(('embeddings/test/test_' + test_dataset1 + '_instructor_embeddings_query.pkl'), 'wb') as f:
        pickle.dump(query_embedding_list, f)

    candidate_embedding_list = []
    for i in range(len(data)):
        query_sequence_id = data[i]['sequence_id'] + '_query'
        print('index: ', query_sequence_id)
        seq_str = get_embedding_sequence_str(data[i])
        sentences_a = [['Represent a stock sequence for retrieval: ', seq_str]]
        embeddings_a = instructor.encode(sentences_a)
        candidate_embedding_list.append({'data': data[i], 'embedding': embeddings_a})
        print('finish embedding ', i, 'in ', len(data))

    with open(('embeddings/test/test_' + test_dataset1 + '_instructor_embeddings_candidate.pkl'), 'wb') as f:
        pickle.dump(candidate_embedding_list, f)
    return 0


# https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding
def get_bge_embeddings(test_dataset):
    if os.path.exists('baseline_models/BAAI/bge-large-en-v1.5'):
        print('Loading local model ...')
        model = SentenceTransformer('baseline_models/BAAI/bge-large-en-v1.5')
    else:
        print('No local models, downloading ...')
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    # open sequence data
    data, query_start_date = get_test_data(dataset=test_dataset)

    embedding_list = []
    for i in range(len(data)):
        sentences_a = get_embedding_sequence_str(data[i])
        embeddings_a = model.encode(sentences_a, normalize_embeddings=True)
        embedding_list.append({'data': data[i], 'embedding': embeddings_a})
        print('finish embedding ', i, 'in ', len(data))
    with open(('embeddings/test/test_' + test_dataset + '_bge_embeddings.pkl'),
              'wb') as f:
        pickle.dump(embedding_list, f)
    return 0


# https://huggingface.co/BAAI/llm-embedder
def get_llm_embedder_embeddings(test_dataset):
    data, query_start_date = get_test_data(dataset=test_dataset)
    if os.path.exists('baseline_models/BAAI/llm-embedder'):
        print('Loading local model ...')
        model = SentenceTransformer('baseline_models/BAAI/llm-embedder', device=args.device)
    else:
        print('No local models, downloading ...')
        model = SentenceTransformer('BAAI/llm-embedder')

    embedding_list = []
    for i in range(len(data)):
        sentences_a = get_embedding_sequence_str(data[i])
        embeddings_a = model.encode(sentences_a, normalize_embeddings=True)
        embedding_list.append({'data': data[i], 'embedding': embeddings_a})
        print('finish embedding ', i, 'in ', len(data))
    with open(('embeddings/test/test_' + test_dataset + '_llm_embedder_embeddings.pkl'),
              'wb') as f:
        pickle.dump(embedding_list, f)
    return 0


# https://huggingface.co/intfloat/e5-mistral-7b-instruct/tree/main
def get_e5_embeddings(test_dataset):
    data, query_start_date = get_test_data(dataset=test_dataset)
    if os.path.exists('baseline_models/e5-mistral-7b-instruct'):
        print('Loading local model ...')
        tokenizer = AutoTokenizer.from_pretrained('baseline_models/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('baseline_models/e5-mistral-7b-instruct')
    else:
        print('No local models, downloading ...')
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
    max_length = 4096

    query_embedding_list = []
    for i in range(len(data)):
        query_date = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 注意query的时间范围
        if query_date >= query_start_date:
            # 序列的id
            query_sequence_id = data[i]['sequence_id'] + '_query'
            print('index: ', query_sequence_id)
            # answer = data[i]['movement']
            task = 'Retrieve similar stock sequences from a given query to aid in predicting next day\'s adjusted close price movement.'

            query_sequence = get_embedding_sequence_str(data[i])
            queries = [get_detailed_instruct(task, (query_sequence + ', \'movement\': ?'))]
            # Tokenize the input texts
            batch_dict = tokenizer(queries, max_length=max_length - 1, return_attention_mask=False, padding=False,
                                   truncation=True)
            # append eos_token_id to every input_ids
            batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            query_embedding_list.append({'data': data[i], 'embedding': embeddings})
            print('finish embedding ', i, 'in ', len(data))

    with open(('embeddings/test/test_' + test_dataset + '_e5_embeddings_query.pkl'), 'wb') as f:
        pickle.dump(query_embedding_list, f)

    candidate_embedding_list = []
    for i in range(len(data)):
        query_sequence_id = data[i]['sequence_id'] + '_candidate'
        print('index: ', query_sequence_id)
        query_sequence = get_embedding_sequence_str(data[i])
        batch_dict = tokenizer([query_sequence], max_length=max_length - 1, return_attention_mask=False, padding=False,
                               truncation=True)
        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        candidate_embedding_list.append({'data': data[i], 'embedding': embeddings})
    with open(('embeddings/test/test_' + test_dataset + '_e5_embeddings_candidate.pkl'), 'wb') as f:
        pickle.dump(candidate_embedding_list, f)
    return 0


# https://huggingface.co/WhereIsAI/UAE-Large-V1
def get_uae_embeddings(test_dataset):
    data, query_start_date = get_test_data(dataset=test_dataset)
    if os.path.exists('baseline_models/UAE-Large-V1'):
        print('Loading local model ...')
        angle = AnglE.from_pretrained('baseline_models/UAE-Large-V1', pooling_strategy='cls').cuda()
    else:
        print('No local models, downloading ...')
        angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

    # get candidate embedding
    candidate_embedding_list = []
    for i in range(len(data)):
        query_sequence_id = data[i]['sequence_id'] + '_candidate'
        print('index: ', query_sequence_id)
        query_sequence = get_embedding_sequence_str(data[i])
        vec = angle.encode(query_sequence, to_numpy=True)
        candidate_embedding_list.append({'data': data[i], 'embedding': vec})
        print('finish embedding ', i, 'in ', len(data))
    with open(('embeddings/test/test_' + test_dataset + '_uae_embeddings_candidate.pkl'), 'wb') as f:
        pickle.dump(candidate_embedding_list, f)

    # query prompt
    angle.set_prompt(prompt=Prompts.C)
    query_embedding_list = []
    for i in range(len(data)):
        query_date = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        # 注意query的时间范围
        if query_date >= query_start_date:
            # 序列的id
            query_sequence_id = data[i]['sequence_id'] + '_query'
            print('index: ', query_sequence_id)
            query_sequence = get_embedding_sequence_str(data[i])
            vec = angle.encode({'text': query_sequence}, to_numpy=True)
            query_embedding_list.append({'data': data[i], 'embedding': vec})
            print('finish embedding ', i, 'in ', len(data))
    with open(('embeddings/test/test_' + test_dataset + '_e5_embeddings_query.pkl'), 'wb') as f:
        pickle.dump(query_embedding_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='acl18', type=str)
    parser.add_argument('--retrieve_model', default='e5')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    test_directory = '../../data/processed_data/test/' + args.test_dataset + '_test_list.json'
    if args.retrieve_model == 'bge':
        get_bge_embeddings(test_dataset=args.test_dataset)
    elif args.retrieve_model == 'llm_embedder':
        get_llm_embedder_embeddings(test_dataset=args.test_dataset)
    elif args.retrieve_model == 'instructor':
        get_instructor_embeddings(test_dataset1=args.test_dataset)
    elif args.retrieve_model == 'e5':
        get_e5_embeddings(test_dataset=args.test_dataset)
