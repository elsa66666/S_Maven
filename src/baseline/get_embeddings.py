import json
from InstructorEmbedding import INSTRUCTOR
import pickle
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
import argparse


def get_instructor_embeddings(test_dataset1):
    test_directory = '../../data/processed_data/test/' + test_dataset1 + '_test_list.json'
    instructor_embedding_list = []
    instructor = INSTRUCTOR('baseline_models/instructor-large')
    data = []
    with open(test_directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

        for i in range(len(data)):
            # for i in range(5):
            seq1 = data[i]['adj_close_list']
            sentences_a = [['Show a stock movement sequence: ', str(seq1)]]
            embeddings_a = instructor.encode(sentences_a)
            instructor_embedding_list.append({'data': data[i], 'embedding': embeddings_a})
            print('finish embedding ', i, 'in ', len(data))
    with open(('embeddings/test/test_' + test_dataset1 + '_instructor_embeddings.pkl'), 'wb') as f:
        pickle.dump(instructor_embedding_list, f)
    return 0


def get_bge_or_llm_embeddings(test_dataset1, directory, retrieve_model, flag):
    embedding_list = []
    if retrieve_model == 'bge':
        model = FlagModel('baseline_models/BAAI/bge-large-en-v1.5',
                          query_instruction_for_retrieval="find a similar sequence for stock price movement",
                          use_fp16=True)
    elif retrieve_model == 'llm_embedder':
        model = SentenceTransformer('baseline_models/BAAI/llm-embedder', device="cuda")
    elif retrieve_model == 'stock_maven':
        model = SentenceTransformer('baseline_models/StockMaven', device="cuda")

    # open sequence data
    data = []
    with open(directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

        for i in range(len(data)):
            seq1 = data[i]
            sentences_a = [str(seq1)]
            embeddings_a = model.encode(sentences_a)
            embedding_list.append({'data': data[i], 'embedding': embeddings_a})
            print('finish embedding ', i, 'in ', len(data))
    with open(('embeddings/'+flag+'/' + flag + '_' + test_dataset1 + '_' + retrieve_model + '_embeddings.pkl'), 'wb') as f:
        pickle.dump(embedding_list, f)
    return 0


def open_embedding_file(test_dataset1, retrieve_model):
    with open(('embeddings/' + test_dataset1 + '_' + retrieve_model + '_embeddings.pkl'), 'rb') as f:
        embedding_list = pickle.load(f)
    return embedding_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='cikm18', type=str)
    parser.add_argument('--retrieve_model', default='llm_embedder')
    args = parser.parse_args()

    '''
    for dataset in ['acl18', 'cikm18', 'bigdata22']:
        test_directory = '../../data/processed_data/test/' + dataset + '_test_list.json'
        if args.retrieve_model != 'instructor':
            get_bge_or_llm_embeddings(dataset, directory=test_directory,
                                      retrieve_model=args.retrieve_model, flag='test')
        else:
            get_instructor_embeddings(dataset)
    '''

    test_directory = '../../data/processed_data/test/' + args.test_dataset + '_test_list.json'
    if args.retrieve_model != 'instructor':
        get_bge_or_llm_embeddings(test_dataset1=args.test_dataset,
                                  directory=test_directory,
                                  retrieve_model=args.retrieve_model,
                                  flag='test')
    else:
        get_instructor_embeddings(test_dataset1=args.test_dataset)
