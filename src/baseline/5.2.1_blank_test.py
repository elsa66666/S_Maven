# 5.2.1 空白对照prompt
import json
import os
from datetime import datetime
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
import argparse

import utils
from utils import get_llama_response, get_prompt, get_start_date, check_answer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
setattr(utils, 'get_llama_response', get_llama_response)
setattr(utils, 'get_prompt_for_test', get_prompt)
setattr(utils, 'get_start_date', get_start_date)
setattr(utils, 'check_answer', check_answer)


def get_retrieve_blank(test_dataset1, model_path1):
    data = []
    test_directory = '../../data/processed_data/test/' + test_dataset1 + '_test_list.json'
    with open(test_directory, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    tokenizer = LlamaTokenizer.from_pretrained(model_path1)
    model = LlamaForCausalLM.from_pretrained(model_path1, device_map='auto')

    start_index = -1
    for i in range(len(data)):

        date1 = datetime.strptime(data[i]['date_list'][0], format("%Y-%m-%d"))
        if date1 >= get_start_date(test_dataset1):  # 注意测试的时间范围
            if start_index == -1:
                start_index = i
            # 序列的index
            query_sequence_id = data[i]['sequence_id']
            print('index: ', query_sequence_id)
            # 输入llm的序列
            query_sequence = {"date_list": data[i]['date_list'],
                              "open_list": data[i]['open_list'],
                              "high_list": data[i]['high_list'],
                              "low_list": data[i]['low_list'],
                              "close_list": data[i]['close_list'],
                              "adj_close_list": data[i]['adj_close_list']}
            query = get_prompt(retrieve_number=0, query_sequence=query_sequence, example_sequence_list=[])
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
            retrieve_output_directory = '../../retrieve_result/5.2.1_blank_retrieve/[llm_output]' + test_dataset1 + '_blank_retrieve.csv'
            if i == start_index:
                df1.to_csv(retrieve_output_directory, mode='a', index=False, header=True)
            else:
                df1.to_csv(retrieve_output_directory, mode='a', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dataset', default='cikm18', choices=['acl18', 'bigdata22', 'cikm18'], type=str)
    parser.add_argument('--llm_path', default='/data/xiaomengxi/stock_rag/src/model/LLMs/merged_1310')
    args = parser.parse_args()

    get_retrieve_blank(test_dataset1=args.test_dataset, model_path1=args.llm_path)
