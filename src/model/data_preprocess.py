# deprecated
# c_xxx: 当日涨跌
# c_open: open / close -1
# c_high: high / open - 1
# c_low: low / close -1

# n_xxx: 和前一个交易日的涨跌
# n_close：当日close / 前一个交易日的close - 1
# n_adj_close

# raw data
# Date,Open,High,Low,Close,Adj Close,Volume

import os.path
import pandas as pd
import json


def get_sub_directory_names(path):
    # 判断路径是否存在
    if os.path.exists(path):
        # 获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
    return files


def get_slide_set(company_number, split_dataset):
    if split_dataset[0] != 0:
        if 1 <= company_number <= split_dataset[0]:
            return 'train'
        elif split_dataset[0] + 1 <= company_number <= split_dataset[0] + split_dataset[1]:
            return 'valid'
        else:
            return 'test'
    elif (split_dataset[0] == 0) and (split_dataset[1] != 0):
        if 1 <= company_number <= split_dataset[1]:
            return 'valid'
        else:
            return 'test'
    elif (split_dataset[0] == 0) and (split_dataset[1] == 0):
        return 'test'


def get_datastore_df(sequence_id, movement, df_sequence):
    # Date,Open,High,Low,Close,Adj Close,Volume
    sequence_dict = {
        'sequence_id': sequence_id,
        'movement': movement,
        'date_list': df_sequence['Date'].values.tolist(),
        'open_list': df_sequence['Open'].values.tolist(),
        'high_list': df_sequence['High'].values.tolist(),
        'low_list': df_sequence['Low'].values.tolist(),
        'close_list': df_sequence['Close'].values.tolist(),
        'adj_close_list': df_sequence['Adj Close'].values.tolist(),
        'volume_list': df_sequence['Volume'].values.tolist()
    }
    return sequence_dict


def get_movement(today_adj_close, next_adj_close):
    if (next_adj_close / today_adj_close) - 1 > 0.0055:
        movement = 'rise'
    elif (next_adj_close / today_adj_close) - 1 < -0.005:
        movement = 'fall'
    else:
        movement = 'freeze'
    return movement


# 交易信息切分为序列
def slice_sequence(sequence_length, dataset_name, split_dataset, remove_freeze=True):
    data_directory = '../../data/raw_data/'

    train_sequence_list1 = []
    test_sequence_list1 = []
    valid_sequence_list1 = []
    dataset_directory = data_directory + dataset_name
    company_name_list = get_sub_directory_names(dataset_directory)
    company_count = 0
    all_count = 0
    for company_name in company_name_list:
        company_count += 1
        company_directory = dataset_directory + '/' + company_name
        df1 = pd.read_csv(company_directory)
        for i in range(len(df1) - sequence_length):
            df_sequence = df1[i:i + sequence_length]
            # print(df_sequence)
            # 序列编号
            sequence_id = dataset_name + '_' + company_name.replace('.csv', '') + '_' + str(i)
            all_count += 1
            # 下一个交易日的close price涨幅
            today_adj_close = df1[(i + sequence_length - 1):(i + sequence_length)]['Adj Close'].values.tolist()[0]
            next_adj_close = df1[(i + sequence_length):(i + sequence_length + 1)]['Adj Close'].values.tolist()[0]
            movement = get_movement(today_adj_close, next_adj_close)

            sequence_dict = get_datastore_df(sequence_id, movement, df_sequence)
            if dataset_name == 'acl18':
                sequence_dict['sequence_index'] = 1000000 + all_count
            elif dataset_name == 'bigdata22':
                sequence_dict['sequence_index'] = 2000000 + all_count
            elif dataset_name == 'cikm18':
                sequence_dict['sequence_index'] = 3000000 + all_count
            else:
                sequence_dict['sequence_index'] = 4000000 + all_count

            split_result = get_slide_set(company_number=company_count, split_dataset=split_dataset)
            if remove_freeze:
                if movement != 'freeze':  # 去掉freeze的股票，只做二分类
                    if split_result == 'train':
                        train_sequence_list1.append(sequence_dict)
                    elif split_result == 'valid':
                        valid_sequence_list1.append(sequence_dict)
                    else:
                        test_sequence_list1.append(sequence_dict)
            else:
                if split_result == 'train':
                    train_sequence_list1.append(sequence_dict)
                elif split_result == 'valid':
                    valid_sequence_list1.append(sequence_dict)
                else:
                    test_sequence_list1.append(sequence_dict)

    if train_sequence_list1:
        with open("../../data/processed_data/train/" + dataset_name + "_train_list.json", "w") as outfile:
            for obj in train_sequence_list1:
                json_str = json.dumps(obj)  # 将JSON对象转换为字符串
                outfile.write(json_str + "\n")  # 将字符串写入文件，并添加换行符
            print('Finish slicing the ' + dataset_name + ' train dataset.')

    if test_sequence_list1:
        with open("../../data/processed_data/test/" + dataset_name + "_test_list.json", "w") as outfile:
            for obj in test_sequence_list1:
                json_str = json.dumps(obj)
                outfile.write(json_str + "\n")
            print('Finish slicing the ' + dataset_name + ' test dataset.')

    if valid_sequence_list1:
        with open("../../data/processed_data/valid/" + dataset_name + "_valid_list.json", "w") as outfile:
            for obj in valid_sequence_list1:
                json_str = json.dumps(obj)
                outfile.write(json_str + "\n")
            print('Finish slicing the ' + dataset_name + ' valid dataset.')
    return 0


if __name__ == "__main__":
    # 默认的dataset
    # slice_sequence(sequence_length=5, dataset_name='acl18', split_dataset=[33, 5, 33])
    # slice_sequence(sequence_length=5, dataset_name='bigdata22', split_dataset=[22, 3, 22])
    # slice_sequence(sequence_length=5, dataset_name='cikm18', split_dataset=[19, 3, 19])
    slice_sequence(sequence_length=5, dataset_name='stock23', split_dataset=[24, 3, 24])

    '''
    slice_sequence(sequence_length=5, # 每条序列的长度
                   dataset_name='', # 数据集的名字
                   remove_freeze=True,  # 是否将停摆的股票序列去掉
                   split_dataset=[33, 5, 33])  # 数据集划分[train,valid,test]，如果全作为测试，e.g. [0,0,71]
    '''
