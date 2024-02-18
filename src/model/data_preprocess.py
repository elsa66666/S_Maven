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


def get_slide_set(dataset_name, company_number):
    if dataset_name == 'acl18':
        if 1 <= company_number <= 33:  # 1-33
            return 'train'
        elif 34 <= company_number <= 38:  # 34-38
            return 'valid'
        else:  # 39-71
            return 'test'
    elif dataset_name == 'bigdata22':
        if 1 <= company_number <= 22:
            return 'train'
        elif 23 <= company_number <= 25:
            return 'valid'
        else:  # 26-47
            return 'test'
    elif dataset_name == 'cikm18':
        if 1 <= company_number <= 19:
            return 'train'
        elif 20 <= company_number <= 22:
            return 'valid'
        else:  # 23-41
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
def slice_sequence(sequence_length, dataset_name):
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
            df_sequence = df1[i:i + 5]
            # print(df_sequence)
            # 序列编号
            sequence_id = dataset_name + '_' + company_name.replace('.csv', '') + '_' + str(i)
            all_count += 1
            # 下一个交易日的close price涨幅
            today_adj_close = df1[i + 4:i + 5]['Adj Close'].values.tolist()[0]
            next_adj_close = df1[i + 5:i + 6]['Adj Close'].values.tolist()[0]
            movement = get_movement(today_adj_close, next_adj_close)

            sequence_dict = get_datastore_df(sequence_id, movement, df_sequence)
            if dataset_name == 'acl18':
                sequence_dict['sequence_index'] = 1000000 + all_count
            elif dataset_name == 'bigdata22':
                sequence_dict['sequence_index'] = 2000000 + all_count
            elif dataset_name == 'cikm18':
                sequence_dict['sequence_index'] = 3000000 + all_count

            if movement != 'freeze':  # 去掉freeze的股票，只做二分类
                if get_slide_set(dataset_name, company_count) == 'train':
                    train_sequence_list1.append(sequence_dict)
                elif get_slide_set(dataset_name, company_count) == 'valid':
                    valid_sequence_list1.append(sequence_dict)
                else:
                    test_sequence_list1.append(sequence_dict)

    with open("../../data/processed_data/train/" + dataset_name + "_train_list.json", "w") as outfile:
        for obj in train_sequence_list1:
            # 将JSON对象转换为字符串
            json_str = json.dumps(obj)
            # 将字符串写入文件，并添加换行符
            outfile.write(json_str + "\n")
        print('Finish slicing the ' + dataset_name + ' train dataset.')
    with open("../../data/processed_data/test/" + dataset_name + "_test_list.json", "w") as outfile:
        for obj in test_sequence_list1:
            # 将JSON对象转换为字符串
            json_str = json.dumps(obj)
            # 将字符串写入文件，并添加换行符
            outfile.write(json_str + "\n")
        print('Finish slicing the ' + dataset_name + ' test dataset.')
    with open("../../data/processed_data/valid/" + dataset_name + "_valid_list.json", "w") as outfile:
        for obj in valid_sequence_list1:
            # 将JSON对象转换为字符串
            json_str = json.dumps(obj)
            # 将字符串写入文件，并添加换行符
            outfile.write(json_str + "\n")
        print('Finish slicing the ' + dataset_name + ' valid dataset.')
    return 0


if __name__ == "__main__":
    slice_sequence(sequence_length=5, dataset_name='acl18')
    slice_sequence(sequence_length=5, dataset_name='bigdata22')
    slice_sequence(sequence_length=5, dataset_name='cikm18')
