import time
import datetime
import pandas as pd
import requests
import os

# yahoo finance example: https://finance.yahoo.com/quote/AAL/history?p=AAL
# tutorial: https://www.youtube.com/watch?v=NjEc7PB0TxQ

# 注意需要代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


def get_company_list_and_data(dataset_name):
    if dataset_name == 'acl18':
        company_list = ['APPL', 'ABB', 'ABBV', 'AEP', 'AGFS', 'AMGN', 'AMZN',
                        'BA', 'BABA', 'BAC', 'BBL', 'BCH', 'BHP', 'BP', 'BRK-A', 'BSAC', 'BUD',
                        'C', 'CAT', 'CELG', 'CHL', 'CHTR', 'CMCSA', 'CODI', 'CSCO', 'CVX',
                        'D', 'DHR', 'DIS', 'DUK',
                        'EXC', 'FB', 'GD', 'GE', 'GMRE', 'GOOG',
                        'HD', 'HON', 'HRG', 'HSBC',
                        'IEP', 'INTC', 'JNJ', 'JPM', 'KO', 'LMT',
                        'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MSFT',
                        'NEE', 'NGG', 'NVS', 'ORCL',
                        'PCG', 'PCLN', 'PEP', 'PFE', 'PG', 'PICO', 'PM', 'PPL', 'PTR',
                        'RDS-B', 'REX', 'SLB', 'SNP', 'SNY', 'SO', 'SPLP', 'SRE', 'T', 'TM', 'TOT', 'TSM',
                        'UL', 'UN', 'UNH', 'UPS', 'UTX', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
                        ]
        period1 = int(time.mktime(datetime.datetime(2014, 6, 2, 0, 0).timetuple()))
        period2 = int(time.mktime(datetime.datetime(2015, 12, 31, 23, 59).timetuple()))

    if dataset_name == 'cikm18':
        company_list = ['AAL', 'AAPL', 'ABBV', 'ABT', 'AGN', 'AMGN', 'AMZN', 'AXP', 'BA', 'BAC', 'BLK', 'BMY', 'C',
                        'CAT', 'CELG', 'CHK', 'CMCSA', 'CMG', 'CSCO', 'CVS', 'CVX', 'DIS', 'F', 'FB', 'FCX', 'GE',
                        'GILD', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
                        'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'UZ', 'WMT', 'XOM']
        period1 = int(time.mktime(datetime.datetime(2017, 1, 3, 0, 0).timetuple()))
        period2 = int(time.mktime(datetime.datetime(2018, 1, 23, 23, 59).timetuple()))

    elif dataset_name == 'bigdata22':
        company_list = ['APPL', 'AEP', 'AGFS', 'AMGN', 'AMZN', 'BA', 'BAC', 'C', 'CAT', 'CACSA', 'CODI', 'CSCO', 'CVX',
                        'D', 'DIS', 'DUK', 'EXC', 'GD', 'GE', 'GMRE', 'GOOG', 'HD', 'HON', 'INTC', 'JNJ', 'JPM', 'KO',
                        'LMT', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'ORCL', 'PCG', 'PM', 'PPL', 'REX',
                        'SO', 'SRE', 'T', 'UPS', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
        period1 = int(time.mktime(datetime.datetime(2019, 4, 1, 0, 0).timetuple()))
        period2 = int(time.mktime(datetime.datetime(2020, 12, 31, 23, 59).timetuple()))

    return company_list, period1, period2


def download_dataset(dataset_name, company_list, start_date, end_date):
    interval = '1d'

    for company in company_list:
        download_address = (f'https://query1.finance.yahoo.com/v7/finance/download/{company}?'
                            f'period1={start_date}'
                            f'&period2={end_date}'
                            f'&interval={interval}'
                            f'&events=history&includeAdjustedClose=true')

        # google一下： what is my user agent,然后把自己的那串字符填在里面
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }

        success = False
        while not success:
            f = requests.get(download_address, headers=headers)  # 把下载地址发送给requests模块
            if f.status_code == requests.codes.ok:
                success = True
                with open(("../../data/raw_data/" + dataset_name + "/" + company + ".csv"), "wb") as code:
                    code.write(f.content)
    print("Finish writing to: ", "../../data/raw_data/", dataset_name)


if __name__ == "__main__":
    # 默认的dataset
    '''
    company_list, period1, period2 = get_company_list_and_data('acl18')
    download_dataset(dataset_name='acl18', 
                     company_list=company_list,
                     start_date=period1,
                     end_date=period2)
    '''

    # 自定义的数据集
    company_list = ['', '']  # 示例：'APPL', 'AAL' ...
    start_date = int(time.mktime(datetime.datetime(2019, 4, 11,
                                                   0, 0).timetuple()))
    end_date = int(time.mktime(datetime.datetime(2020, 12, 31,
                                                 23, 59).timetuple()))
    download_dataset(dataset_name='',
                     company_list=company_list,
                     start_date=start_date,
                     end_date=end_date)
