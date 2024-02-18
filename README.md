# 1 安装的库

```shell
transformers=4.31

pip install InstructorEmbedding
pip install -U FlagEmbedding
pip install sentence-transformers==2.2.2
pip install protobuf==3.20.0
```



# 2 复现步骤

## 2.1 下载数据（无需gpu）

进入src/model/download_price_data.py

- 修改代理地址，在国外就不需要了。在“设置”->“网络和internet”->“代理”中查看。

```
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
```

- download_dataset 函数中，修改agent为自己的agent。具体的方法是google搜索框输入what is my agent，就会返回自己的agent.

  类似这样：Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36

```python
        headers = {
            'User-Agent': ''
        }
```

- 在main函数中，填写股票名称，修改起止日期，和给数据集起的名字

```python
    # 自定义的数据集
    company_list = ['', '']  # 要下载的股票名称填在这里，示例：'APPL', 'AAL' ...
    start_date = int(time.mktime(datetime.datetime(2019, 4, 11,  # 起始日期在这里修改
                                                   0, 0).timetuple()))
    end_date = int(time.mktime(datetime.datetime(2020, 12, 31,  # 截止日期在这里修改
                                                 23, 59).timetuple()))
    download_dataset(dataset_name='',  # 给数据集起的名字填在这里
                     company_list=company_list,
                     start_date=start_date,
                     end_date=end_date)
```

- 运行代码，下载下来的数据存储在 S_Maven/data/raw_data/数据集名字/



## 2.2 数据预处理为sequence （无需gpu）

这一步将所有股票数据预处理为长度为sequence_length的序列，并存储。

进入src/model/data_preprocess.py

- get_movement函数中，定义（或修改）涨跌的阈值
  - 例如这里，涨了0.55%（即0.0055）为涨，跌了0.5%（即-0.005）为跌，这两中间的为停摆

```python
def get_movement(today_adj_close, next_adj_close):
    if (next_adj_close / today_adj_close) - 1 > 0.0055:
        movement = 'rise'
    elif (next_adj_close / today_adj_close) - 1 < -0.005:
        movement = 'fall'
    else:
        movement = 'freeze'
    return movement
```

- 在main函数中，设置：
  - 每条序列的长度
  - 数据集的名字（和上一步给数据集起的名字一样）
  - 是否去掉停摆的序列，去掉是True，不去掉是False
  - 如何划分数据。[train,valid,test]，分别填入作为训练、验证、测试的股票有几支。注意训练、验证、测试的股票支数加起来要等于数据集里面一共的股票支数。

```python
    slice_sequence(sequence_length=5, # 每条序列的长度
                   dataset_name='', # 数据集的名字
                   remove_freeze=True, # True或者False
                   split_dataset=[33, 5, 33])  # 数据集划分[train,valid,test]，如果全作为测试，e.g. [0,0,71]
```

运行代码，下载下来的数据存储在 S_Maven/data/processed_data/数据集名字/



## 2.3 编码序列

这一步用GPU会更快，不用也能运。

为了运行baselines模型，需要进入src/baseline/baseline_models文件夹

新建文件夹BAAI，运行以下代码。这两行代码分别下载bge和llm-embedder模型。

```
git clone https://huggingface.co/BAAI/bge-large-en-v1.5
git clone https://huggingface.co/BAAI/llm-embedder
```

回到src/baseline/baseline_models文件夹，运行以下代码。这一行代码会下载instructor模型。

```
git clone https://huggingface.co/hkunlp/instructor-large
```

回到src/baseline文件夹，进入get_embeddings代码。

如果没有gpu，修改get_bge_or_llm_embeddings函数中, device='cpu'

```python
model = SentenceTransformer('baseline_models/BAAI/llm-embedder', device="cpu")
```

然后在命令行运行以下代码：

```
python get_embeddings.py --test_dataset 你的数据集名称 --retrieve_model llm_embedder
```

这样你的数据集中test子集序列的编码就会保存在S_Maven/src/baseline/embeddings文件夹中



## 2.4 检索最相似的k条（不用gpu）

回到src/baseline文件夹，在命令行运行以下代码：

```
python get_similarity.py --test_dataset 你的数据集名称 --retrieve_model llm_embedder
```

检索结果以.pkl文件的形式存储在S_Maven/src/baseline/similar_candidates/test文件夹中

查看.pkl文件的代码如下，该代码将pkl转化为list。

将断点设在“return pkl_list”这一行，就能查看pkl_list具体内容。

检查：检索出的几条中，涨跌一致吗

```
import pickle

def open_embedding_file():
    with open(('文件'), 'rb') as f:
        pkl_list = pickle.load(f)
    return pkl_list
```

