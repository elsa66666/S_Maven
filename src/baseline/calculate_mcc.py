import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

df1 = pd.read_csv('../../retrieve_result/5.2.3_similar_retrieve/bge[llm_output]bigdata22_similar_retrieve_5.csv')
#define array of actual classes
actual = df1['reference_label'].values

#define array of predicted classes
pred = df1['generated_label'].values

#calculate Matthews correlation coefficient
mcc = matthews_corrcoef(actual, pred)
print(format(mcc, '.3f'))
