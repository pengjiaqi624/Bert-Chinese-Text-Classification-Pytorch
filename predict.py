# -*-coding = uft-8 -*-
# @Time : 2022/8/31 20:39
# @Author : PENG
# @File : predict.py
# @Software : PyCharm
import os

import pandas as pd
import pymysql
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from sqlalchemy import create_engine
from tqdm import tqdm

from pytorch_pretrained import BertModel, BertTokenizer

# 识别的类型
key = {0: '0',
       1: '1'
       }

class Config:
    """配置参数"""

    def __init__(self):
        cru = os.path.dirname(__file__)
        self.class_list = [str(i) for i in range(len(key))]  # 类别名单
        self.save_path = os.path.join(cru, 'weibo/saved_dict/bert.ckpt')
        self.device = torch.device('cpu')   # 设备'cuda' if torch.cuda.is_available() else；'cpu'
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.bert_path = './bert_pretrain' #os.path.join(cru, 'bert')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

    def build_dataset(self, text):
        lin = text.strip()
        pad_size = self.pad_size#len(lin)
        token = self.tokenizer.tokenize(lin)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = []
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
        return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])
        # token = self.tokenizer.tokenize(lin)
        # token = ['[CLS]'] + token
        # token_ids = self.tokenizer.convert_tokens_to_ids(token)
        # mask = [1] * pad_size
        # token_ids = token_ids[:pad_size]
        # return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


config = Config()
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))#


def prediction_model(df):
    labels = []
    print(df.shape[0])
    for i in tqdm(range(df.shape[0])):
        data = config.build_dataset(str(df.iloc[i,5]))
        with torch.no_grad():
            outputs = model(data)
            num = torch.argmax(outputs)
            labels.append(key[int(num)])   #append会修改L本身，并且返回None。不能把返回值再赋值给L;但对于pd 要返回！
    # df['lable'] = labels
    #列表结果插入text之前
    df.insert(5, 'lable', labels)



if __name__ == '__main__':
    conn = pymysql.connect(host="127.0.0.1",
                           port=3306,
                           user="root",
                           password="pengjiaqi",
                           db="zzweibo2",
                           charset="utf8")


    def load_data_from_mysql(table):
        '''
        :param table: 查询表名
        :return: 表数据
        '''
        sql = "SELECT * FROM " + str(table)
        # cursor = conn.cursor()
        # cursor.execute(sql)
        # result = cursor.fetchall()
        data_frame = pd.read_sql(sql, conn)
        return data_frame
    engine = create_engine(
        "mysql+pymysql://root:pengjiaqi@127.0.0.1:3306/zzweibo2")  # https://zhuanlan.zhihu.com/p/364688931

    # df = load_data_from_mysql('weibo_dropduplicatetext_t2s_n')
    # # df = df.iloc[200:250,:] #为了测试
    # prediction_model(df)
    # df_l = load_data_from_mysql('weibo_dropduplicatetext_t2s_l')
    # df_l = df_l.append(df)
    df = load_data_from_mysql('weibo_dropduplicate2text2_t2sclean2')
    prediction_model(df)
    df.to_sql('weibo_dropduplicate2text2_t2s_l', engine, if_exists="append",
                   index=False)  # pd 的 to_sql 不能使用 pymysql 的连接，否则就会直接报错

