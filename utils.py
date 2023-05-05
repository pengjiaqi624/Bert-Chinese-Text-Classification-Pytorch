# coding: UTF-8
import pandas as pd
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):



    def load_dataset(path):   #这里改了 32 , pad_size=128
        '''
        加载数据集
        :param path:   数据集路径
        :param pad_size:
        :return:
        '''
        pad_size = config.pad_size
        contents = []
        df = pd.read_csv(path,sep='\t',header=None)
        for i in tqdm(range(df.shape[0])):
            content, label = str(df.iloc[i, 0]).strip(), df.iloc[i, 1]
            token = config.tokenizer.tokenize(content)  #分词，x=动态引入的bert.py, config = x.Config(self.bert_path )， Config self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
            #BertTokenizer.from_pretrained('THUCNews').tokenize(content)
            token = [CLS] + token   #单分类任务需要句首加[CLS]
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)   #tokenize后的结果构造一个字典vocab.txt（引入），因为只有vocab.txt中每个字的索引顺序才与开源模型中每个字的Embedding向量一一对应；猜测是与vocab.txt中进行角标对应？？？
        # with open(path, 'r', encoding='UTF-8') as f:
        #     for line in tqdm(f):    #tqdm--进度条库
        #         lin = line.strip()  #strip()移除字符串头尾指定的字符（默认为空格或换行符）
        #         if not lin:
        #             continue    #判断是否为NONE
        #         content, label = lin.rsplit(r'\t',1)    #分割制表符 lin.rsplit('\t',1)
        #         token = config.tokenizer.tokenize(content)  #分词，x=动态引入的bert.py, config = x.Config(self.bert_path )， Config self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        #         #BertTokenizer.from_pretrained('THUCNews').tokenize(content)
        #         token = [CLS] + token   #单分类任务需要句首加[CLS]
        #         seq_len = len(token)
        #         mask = []
        #         token_ids = config.tokenizer.convert_tokens_to_ids(token)   #猜测是与vocab.txt中进行角标对应？？？

                # if pad_size:
                #     if len(token) < pad_size:
                #         mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                #         token_ids += ([0] * (pad_size - len(token)))    #token_ids 在原来基础上，补上[0]
                #     else:
                #         mask = [1] * pad_size
                #         token_ids = token_ids[:pad_size]    #token_ids 只取到pad_size
                #         seq_len = pad_size  #重置seq_len
                # contents.append((token_ids, int(label), seq_len, mask)) #vocab.txt中的角标，label，句子长度，mask
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))    #token_ids 在原来基础上，补上[0];token补[PAD]？
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]    #token_ids 只取到pad_size
                    seq_len = pad_size  #重置seq_len
            contents.append((token_ids, int(label), seq_len, mask)) #vocab.txt中的角标，label，句子长度，mask
        return contents
    train = load_dataset(config.train_path) #生成[vocab.txt中的角标，label，句子长度，mask]
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterater(object):
    #把数据的dataset变为可迭代形式的，或者说batch形式的
    def __init__(self, batches, batch_size, device):
        '''

        :param batches: dataset
        :param batch_size:batch_size
        :param device:gpu/cpu
        '''
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size #// 返回商的整数部分（向下取整）
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  #%	取模-返回除法的余数
            self.residue = True
        self.index = 0  #应该是第几个batch
        self.device = device

    def _to_tensor(self, datas):    #根据build_dataset中返回的dataset：((token_ids, int(label), seq_len, mask))
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) #token_ids
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device) #int(label)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        # return 结果是什么考虑
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:   #如果不是正好能分成n个batch；且index已经达到了1个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)] #从上一个整分的batch结束 到 数据的结尾，作为一个batches
            self.index += 1 #index加1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:  #如果index超过或者等于batch数
            self.index = 0  #重置index，为下一个Epoch进行准备
            raise StopIteration #停止迭代
        else:#正好可以分为n个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        '''

        :return: 迭代？？？
        '''
        return self

    def __len__(self):
        '''

        :return: batch的个数
        '''
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()  #格式类似：1661414602.2412865
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
    #round( x [, n]  ) ->四舍五入，n位小数；timedelta -> 转换为 0:00:00
