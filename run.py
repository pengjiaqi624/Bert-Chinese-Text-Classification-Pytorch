# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import sys
import os
import time

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, ERNIE')
args = parser.parse_args()


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
'''以下自己补充log'''



if __name__ == '__main__':
	
    model_name = args.model  # bert
    x = import_module('models.' +model_name)    # import_module 动态导入模块，绝对导入 models/bert.py
    dataset = 'weibo'  # 数据集,THUCNews
    config = x.Config(dataset)  #models/bert.py/class Config
    np.random.seed(1)   #生成指定随机数,起一次作用
    torch.manual_seed(1)    #设置种子生成随机数
    torch.cuda.manual_seed_all(1)   #为所有的GPU设置种子
    torch.backends.cudnn.deterministic = True   # 保证每次结果一样

    start_time = time.time()    #返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # 自定义目录存放日志文件
    log_path = '../Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)
    
    # train
    model = x.Model(config).to(config.device)   #models/bert.py/class Model
    train(config, model, train_iter, dev_iter, test_iter)

