# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer # pytorch_pretrained_bert 包不好导入，因此提前下载好，放在pytorch_pretrained文件夹中


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        '''

        :param dataset: 数据集名称
        '''
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备'cuda' if torch.cuda.is_available() else；'cpu'

        self.require_improvement = 1000                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 15                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = './bert_pretrain'  #bert模型位置文件
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  #加载bert模型
        self.hidden_size = 768
        self.log_path = "Logs"


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path) #加载bert分词器
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子的token_ids
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0];
        # train_eval.py: for i, (trains, labels) in enumerate(train_iter): ;
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        #output:(hidden_rep, cls_head ),
        # hidden_rep.shape->[batch_size, sequence_length, hidden_size],可以获取给定句子中所有单词的上下文词嵌入表示；
        #cls_head.shape->[batch_size, hidden_size]，cls_head包含整个句子的嵌入表示；
        #获取每个encoder的嵌入表示：
        #model中添加->output_hidden_states = True；则可以返回（last_hidden_state, pooler_output, hidden_states）
        #last_hidden_state：等同于hidden_rep，从最后一个编码器层得到所有标记的嵌入表示。
        #pooler_output：等同于cls_head，
        #hidden_states：是一个13个元素的元组，包含了h0-h12的所有嵌入表示；
        out = self.fc(pooled)
        return out
