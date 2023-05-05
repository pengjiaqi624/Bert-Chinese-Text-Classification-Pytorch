# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from torch.utils.tensorboard import SummaryWriter

from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam    #BERT的权值衰减固定、预热和学习速率线性衰减的Adam Optimize


# 权重初始化，默认xavier；哪里被用到？？？
def init_network(model, method='xavier', exclude='embedding', seed=123):
    '''

    :param model:
    :param method:
    :param exclude:
    :param seed: 为了结果的“可复现”?在同一台机器上如果使用完全相同的seed，可能会在各类参数随机初始化的过程中成为一定的“随机定值”，使得结果可复现。
    :return:
    '''
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, alpha=0.3):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()   #把model切换到train的状态
    #设置模型待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,   #warmup->让学习率先增大，后减小?0.1？
                         t_total=len(train_iter) * config.num_epochs)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')    #初始化在验证集上的loss为inf（无穷大?）
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    Train_loss = []
    Train_acc = []
    Dev_loss = []
    Dev_acc = []
    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        for i, (trains, labels) in enumerate(train_iter):   #emumerate 生成索引+元素;
            #utils.py:  train_iter 是 (token_ids, seq_len, mask), int(label)
            #labels.shape->[batch_size]
            outputs = model(trains) #outputs.shape->[batch_size,num_classes ]
            model.zero_grad()   #每个batch 都清空梯度
            Floss = FocalLoss()
            loss = Floss(outputs, labels)
            # loss = F.cross_entropy(outputs, labels) #交叉熵损失函数
            loss.backward()
            optimizer.step()
            if (total_batch) % 1 == 0: #+1，26，53
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()    #torch.max(input,dim),dim=1 每行最大值；output->(每行最大值，最大值的索引)
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))    #.item() 精度比直接取值要高。另外还可用于遍历中。
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()

                Train_loss.append(loss.item())
                Train_acc.append(train_acc)
                Dev_loss.append(dev_loss)
                Dev_acc.append(dev_acc)

            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

    writer.close()
    print('Train_loss:',Train_loss)
    print('Train_acc:', Train_acc)
    print('Dev_loss:', Dev_loss)
    print('Dev_acc:', Dev_acc)
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False): #训练过程的test传为False，在测试过程的test传为True
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():   #不在传输梯度
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
