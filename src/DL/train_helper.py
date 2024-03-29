'''
@Author: wangxin
@Date: 2020-04-09 17:46:03
LastEditTime: 2021-08-27 16:25:07
LastEditors: Please set LastEditors
@Description: train
@FilePath: /bookClassification(ToDo)/src/DL/train_helper.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from tqdm import tqdm
from __init__ import *
from src.utils.tools import get_time_dif
from transformers import AdamW, get_linear_schedule_with_warmup


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
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


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    if config.model_name.isupper():
        print('User Adam...')
        print(config.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        print('User AdamW...')
        print(config.device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        # TODO
        # 1. 初始化AdamW 优化器
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate,
                          eps=config.eps)


#         scheduler = get_linear_schedule_with_warmup(
#                                 optimizer,
#                                 num_warmup_steps=0,
#                                 num_training_steps=((len(train_iter) * num_train_epochs) // \
#                                                     (config.batch_size * config.gradient_accumulation_steps)) + 1000)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, mask, tokens, labels) in tqdm(enumerate(train_iter)):
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            
            # TODO
            # 1. 加载模型进行训练
            # 2. 清空梯度
            # 3. 计算loss
            # 4. loss backpropergation
            # 5. 优化器step
            outputs = model((trains, mask, tokens))
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            #             scheduler.step()
            if total_batch % 1000 == 0 and total_batch != 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
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
                print(
                    msg.format(total_batch, loss.item(), train_acc, dev_loss,
                               dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config,
                                                                model,
                                                                test_iter,
                                                                test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, mask, tokens, labels in tqdm(data_iter):
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            outputs = model((texts, mask, tokens))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all,
                                               predict_all,
                                               target_names=config.class_list,
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
