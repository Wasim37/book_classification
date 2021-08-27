'''
@Author: your name
@Date: 2020-06-18 21:15:35
LastEditTime: 2021-08-27 09:20:19
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /bookClassification(ToDo)/src/DL/models/bert.py
'''
# coding: UTF-8
import torch.nn as nn
from transformers import BertModel, BertConfig


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(
            config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path,
                                              config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # TODO
        # 构建bert 分类模型
        context = x[0]
        mask = x[1]
        token_type_ids = x[2]
        _, pooled = self.bert(context,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        out = self.fc(pooled)
        return out
        