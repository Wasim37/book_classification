'''
@Author: your name
@Date: 2020-04-08 17:21:28
@LastEditTime: 2020-07-06 20:18:42
@LastEditors: xiaoyao jiang
@Description: Process data then get feature
@FilePath: /bookClassification(ToDo)/src/data/mlData.py
'''

import numpy as np
import pandas as pd
import json
import os
from __init__ import *
from src.utils import config
from src.utils.tools import create_logger, wam, query_cut
from src.word2vec.embedding import Embedding
logger = create_logger(config.log_dir + 'data.log')


class MLData(object):
    def __init__(self, debug_mode=False):
        '''
        @description: initlize ML dataset class
        @param {type}
        debug_mode: if debug_Mode the only deal 10000 data
        em, new embedding class
        @return:None
        '''
        self.debug_mode = debug_mode
        self.em = Embedding()
        self.em.load()
        self.preprocessor()

    def preprocessor(self):
        '''
        @description: Preprocess data, segment, transform label to id
        @param {type}None
        @return: None
        '''
        logger.info('load data')
        self.train = pd.read_csv(config.root_path + '/data/train.tsv',
                                 sep='\t').dropna()
        self.dev = pd.read_csv(config.root_path + '/data/dev.tsv',
                               sep='\t').dropna()
        if self.debug_mode:
            self.train = self.train.sample(n=1000).reset_index(drop=True)
            self.dev = self.dev.sample(n=100).reset_index(drop=True)
        ### TODO:
        # 1. 分词
        # 2. 去除停止词
        # 3. 将label 转换为id
        self.train["queryCut"] =
        self.dev["queryCut"] =
        self.train["queryCutRMStopWord"] =
        self.dev["queryCutRMStopWord"] =
        self.train["labelIndex"] =
        self.dev["labelIndex"] =

    def process_data(self, method='word2vec'):
        '''
        @description: generate data used for sklearn
        @param {type}
        method: three options, word2vec, fasttext, tfidf
        @return:
        X_train, feature of train set
        X_test, feature of test set
        y_train, label of train set
        y_test, label of test set
        '''
        X_train = self.get_feature(self.train, method)
        X_test = self.get_feature(self.dev, method)
        y_train = self.train["labelIndex"]
        y_test = self.dev["labelIndex"]
        return X_train, X_test, y_train, y_test

    def get_feature(self, data, method='word2vec'):
        '''
        @description: generate feature
        @param {type}
        data, input dataset
        method: three options, word2vec, fasttext, tfidf
        @return: coresponding feature
        '''
        if method == 'tfidf':
            data = [' '.join(query) for query in data["queryCutRMStopWord"]]
            return self.em.tfidf.transform(data)
        elif method == 'word2vec':
            # return [np.array(wam(x, self.em.w2v)) for x in data['text'].values.tolist()]
            return np.vstack(data['queryCutRMStopWord'].apply(
                lambda x: wam(x, self.em.w2v)[0]))
        elif method == 'fasttext':
            return np.vstack(data['queryCutRMStopWord'].apply(
                lambda x: wam(x, self.em.fast)[0]))
        else:
            NotImplementedError
