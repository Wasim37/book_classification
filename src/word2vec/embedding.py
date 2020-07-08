'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
@LastEditTime: 2020-07-06 22:01:20
@LastEditors: xiaoyao jiang
@Description: train embedding & tfidf & autoencoder
@FilePath: /bookClassification(ToDo)/src/word2vec/embedding.py
'''
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
import gensim

from __init__ import *
from src.utils.config import root_path
from src.utils.tools import create_logger, query_cut
from src.word2vec.autoencoder import AutoEncoder
logger = create_logger(root_path + '/logs/embedding.log')


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''
        self.stopWords = open(root_path + '/data/stopwords.txt').readlines()
        self.ae = AutoEncoder()

    def load_data(self):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        logger.info('load data')
        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train.tsv', sep='\t'),
            pd.read_csv(root_path + '/data/dev.tsv', sep='\t'),
            pd.read_csv(root_path + '/data/test.tsv', sep='\t')
        ])
        self.data["text"] = self.data['title'] + self.data['desc']
        self.data["text"] = self.data["text"].apply(query_cut)
        self.data['text'] = self.data.text.apply(lambda x: " ".join(x))

    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        logger.info('train tfidf')
        ### TODO
        # 1. 训练tfidf模型

        logger.info('train word2vec')

        self.data['text'] = self.data.text.apply(lambda x: x.split(' '))
        ### TODO
        # 1. 训练 word2vec


        logger.info('train fast')
        # 训练fast的词向量
        ### TODO
        # 1. 训练fasttest 模型

        logger.info('train lda')
        ### TODO
        # 1. 训练 LDA 模型
        # hint 使用gensim

        logger.info('train autoencoder')
        ### TODO
        # 1. 训练autoencoder 模型
        # hint 同级目录已经实现

    def saver(self):
        '''
        @description: save all model 
        @param {type} None
        @return: None
        '''
        logger.info('save autoencoder model')
        self.ae.save()

        logger.info('save tfidf model')
        joblib.dump(self.tfidf, root_path + '/model/embedding/tfidf')

        logger.info('save w2v model')
        self.w2v.wv.save_word2vec_format(root_path +
                                         '/model/embedding/w2v.bin',
                                         binary=False)

        logger.info('save fast model')
        self.fast.wv.save_word2vec_format(root_path +
                                          '/model/embedding/fast.bin',
                                          binary=False)

        logger.info('save lda model')
        self.LDAmodel.save(root_path + '/model/embedding/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        logger.info('load tfidf model')
        self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')

        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/w2v.bin', binary=False)

        logger.info('load fast model')
        self.fast = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/fast.bin', binary=False)

        logger.info('load lda model')
        self.lda = LdaModel.load(root_path + '/model/embedding/lda')

        logger.info('load autoencoder model')
        self.ae.load()


if __name__ == "__main__":
    em = Embedding()
    em.load_data()
    em.trainer()
    em.saver()
