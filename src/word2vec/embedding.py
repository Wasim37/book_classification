'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
@LastEditTime: 2020-07-06 22:01:20
@LastEditors: xiaoyao jiang
@Description: train embedding & tfidf & autoencoder
@FilePath: /bookClassification(ToDo)/src/word2vec/embedding.py
'''
import pandas as pd
from gensim import models, corpora
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
        self.stopWords = open(root_path + '/data/stopwords.txt',
                              encoding='utf-8').readlines()
        self.ae = AutoEncoder()

    def load_data(self):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        logger.info('load data')
        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train.csv', sep='\t', nrows=212),
            pd.read_csv(root_path + '/data/dev.csv', sep='\t', nrows=60),
            pd.read_csv(root_path + '/data/test.csv', sep='\t', nrows=30)
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
        count_vect = TfidfVectorizer()
        self.tfidf = count_vect.fit_transform(self.data.text)

        logger.info('train word2vec')
        self.data['text'] = self.data.text.apply(lambda x: x.split(' '))
        self.w2v = models.Word2Vec(min_count=2,
                                   window=3,
                                   size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   iter=10,
                                   max_vocab_size=50000)
        self.w2v.build_vocab(self.data.text)
        self.w2v.train(self.data.text,
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)

        # logger.info('train fast')
        # self.fast = models.FastText(sentences=self.data.text,
        #                             size=300,
        #                             window=3,
        #                             alpha=0.03,
        #                             min_alpha=0.0007,
        #                             min_count=2,
        #                             max_vocab_size=1000,
        #                             word_ngrams=1,
        #                             sample=1e-3,
        #                             seed=1,
        #                             workers=1,
        #                             negative=5,
        #                             iter=1)

        logger.info('train lda')
        # hint 使用gensim
        dictionary = corpora.Dictionary(self.data.text)
        # corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
        # corpus是把每条新闻ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
        corpus = [dictionary.doc2bow(text) for text in self.data.text]
        self.LDAmodel = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

        logger.info('train autoencoder')
        self.ae.train(data=self.data)

    def saver(self):
        '''
        @description: save all model 
        @param {type} None
        @return: None
        '''

        logger.info('save tfidf model')
        joblib.dump(self.tfidf, root_path + '/model/embedding/tfidf')

        logger.info('save w2v model')
        self.w2v.wv.save_word2vec_format(root_path +
                                         '/model/embedding/w2v.bin',
                                         binary=False)

        # logger.info('save fast model')
        # self.fast.wv.save_word2vec_format(root_path +
        #                                   '/model/embedding/fast.bin',
        #                                   binary=False)

        logger.info('save lda model')
        self.LDAmodel.save(root_path + '/model/embedding/lda')

        logger.info('save autoencoder model')
        self.ae.save()

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        logger.info('load tfidf model')
        self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')
        # print()

        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/w2v.bin', binary=False)
        print("w2v_embedding输出词表的个数{}".format(len(self.w2v.wv.vocab.keys())))


        # logger.info('load fast model')
        # self.fast = models.KeyedVectors.load_word2vec_format(
        #     root_path + '/model/embedding/fast.bin', binary=False)
        # print(fast_embedding输出词表的个数{}".format(len(self.fast.wv.vocab.keys())))

        logger.info('load lda model')
        self.lda = LdaModel.load(root_path + '/model/embedding/lda')
        print(self.lda)

        logger.info('load autoencoder model')
        self.ae.load()


if __name__ == "__main__":
    em = Embedding()
    # em.load_data()
    # em.trainer()
    # em.saver()
    em.load()
