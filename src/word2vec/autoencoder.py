'''
@Author: your name
@Date: 2020-06-28 13:59:12
LastEditTime: 2021-08-26 15:44:04
LastEditors: Please set LastEditors
@Description: Train a autoencoder model
@FilePath: /bookClassification(ToDo)/src/word2vec/autoencoder.py
'''
from keras.layers import Input, Dense, Bidirectional, Embedding, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras import regularizers
from src.utils.tools import format_data
import joblib
import os
from src.utils.config import root_path
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


class AutoEncoder(object):
    def __init__(self, max_features=500, max_len=200):
        self.max_len = max_len
        self.max_features = max_features
        self.init_model()

    def init_model(self):
        '''
        @description: initlize autoencoder model
        @param {type} None
        @return: None
        '''

        # Input shape
        inp = Input(shape=(self.max_len, ))
        # 查找每个句子的embedding
        encoder = Embedding(self.max_features, 50)(inp)
        # encoder 第一层双向lstm
        encoder = Bidirectional(LSTM(75, return_sequences=True))(encoder)
        # encoder 第二层lstm
        encoder = Bidirectional(
            LSTM(25,
                 return_sequences=True,
                 activity_regularizer=regularizers.l1(10e-5)))(encoder)

        # decoder 双向lstm
        decoder = Bidirectional(LSTM(75, return_sequences=True))(encoder)
        # pooling
        decoder = GlobalMaxPooling1D()(decoder)
        # 两层 全联接 改变维度大小
        decoder = Dense(50, activation='relu')(decoder)
        decoder = Dense(self.max_len)(decoder)
        # 编译模型
        self.model = Model(inputs=inp, outputs=decoder)
        self.model.compile(loss='mean_squared_error',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train(self, data, epochs=1):
        '''
        @description: Train autoencoder model
        @param {type}
        data, train data
        epochs, train how many times
        @return:
        '''
        self.X, self.tokenizer = format_data(data,
                                             self.max_features,
                                             self.max_len,
                                             shuffle=True)
        self.model.fit(self.X,
                       self.X,
                       epochs=epochs,
                       batch_size=128,
                       verbose=1)

    def save(self):
        '''
        @description: 保存模型， 只保存encoder部分， 根据encoder的输出 获取编码后的向量
        @param {type} None
        @return: None
        '''
        joblib.dump(self.tokenizer, root_path + '/model/embedding/tokenizer')

        # TODO
        # 1. 保存 模型文件到/model/embedding/目录下
        self.model.save_weights(root_path + '/model/embedding/autoencoder')
        self.encoder.save_weights(root_path + '/model/embedding/autoencoder_encoder')

    def load(self):
        '''
        @description:  load tokenizer and model
        @param {type} None
        @return: None
        '''
        self.tokenizer = joblib.load(root_path + '/model/embedding/tokenizer')

        # TODO
        # 1. 加载 模型文件
        self.model.load_weights(root_path + '/model/embedding/autoencoder')
        self.encoder.load_weights(root_path + '/model/embedding/autoencoder_encoder')
