'''
@Author: xiaoyao jiang
@Date: 2020-04-08 19:39:30
@LastEditTime: 2020-07-06 22:02:42
@LastEditors: xiaoyao jiang
@Description: There are two options. One is using pretrained embedding as feature to compare common ML models.
              The other is using feature engineering + param search tech + imbanlance to train a liaghtgbm model.
@FilePath: /bookClassification(ToDo)/src/ML/models.py
'''
import os

import lightgbm as lgb
import torchvision
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import BertModel, BertTokenizer

from src.data.mlData import MLData
from src.utils import config
from src.utils.config import root_path
from src.utils.tools import (Grid_Train_model, bayes_parameter_opt_lgb,
                             create_logger, formate_data, get_score)
from src.utils.feature import (get_embedding_feature, get_img_embedding,
                               get_lda_features, get_pretrain_embedding,
                               get_autoencoder_feature, get_basic_feature)

logger = create_logger(config.log_dir + 'model.log')


class Models(object):
    def __init__(self, feature_engineer=False):
        '''
        @description: initlize Class, EX: model
        @param {type} :
        feature_engineer: whether using feature engineering, if `False`, then compare common ML models
        res_model: res network model
        resnext_model: resnext network model
        wide_model: wide res network model
        bert: bert model
        ml_data: new mldata class
        @return: No return
        '''
        ### TODO
        # 1. 使用torchvision 初始化resnet152模型
        # 2. 使用torchvision 初始化 resnext101_32x8d 模型
        # 3. 使用torchvision 初始化  wide_resnet101_2 模型
        # 4. 加载bert 模型
        self.res_model =
        self.res_model = self.res_model.to(config.device)
        self.resnext_model =
        self.resnext_model = self.resnext_model.to(config.device)
        self.wide_model =
        self.wide_model = self.wide_model.to(config.device)
        self.bert_tonkenizer = BertTokenizer.from_pretrained(
            config.root_path + '/../textClassification/model/bert')
        self.bert =
        self.bert = self.bert.to(config.device)

        self.ml_data = MLData(debug_mode=True)
        if feature_engineer:
            self.model = lgb.LGBMClassifier(objective='multiclass',
                                            n_jobs=10,
                                            num_class=33,
                                            num_leaves=30,
                                            reg_alpha=10,
                                            reg_lambda=200,
                                            max_depth=3,
                                            learning_rate=0.05,
                                            n_estimators=2000,
                                            bagging_freq=1,
                                            bagging_fraction=0.9,
                                            feature_fraction=0.8,
                                            seed=1440)
        else:
            self.models = [
                RandomForestClassifier(n_estimators=500,
                                       max_depth=5,
                                       random_state=0),
                LogisticRegression(solver='liblinear', random_state=0),
                MultinomialNB(),
                SVC(),
                lgb.LGBMClassifier(objective='multiclass',
                                   n_jobs=10,
                                   num_class=33,
                                   num_leaves=30,
                                   reg_alpha=10,
                                   reg_lambda=200,
                                   max_depth=3,
                                   learning_rate=0.05,
                                   n_estimators=2000,
                                   bagging_freq=1,
                                   bagging_fraction=0.8,
                                   feature_fraction=0.8),
            ]

    def feature_engineer(self):
        '''
        @description: This function is building all kings of features
        @param {type} None
        @return:
        X_train, feature of train set
        X_test, feature of test set
        y_train, label of train set
        y_test, label of test set
        '''
        logger.info("generate embedding feature ")
        train_tfidf, test_tfidf, train, test = get_embedding_feature(
            self.ml_data)

        logger.info("generate basic feature ")
        #### 本代码中的函数实现均在utils/feature.py中
        #### 本代码中的函数实现均在utils/feature.py中
        #### 本代码中的函数实现均在utils/feature.py中
        ### TODO
        # 1. 获取 基本的 NLP feature

        logger.info("generate modal feature ")
        cover = os.listdir(config.root_path + '/data/book_cover/')
        train['cover'] = train.title.progress_apply(
            lambda x: config.root_path + '/data/book_cover/' + x + '.jpg'
            if x + '.jpg' in cover else '')
        test['cover'] = test.title.progress_apply(
            lambda x: config.root_path + '/data/book_cover/' + x + '.jpg'
            if x + '.jpg' in cover else '')

        ### TODO
        # 1. 获取 三大CV模型的 modal embedding
        train['res_embedding'] =
        test['res_embedding'] =

        train['resnext_embedding'] =
        test['resnext_embedding'] =

        train['wide_embedding'] =
        test['wide_embedding'] =

        logger.info("generate bert feature ")
        ### TODO
        # 1. 获取bert embedding
        train['bert_embedding'] =
        test['bert_embedding'] =

        logger.info("generate lda feature ")

        ### TODO
        # 1. 获取 lda feature
        train['bow'] =
        test['bow'] =
        train['lda'] =
        test['lda'] =

        logger.info("generate autoencoder feature ")
        ### TODO
        # 1. 获取 autoencoder feature
        train_ae, test_ae =

        logger.info("formate data")
        train, test = formate_data(train, test, train_tfidf, test_tfidf,
                                   train_ae, test_ae)
        cols = [x for x in train.columns if str(x) not in ['labelIndex']]
        X_train = train[cols]
        X_test = test[cols]
        train["labelIndex"] = train["labelIndex"].astype(int)
        test["labelIndex"] = test["labelIndex"].astype(int)
        y_train = train["labelIndex"]
        y_test = test["labelIndex"]
        return X_train, X_test, y_train, y_test

    def param_search(self, search_method='grid'):
        '''
        @description: use param search tech to find best param
        @param {type}
        search_method: two options. grid or bayesian optimization
        @return: None
        '''
        if search_method == 'grid':
            logger.info("use grid search")
            self.model = Grid_Train_model(self.model, self.X_train,
                                          self.X_test, self.y_train,
                                          self.y_test)
        elif search_method == 'bayesian':
            logger.info("use bayesian optimization")
            trn_data = lgb.Dataset(data=self.X_train,
                                   label=self.y_train,
                                   free_raw_data=False)
            tst_data = lgb.Dataset(data=self.X_test,
                                   label=self.y_test,
                                   free_raw_data=False)
            param = bayes_parameter_opt_lgb(trn_data)
            logger.info("best param", param)
            param['objective'] = 'multiclass'
            param['metric'] = 'auc'
            self.model = lgb.train(param,
                                   trn_data,
                                   valid_sets=[trn_data, tst_data],
                                   verbose_eval=1000,
                                   early_stopping_rounds=100)

    def unbalance_helper(self,
                         imbalance_method='under_sampling',
                         search_method='grid'):
        '''
        @description: handle unbalance data, then search best param
        @param {type}
        imbalance_method,  three option, under_sampling for ClusterCentroids, SMOTE for over_sampling, ensemble for BalancedBaggingClassifier
        search_method: two options. grid or bayesian optimization
        @return: None
        '''
        logger.info("get all freature")
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer(
        )
        model_name = None
        if imbalance_method == 'over_sampling':
            logger.info("Use SMOTE deal with unbalance data ")
            ### TODO
            # 1. 使用over_sampling 处理样本不平衡问题
            self.X_train, self.y_train =
            self.X_test, self.y_test =
            model_name = 'lgb_over_sampling'
        elif imbalance_method == 'under_sampling':
            logger.info("Use ClusterCentroids deal with unbalance data ")
            ### TODO
            # 1. 使用 under_sampling 处理样本不平衡问题
            self.X_train, self.y_train =
            self.X_test, self.y_test =
            model_name = 'lgb_under_sampling'
        elif imbalance_method == 'ensemble':
            self.model = BalancedBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                sampling_strategy='auto',
                replacement=False,
                random_state=0)
            model_name = 'ensemble'
        logger.info('search best param')
        if imbalance_method != 'ensemble':
            ### TODO
            # 1. 使用 参数搜索技术
        logger.info('fit model ')
        self.model.fit(self.X_train, self.y_train)
        ### TODO
        # 1. 预测测试集的label
        # 2. 预测训练机的label
        # 3. 计算percision , accuracy, recall, fi_score
        Test_predict_label =
        Train_predict_label =
        per, acc, recall, f1 =
        # 输出训练集的准确率
        logger.info('Train accuracy %s' % per)
        # 输出测试集的准确率
        logger.info('test accuracy %s' % acc)
        # 输出recall
        logger.info('test recall %s' % recall)
        # 输出F1-score
        logger.info('test F1_score %s' % f1)
        self.save(model_name)

    def model_select(self,
                     X_train,
                     X_test,
                     y_train,
                     y_test,
                     feature_method='tf-idf'):
        '''
        @description: using different embedding feature to train common ML models
        @param {type}
        X_train, feature of train set
        X_test, feature of test set
        y_train, label of train set
        y_test, label of test set
        feature_method, three options , tfidf, word2vec and fasttext
        @return: None
        '''
        for model in self.models:
            model_name = model.__class__.__name__
            print(model_name)
            clf = model.fit(X_train, y_train)
            Test_predict_label = clf.predict(X_test)
            Train_predict_label = clf.predict(X_train)
            per, acc, recall, f1 = get_score(y_train, y_test,
                                             Train_predict_label,
                                             Test_predict_label)
            # 输出训练集的准确率
            logger.info(model_name + '_' + 'Train accuracy %s' % per)

            # 输出测试集的准确率
            logger.info(model_name + '_' + ' test accuracy %s' % acc)

            # 输出recall
            logger.info(model_name + '_' + 'test recall %s' % recall)

            # 输出F1-score
            logger.info(model_name + '_' + 'test F1_score %s' % f1)

    def predict(self, text):
        '''
        @description: for a given input, predict its label
        @param {type}
        text: input
        @return: label
        '''
        ### TODO
        # 1. 预测结果


    def save(self, model_name):
        '''
        @description:save model
        @param {type}
        model_name, file name for saving
        @return: None
        '''
        ### TODO
        # 1. 保存模型

    def load(self, path):
        '''
        @description: load model
        @param {type}
        path: model path
        @return:None
        '''
        ### TODO
        # 1. 加载模型

