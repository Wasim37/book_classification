'''
@Author: xiaoyao jiang
@LastEditors: xiaoyao jiang
@Date: 2020-07-01 15:52:07
@LastEditTime: 2020-07-01 16:05:55
@FilePath: /bookClassification/src/ML/main.py
@Desciption: Machine Learning model main function
'''
import argparse

from __init__ import *
from src.utils import config
from src.utils.tools import create_logger
from models import Models

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--feature_engineering', default=False, type=bool, required=True, help='whether use feature engineering')
parser.add_argument('--search_method', default='grid', type=str, required=True, help='grid / bayesian optimzation')
parser.add_argument('--unbalance', default=True, type=bool, required=True, help='wether use imbalance tech')
parser.add_argument('--imbalance_method', default='under_sampling', type=str, required=True, help='under_sampling, over_sampling, ensemble')
parser.add_argument('--model_name', default='lgb_under_sampling', type=str, required=True, help='model_name')
args = parser.parse_args()

logger = create_logger(config.root_path + '/logs/main.log')


if __name__ == '__main__':
    feature_engineering = args.feature_engineering
    m = Models(model_path=config.root_path+'model/ml_model/' + args.model_name)
    if feature_engineering:
        m.unbalance_helper(imbalance_method=args.imbalance_method, search_method=args.search_method)
    else:
        X_train, X_test, y_train, y_test = m.ml_data.process_data(method='tfidf')
        logger.info('model select with tfidf')
        m.model_select(X_train,
                       X_test,
                       y_train,
                       y_test,
                       feature_method='tfidf')

        X_train, X_test, y_train, y_test = m.ml_data.process_data(
            method='word2vec')
        logger.info('model select with word2vec')
        m.model_select(X_train,
                       X_test,
                       y_train,
                       y_test,
                       feature_method='word2vec')

        X_train, X_test, y_train, y_test = m.ml_data.process_data(
            method='fasttext')
        logger.info('model select with fasttext')
        m.model_select(X_train,
                       X_test,
                       y_train,
                       y_test,
                       feature_method='fasttext')
