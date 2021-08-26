'''
@Author: your name
@Date: 2020-06-28 14:02:35
LastEditTime: 2021-08-26 15:20:40
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /bookClassification(ToDo)/app.py
'''

from flask import Flask, request

from __init__ import *
from src.ML.models import Models
from src.utils import config
import tensorflow as tf
import json
import keras

# sim = Similarity()
tf.get_default_graph()
graph = tf.get_default_graph()
sess = keras.backend.get_session()

model = Models(model_path=config.root_path + '/model/ml_model/lightgbm', train_model=False)

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def gen_ans():
    ### TODO
    # 1. 接受request 输入 并返回预测结果
    result = {}
    title = request.form.get('title')
    desc = request.form.get('desc')
    if not title:
        return u'缺失title参数'
    if not desc:
        return u'缺失desc参数'

    with sess.as_default():
        with graph.as_default():
            label, score = model.predict(title, desc)
    result = {
        "label": label,
        "proba": str(score)
    }
    return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890, debug=True)
