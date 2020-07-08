'''
@Author: your name
@Date: 2020-06-28 14:02:35
@LastEditTime: 2020-07-06 22:00:21
@LastEditors: xiaoyao jiang
@Description: In User Settings Edit
@FilePath: /bookClassification(ToDo)/app.py
'''
from flask import Flask, request
from src.word2vec.test import Similarity
import json

sim = Similarity()

app = Flask(__name__)
# depth filepath


@app.route('/predict', methods=["POST"])
def gen_ans():
    ### TODO
    # 1. 接受request 输入 并返回预测结果


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890, debug=True)