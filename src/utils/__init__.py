'''
@Author: your name
@Date: 2020-04-08 16:16:44
@LastEditTime: 2020-04-08 16:16:44
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /textClassification/src/utils/__init__.py
'''
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
