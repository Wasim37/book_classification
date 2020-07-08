## 文件介绍与运行说明

### First, run `src/word2vec/embedding.py` to generate all word embedding (include word2vec, fasttext, tfidf, lda, autoencoder).

### Second, run `src/ML/main.py` to run all kinds of models.

### Third, run `src/DL/train.py` to run bert model.

## 代码结构介绍
`data`: 数据存放目录
`model` : 模型存放目录
`logs` : 日志存放目录
`src` : 核心代码部分
`app.py` : 代码部署部分
`src/data` : 数据处理部分
`src/data/dataset.py` : 主要用于深度学习的数据处理
`src/data/mlData.py` : 主要用于机器学习的数据处理
`src/DL/` : 包含各类深度学习模型， 运行主入口为`src/DL/train.py`
`src/ML/` : 包含各类机器学习模型， 运行主入口为`src/ML/main.py`
`src/utils/` : 包含配置文件，特征工程函数，以及通用函数
`src/word2vec/` : 包含各类embedding的训练，保存加载。运行主入口为`src/word2vec/embedding.py`

## 作业
在本次作业中， 你需要完成一下几项内容：
1. 使用`gensim` 训练` LDA ` 模型。 并学会加载和保存
2. 了解如何保存/加载 `keras` 模型
3. 对于代码中的类可能会多次初始化， 为了避免浪费资源以及加快效率， 需要了解什么是单例模式。
4. 通过` jieba ` 进行词性标注， 掌握提取名词， 动词， 形容词的技术。
5. 通过` torchvision ` 记载预训练模型， 并转化图片为embedding（下载下来的模型文件， 需复制到`/home/jovyan/.cache/torch/checkpoints/` 目录下。 另外需要注意的是需要将book_cover的压缩包解压至代码的`data` 目录， 不是`src/data`目录下）
6. 下载bert 预训练模型， 保存到`model/`下， 掌握如何获取`bert embedding `
7. 加载预训练好的`lda`模型， 并生成`lda feature`
8. 加载训练好的`autoencoder`模型， 并生成`autoencoder feature`
9. 了解`pytorch ` 的 `Dataset` 和 `DataLoader`， 了解如何进行`padding`
10. 完善`bert` 模型的` forward `方法， 以及训练时 创建 `Dataset, DataLoader`， 初始化` optimizer` 以及训练时的核心代码
11. 如何使用`under_sampling`, `over_sampling`等技术， 处理不平衡样本。
12. 熟练掌握计算`percision`, `accuracy`, `recall`, `f1_score` 等指标。
13. 熟练掌握使用`flask`部署模型。