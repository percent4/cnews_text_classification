# cnews_text_classification
利用kashgari轻松搭建文本分类模型。示例数据集下载网址：https://download.csdn.net/download/wenweno0o/11206035 。

### 背景介绍

&emsp;&emsp;文本分类是NLP中的常见的重要任务之一，它的主要功能就是将输入的文本以及文本的类别训练出一个模型，使之具有一定的泛化能力，能够对新文本进行较好地预测。它的应用很广泛，在很多领域发挥着重要作用，例如垃圾邮件过滤、舆情分析以及新闻分类等。\
&emsp;&emsp;现阶段的文本分类模型频出，种类繁多，花样百变，既有机器学习中的朴素贝叶斯模型、SVM等，也有深度学习中的各种模型，比如经典的CNN, RNN，以及它们的变形，如CNN-LSTM，还有各种高大上的Attention模型。\
&emsp;&emsp;无疑，文本分类是一个相对比较成熟的任务，我们尽可以选择自己喜欢的模型来完成该任务。本文以kashgari-tf为例，它能够支持各种文本分类模型，比如BiLSTM，CNN_LSTM，AVCNN等，且对预训练模型，比如BERT的支持较好，它能让我们轻松地完成文本分类任务。\
&emsp;&emsp;下面，让我们一起走进文本分类的世界，分分钟搞定text classification！

### 项目

&emsp;&emsp;首先，我们需要找一份数据作为例子。我们选择THUCNews，THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，从中选择10个候选分类类别：体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经。\
&emsp;&emsp;数据总量一共为6.5万条，其中训练集数据5万条，每个类别5000条，验证集数据0.5万条，每个类别500条，测试集数据1万条，每个类别1000条。笔者已将数据放在Github上，读者可以在最后的总结中找到。\
&emsp;&emsp;项目结构，如下图：

![文本分类项目结构](https://upload-images.jianshu.io/upload_images/9419034-bcf709cd3f7ed102.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


&emsp;&emsp;接着，我们尝试着利用kashgari-tf来训练一个文本分类模型，其中模型我们采用CNN-LSTM，完整的Python代码（text_classification_model_train.py）如下：

```python
# -*- coding: utf-8 -*-
# time: 2019-08-13 11:16
# place: Pudong Shanghai

from kashgari.tasks.classification import CNN_LSTM_Model

# 获取数据集
def load_data(data_type):
    with open('./data/cnews.%s.txt' % data_type, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines() if _.strip()]

    x, y = [], []
    for line in content:
        label, text = line.split(maxsplit=1)
        y.append(label)
        x.append([_ for _ in text])

    return x, y

# 获取数据
train_x, train_y = load_data('train')
valid_x, valid_y = load_data('val')
test_x, test_y = load_data('test')

# 训练模型
model = CNN_LSTM_Model()
model.fit(train_x, train_y, valid_x, valid_y, batch_size=16, epochs=5)

# 评估模型
model.evaluate(test_x, test_y)

# 保存模型
model.save('text_classification_model')
```

输出的模型结果如下：

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           (None, 2544)              0
_________________________________________________________________
layer_embedding (Embedding)  (None, 2544, 100)         553200
_________________________________________________________________
conv1d (Conv1D)              (None, 2544, 32)          9632
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 1272, 32)          0
_________________________________________________________________
cu_dnnlstm (CuDNNLSTM)       (None, 100)               53600
_________________________________________________________________
dense (Dense)                (None, 10)                1010
=================================================================
Total params: 617,442
Trainable params: 617,442
Non-trainable params: 0
```

&emsp;&emsp;设定模型训练次数为5个epoch，batch_size为16。模型训练完后，在训练集、验证集上的结果如下：

|数据集|accuracy|loss|
|---|---|---|
|训练集|0.9661|0.1184|
|验证集|0.9204|0.2567|

在测试集上的结果如下：

```
             precision    recall  f1-score   support

          体育     0.9852    0.9970    0.9911      1000
          娱乐     0.9938    0.9690    0.9813      1000
          家居     0.9384    0.8830    0.9098      1000
          房产     0.9490    0.9680    0.9584      1000
          教育     0.9650    0.8820    0.9216      1000
          时尚     0.9418    0.9710    0.9562      1000
          时政     0.9732    0.9450    0.9589      1000
          游戏     0.9454    0.9700    0.9576      1000
          科技     0.8910    0.9560    0.9223      1000
          财经     0.9566    0.9920    0.9740      1000

    accuracy                         0.9533     10000
   macro avg     0.9539    0.9533    0.9531     10000
weighted avg     0.9539    0.9533    0.9531     10000
```

&emsp;&emsp;总的来说，上述模型训练的效果还是很不错的。接下来，是考验模型的预测能力的时刻了，看看它是否具体文本分类的泛化能力。

### 测试

&emsp;&emsp;我们已经有了训练好的模型`text_classification_model`，接着让我们利用该模型来对新的数据进行预测，预测的代码（model_predict.py）如下:

```python
# -*- coding: utf-8 -*-
# time: 2019-08-14 00:21
# place: Pudong Shanghai

import kashgari

# 加载模型
loaded_model = kashgari.utils.load_model('text_classification_model')

text = '华夏幸福成立于 1998 年，前身为廊坊市华夏房地产开发有限公司，初始注册资本 200 万元，其中王文学出资 160 万元，廊坊市融通物资贸易有限公司出资 40 万元，后经多次股权转让和增资，公司于 2007 年整体改制为股份制公司，2011 年完成借壳上市。'

x = [[_ for _ in text]]

label = loaded_model.predict(x)
print('预测分类:%s' % label)
```

以下是测试结果：

> 原文1: 华夏幸福成立于 1998 年，前身为廊坊市华夏房地产开发有限公司，初始注册资本 200 万元，其中王文学出资 160 万元，廊坊市融通物资贸易有限公司出资 40 万元，后经多次股权转让和增资，公司于 2007 年整体改制为股份制公司，2011 年完成借壳上市。
分类结果：预测分类:['财经']

> 原文2: 现今常见的短袖衬衫大致上可以分为：夏威夷衬衫、古巴衬衫、保龄球衫，三者之间虽有些微分别，但其实有些时候，一件衬衫也可能包含了多种款式的特色。而‘古巴（领）衬衫’最显而易见的特点在于‘领口’，通常会设计为V领，且呈现微微的外翻，也因此缺少衬衫领口常见的‘第一颗钮扣’，衣服到领子的剪裁为一体成形，整体较宽松舒适。
分类结果：预测分类:['时尚']

> 原文3:周琦2014年加盟新疆广汇篮球俱乐部，当年就代表俱乐部青年队接连拿下全国篮球青年联赛冠军和全国俱乐部青年联赛冠军。升入一队后，周琦2016年随队出战第25届亚冠杯，获得冠军。2016-2017赛季，周琦为新疆广汇队夺得队史首座总冠军奖杯立下汗马功劳，他在总决赛中带伤出战，更是传为佳话。
分类结果：预测分类:['体育']

> 原文4: 周杰伦[微博]监制赛车电影《叱咤风云》13日释出花絮导演篇，不仅真实赛车竞速画面大量曝光，几十辆百万赛车在国际专业赛道、山路飙速，场面浩大震撼，更揭开不少
现场拍摄的幕后画面。监制周杰伦在现场与导演讨论剧本、范逸臣[微博]与高英轩大打出手、甚至有眼尖网友发现在花絮中闪过“男神”李玉玺[微博]的画面。
分类结果：预测分类:['娱乐']

> 原文5: 北京时间8月13日上午消息，据《韩国先驱报》网站报道，近日美国知识产权所有者协会（ Intellectual Property Owners Association）发布的一份报告显示，在获得的
美国专利数量方面，IBM、微软和通用电气等美国企业名列前茅，排在后面的韩国科技巨头三星、LG与之竞争激烈。
分类结果：预测分类:['科技']

### 总结

&emsp;&emsp;虽然我们上述测试的文本分类效果还不错，但也存在着一些分类错误的情况。\
&emsp;&emsp;本文讲述了如何利用kashgari-tf模块来快速地搭建文本分类任务，其实，也没那么难！\
&emsp;&emsp;本文代码和数据及已上传至Github, 网址为：
[https://github.com/percent4/cnews_text_classification](https://github.com/percent4/cnews_text_classification)

> 注意：不妨了解下笔者的微信公众号： Python爬虫与算法（微信号为：easy_web_scrape）， 欢迎大家关注~

