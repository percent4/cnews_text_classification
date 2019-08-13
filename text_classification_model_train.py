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