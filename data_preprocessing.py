# -*- coding: utf-8 -*-
# time: 2019-08-13 11:16
# place: Pudong Shanghai

from collections import defaultdict

# 获取每个数据集的分类及数量
def get_label_num(data_type):
    with open('./data/cnews.%s.txt' % data_type, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines() if _.strip()]

    label_dict = defaultdict(int)
    for line in content:
        label, text = line.split(maxsplit=1)
        label_dict[label] += 1

    return dict(label_dict)

train_label_dict = get_label_num('train')
val_label_dict = get_label_num('val')
test_label_dict = get_label_num('test')

print('、'.join(list(train_label_dict.keys())))
print(train_label_dict)
print(val_label_dict)
print(test_label_dict)

'''
输出结果：
体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经
{'体育': 5000, '娱乐': 5000, '家居': 5000, '房产': 5000, '教育': 5000, '时尚': 5000, '时政': 5000, '游戏': 5000, '科技': 5000, '财经': 5000}
{'体育': 500, '娱乐': 500, '家居': 500, '房产': 500, '教育': 500, '时尚': 500, '时政': 500, '游戏': 500, '科技': 500, '财经': 500}
{'体育': 1000, '娱乐': 1000, '家居': 1000, '房产': 1000, '教育': 1000, '时尚': 1000, '时政': 1000, '游戏': 1000, '科技': 1000, '财经': 1000}
'''