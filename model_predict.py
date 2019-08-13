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
