__author__ = 'zhenxing'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("../Data/train.csv")

print train.info()

print "********************************************"

print train.columns.values

Embarked_stat = train["Embarked"].value_counts()

plt.figure();
Embarked_stat.plot(kind='bar')
plt.show()

print Embarked_stat

# plt.bar(Embarked_stat.axes, Embarked_stat.data)
# plt.ylabel('some numbers')
# plt.show()