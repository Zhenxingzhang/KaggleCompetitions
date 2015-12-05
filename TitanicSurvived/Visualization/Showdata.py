__author__ = 'zhenxing'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


titanic_df = pd.read_csv("../Data/train.csv")

print titanic_df.head()

print("----------------------------")
titanic_df.info()
# print train

titanic_df["Sex"].value_counts()

print "********************************************"

print titanic_df.columns

Embarked_stat = titanic_df["Embarked"].value_counts()

plt.figure();
Embarked_stat.plot(kind='bar')
plt.show()

print Embarked_stat

# plt.bar(Embarked_stat.axes, Embarked_stat.data)
# plt.ylabel('some numbers')
# plt.show()