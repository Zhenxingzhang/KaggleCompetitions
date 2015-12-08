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

Sex_stats = titanic_df["Sex"].value_counts()

plt.figure();
Sex_stats.plot(kind='bar')
plt.show()


# plt.bar(Embarked_stat.axes, Embarked_stat.data)
# plt.ylabel('some numbers')
# plt.show()

plt.hist(titanic_df['Age'], bins = 90, range = (titanic_df['Age'].min(),titanic_df['Age'].max()))
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()

plt.hist(titanic_df['Fare'], bins = 10, range = (titanic_df['Fare'].min(),titanic_df['Fare'].max()))
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count of Passengers")
plt.show()

titanic_df.boxplot(column = 'Fare')
plt.show()

titanic_df.boxplot(column = 'Fare', by = 'Pclass')
plt.show()