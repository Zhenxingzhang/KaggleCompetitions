__author__ = 'zhenxing'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

def Name_Extract(full_name):
    return full_name.split(',')[1].split('.')[0].strip()

def groud_Salutation(salutation):
    if salutation in ("Mr" , "Mrs" , "Master" , "Miss"):
        return False
    else:
        return True

train = pd.read_csv("../Data/train.csv")

print train.info()

print "********************************************"

print train.columns.values

Embarked_stat = train["Embarked"].value_counts()

Sex_stats = train["Sex"].value_counts()

plt.figure();
Sex_stats.plot(kind='bar')
plt.show()

# train['Salutation'] = pd.DataFrame({'Salutation':train['Name'].apply(Name_Extract)})
Salutation = pd.DataFrame({'Salutation':train['Name'].apply(Name_Extract)})
Others_idx = Salutation['Salutation'].apply(groud_Salutation)
Salutation['Salutation'][Others_idx]= 'Others'

train['Salutation'] = Salutation

Salutation_stats = train['Salutation'].value_counts()

train.boxplot(column ='Age', by='Salutation')
plt.xlabel("Salutation")
plt.ylabel("Age")
plt.show()

Age_Train = train[train['Age'].notnull()][['Sex','Salutation', 'Pclass']]

Age_Train.loc[Age_Train["Sex"] == "male", "Sex"] = 0
Age_Train.loc[Age_Train["Sex"] == "female", "Sex"] = 1

Age_Train.loc[Age_Train["Salutation"] == "Mr", "Salutation"] = 0
Age_Train.loc[Age_Train["Salutation"] == "Mrs", "Salutation"] = 1
Age_Train.loc[Age_Train["Salutation"] == "Master", "Salutation"] = 2
Age_Train.loc[Age_Train["Salutation"] == "Miss", "Salutation"] = 3
Age_Train.loc[Age_Train["Salutation"] == "Others", "Salutation"] = 4

Age_Result = train[train['Age'].notnull()]['Age']

clf = LinearRegression()

scores = cross_validation.cross_val_score(clf, Age_Train, Age_Result, cv=5)

clf.fit(Age_Train, Age_Result)

predictions = clf.predict(Age_Train)

plt.subplot(2, 1, 1)
plt.hist(Age_Result, bins=10, range = (Age_Result.min(),Age_Result.max()))

plt.subplot(2, 1, 2)
plt.hist(predictions, bins=10, range = (Age_Result.min(),Age_Result.max()))
plt.show()

print "Mean Score: ",scores.mean()

exit()

Age_Test = train[train['Age'].isnull()]

# plt.bar(Embarked_stat.axes, Embarked_stat.data)
# plt.ylabel('some numbers')
# plt.show()

plt.hist(train['Age'], bins = 90, range = (train['Age'].min(),train['Age'].max()))
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()

plt.hist(train['Fare'], bins = 10, range = (train['Fare'].min(),train['Fare'].max()))
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count of Passengers")
plt.show()

train.boxplot(column = 'Fare')
plt.show()

train.boxplot(column = 'Fare', by = 'Pclass')
plt.show()