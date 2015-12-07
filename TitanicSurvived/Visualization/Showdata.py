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

# titanic_df['Salutation'] = pd.DataFrame({'Salutation':titanic_df['Name'].apply(Name_Extract)})
Salutation = pd.DataFrame({'Salutation':titanic_df['Name'].apply(Name_Extract)})
Others_idx = Salutation['Salutation'].apply(groud_Salutation)
Salutation['Salutation'][Others_idx]= 'Others'

titanic_df['Salutation'] = Salutation

Salutation_stats = titanic_df['Salutation'].value_counts()

titanic_df.boxplot(column ='Age', by='Salutation')
plt.xlabel("Salutation")
plt.ylabel("Age")
plt.show()

Age_titanic_df = titanic_df[titanic_df['Age'].notnull()][['Sex','Salutation', 'Pclass']]

Age_titanic_df.loc[Age_titanic_df["Sex"] == "male", "Sex"] = 0
Age_titanic_df.loc[Age_titanic_df["Sex"] == "female", "Sex"] = 1

Age_titanic_df.loc[Age_titanic_df["Salutation"] == "Mr", "Salutation"] = 0
Age_titanic_df.loc[Age_titanic_df["Salutation"] == "Mrs", "Salutation"] = 1
Age_titanic_df.loc[Age_titanic_df["Salutation"] == "Master", "Salutation"] = 2
Age_titanic_df.loc[Age_titanic_df["Salutation"] == "Miss", "Salutation"] = 3
Age_titanic_df.loc[Age_titanic_df["Salutation"] == "Others", "Salutation"] = 4

Age_Result = titanic_df[titanic_df['Age'].notnull()]['Age']

clf = LinearRegression()

scores = cross_validation.cross_val_score(clf, Age_titanic_df, Age_Result, cv=5)

clf.fit(Age_titanic_df, Age_Result)

predictions = clf.predict(Age_titanic_df)

plt.subplot(2, 1, 1)
plt.hist(Age_Result, bins=10, range = (Age_Result.min(),Age_Result.max()))

plt.subplot(2, 1, 2)
plt.hist(predictions, bins=10, range = (Age_Result.min(),Age_Result.max()))
plt.show()

print "Mean Score: ",scores.mean()

exit()

Age_Test = titanic_df[titanic_df['Age'].isnull()]

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