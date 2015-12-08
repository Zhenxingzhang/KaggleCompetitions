__author__ = 'zhenxing'
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

titanic_df = pd.read_csv("../Data/train.csv")


def Name_Extract(full_name):
    return full_name.split(',')[1].split('.')[0].strip()


def Salutation_Filter(salutation):
    if salutation in ("Mr", "Mrs", "Master", "Miss"):
        return False
    else:
        return True


def Salutation_Assign(data):
    # titanic_df['Salutation'] = pd.DataFrame({'Salutation':titanic_df['Name'].apply(Name_Extract)})
    Salutation = pd.DataFrame({'Salutation': data['Name'].apply(Name_Extract)})
    Others_idx = Salutation['Salutation'].apply(Salutation_Filter)
    Salutation['Salutation'][Others_idx] = 'Others'
    return Salutation


def Convert_Non_Numeric(titanic_df):
    titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0
    titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1

    titanic_df.loc[titanic_df["Salutation"] == "Master", "Salutation"] = 0
    titanic_df.loc[titanic_df["Salutation"] == "Miss", "Salutation"] = 1
    titanic_df.loc[titanic_df["Salutation"] == "Mr", "Salutation"] = 2
    titanic_df.loc[titanic_df["Salutation"] == "Mrs", "Salutation"] = 3
    titanic_df.loc[titanic_df["Salutation"] == "Others", "Salutation"] = 4

    titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
    titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0
    titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 1
    titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 2

    return titanic_df

def Age_Prediction(train_df, Age_Test):
    Feature_List = ['Sex', 'Salutation', 'Pclass']
    Age_titanic_df = train_df[Feature_List]
    Age_Result = train_df['Age']

    clf = LinearRegression(True, True)

    lr_scores = cross_validation.cross_val_score(clf, Age_titanic_df, Age_Result, scoring='r2', cv=5)
    print "Linear Regression mean score: ", lr_scores.mean()

    Age_Test = Age_Test[Feature_List]
    clf.fit(Age_titanic_df, Age_Result)
    age_results = clf.predict(Age_Test)

    # svr_poly = SVR(kernel='linear', C=1e3)
    # svm_scores = cross_validation.cross_val_score(svr_poly, Age_titanic_df, Age_Result, scoring = 'r2', cv=5)
    # print "svr_poly mean score: ",svm_scores.mean()
    #
    # svr_poly.fit(Age_titanic_df, Age_Result)
    # predictions = svr_poly.predict(Age_titanic_df)

    return age_results

titanic_df['Salutation'] = Salutation_Assign(titanic_df)

titanic_df = Convert_Non_Numeric(titanic_df)

Salutation_stats = titanic_df['Salutation'].value_counts()

titanic_df.boxplot(column='Age', by='Salutation')
plt.xlabel("Salutation")
plt.ylabel("Age")
plt.show()

# titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())

train_age = titanic_df[titanic_df['Age'].notnull()]

Age_Result = train_age['Age']

predictions = Age_Prediction(train_age, train_age)

# plt.subplot(2, 1, 1)
# plt.hist(Age_Result, bins=10, range = (Age_Result.min(),Age_Result.max()))
#
# plt.subplot(2, 1, 2)
# plt.hist(predictions, bins=10, range = (Age_Result.min(),Age_Result.max()))
# plt.show()

Age_Test = titanic_df[titanic_df['Age'].isnull()]
predictions = Age_Prediction(train_age, Age_Test)

titanic_df['Age'] = titanic_df['Age'].fillna(pd.Series(predictions, index=Age_Test.index.values))

# print titanic_df['Age'].describe()
# plt.subplot(2, 1, 1)
# plt.hist(train_age['Age'], bins=10, range = (titanic_df['Age'].min(),titanic_df['Age'].max()))
#
# plt.subplot(2, 1, 2)
# plt.hist(titanic_df['Age'], bins=10, range = (titanic_df['Age'].min(),titanic_df['Age'].max()))
# plt.show()

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm
# alg = LogisticRegression(random_state=1)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic_df[predictors], titanic_df["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())