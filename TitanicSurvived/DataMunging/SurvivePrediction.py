import AgePrediction as AP
import pandas as pd

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

titanic_df = pd.read_csv("../Data/train.csv")

titanic_df['Salutation'] = AP.Salutation_Assign(titanic_df)

titanic_df = AP.Convert_Non_Numeric(titanic_df)

train_age = titanic_df[titanic_df['Age'].notnull()]
Age_Test = titanic_df[titanic_df['Age'].isnull()]
predictions = AP.Age_Prediction(train_age, Age_Test)
titanic_df['Age'] = titanic_df['Age'].fillna(pd.Series(predictions, index=Age_Test.index.values))

# titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

titanic_df["FamilySize"] = titanic_df["SibSp"] + titanic_df["Parch"]

titanic_df["NameLength"] = titanic_df["Name"].apply(lambda x: len(x))

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Salutation"]


titanic_train = titanic_df[predictors]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(titanic_train)
titanic_train = scaler.transform(titanic_train)

# Initialize our algorithm
# alg = LogisticRegression(random_state=1)
# alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
alg = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(5, 5, 2), max_iter="1000", random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic_train, titanic_df["Survived"], cv=5)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

# alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
# alg.fit(titanic_train, titanic_df["Survived"])
#
# titanic_test = pd.read_csv("../Data/test.csv")
#
#
#
# titanic_test['Salutation'] = AP.Salutation_Assign(titanic_test)
#
# titanic_test = AP.Convert_Non_Numeric(titanic_test)
#
# Age_Test = titanic_test[titanic_test['Age'].isnull()]
# predictions = AP.Age_Prediction(train_age, Age_Test)
# titanic_test['Age'] = titanic_test['Age'].fillna(pd.Series(predictions, index=Age_Test.index.values))
#
# # titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
# titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].mean())
#
# titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_df["Parch"]
# titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
#
# # print titanic_test.describe()
# titanic_test_data = scaler.transform(titanic_test[predictors])
#
# SurvivedPrediction = alg.predict(titanic_test_data)
#
# result_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
#
# result_df['PassengerId'] = titanic_test['PassengerId']
# result_df['Survived'] = SurvivedPrediction
# result_df.to_csv("Zhenxing_Titanic_Result.csv", header=True,index=False)
#
# print(SurvivedPrediction)