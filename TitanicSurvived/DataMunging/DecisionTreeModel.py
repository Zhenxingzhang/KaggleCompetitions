# python 3.5.0
#import dependencies
import pandas as pd
from sklearn import tree,cross_validation
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


def one_hot_dataframe(data, column_name):
    dummies_data = pd.get_dummies(data[column_name])
    dummies_data.index = data.index
    dummies_data.columns = ["{}_{}".format(column_name, idx) for idx in dummies_data.columns]

    return dummies_data

train_data = pd.read_csv("../Data/train.csv")
test_data = pd.read_csv("../Data/test.csv")

# Data munging
train_data["Gender"] = train_data.Sex.map({"male": 0, "female": 1})
test_data["Gender"] = test_data.Sex.map({"male": 0, "female": 1})

train_data["Age"].fillna(train_data["Age"].mean(), inplace = True)

train_data["FamilyNo"] = train_data["SibSp"] + train_data["Parch"]

import re

combi_df = train_data.append(test_data, ignore_index=True)
# train_data["Name"].apply(lambda x : re.findall(r"[\w']+", x)[1]).value_counts()

combi_df["Title"] = combi_df["Name"].apply(lambda x: re.findall(r",[\w\s]+.", x)[0][2:-1].strip())
combi_df.ix[combi_df.Title.isin(['Capt', 'Don', 'Major', 'Sir', 'Col']), "Title"] = "Sir"
combi_df.ix[combi_df.Title.isin(['Dona', 'Lady', 'the Countess', 'Jonkheer', 'Ms', 'Mlle', 'Mme']), "Title"] = "Lady"
train_data["Title"] = combi_df["Title"][0:train_data.shape[0]]

# Data mungging, convert all string to integer, since sklearn only accept integer value.
train_data["Embarked"].fillna("Q", inplace= True )


train_data = pd.concat([train_data, one_hot_dataframe(train_data, "Embarked")], axis=1)
train_data = pd.concat([train_data, one_hot_dataframe(train_data, "Pclass")], axis=1)
train_data = pd.concat([train_data, one_hot_dataframe(train_data, "Title")], axis=1)


# Training the model
# we want to build an simple decision tree on the Gender feature
# to prove that gender will be picked for the first node
class_names = ["Dead", "Survived"]
feature_columns = ["Gender", "Pclass", "Fare", "Age", "SibSp", "Parch", "Embarked_S", "Embarked_C","Embarked_Q"]
feature_columns = ["Gender","Title_Mr","Title_Mrs","Title_Sir", "Title_Miss",
                   "Title_Master", "Title_Lady", "Pclass_1", "Pclass_2", "Pclass_3",
                   "Title_Sir", "Title_Dr", "Title_Rev",
                   "Fare", "Age", "FamilyNo", "Embarked_S", "Embarked_C","Embarked_Q"]
train_labels = train_data.Survived

# depths = np.arange(3, 20)
#
# train_scores = list()
# cv_scores = list()
# for depth in depths:
#     clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, criterion="gini")
#     cv_score = cross_validation.cross_val_score(clf, train_data[feature_columns], train_labels)
#     cv_scores.append(cv_score.mean())
#
#     clf.fit(train_data[feature_columns], train_labels)
#     train_scores.append(clf.score(train_data[feature_columns], train_labels))
#
# plt.plot(depths, train_scores, "r-")
# plt.plot(depths, cv_scores, "b--")
#
# plt.show()


# Set the parameters by cross-validation
tuned_parameters = [{'criterion': ["gini", "entropy"],
                     "max_depth": [3, 4, 5, 6, 8],
                     'min_samples_leaf': [5, 10, 15, 20],
                     'min_samples_split':[5, 10, 15]}]


clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=10)

clf.fit(train_data[feature_columns], train_labels)

print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)
# for params, mean_score, scores in clf.grid_scores_:
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean_score, scores.std() * 2, params))
# print()

# clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, min_samples_split=15, criterion="entropy")
clf = tree.DecisionTreeClassifier(max_depth=clf.best_params_['max_depth'],
                                  min_samples_leaf=clf.best_params_['min_samples_leaf'],
                                  min_samples_split=clf.best_params_['min_samples_split'],
                                  criterion=clf.best_params_['criterion'])
clf.fit(train_data[feature_columns], train_labels)

"""
1) Use pydotplus if you are using python 3+
2) Change the last line to pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png") as your variable name is 'dotfile' and not 'dot_data'
P.S - reinstall graphviz after you install pydotplus
"""
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_columns, class_names=class_names,
                     filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("dt_one_depth.pdf")
