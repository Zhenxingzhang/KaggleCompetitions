# python 3.5.0
#import dependencies
import pandas as pd
from sklearn import tree

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

combi_df["Title"] = combi_df["Name"].apply(lambda x: re.findall(r"[\w']+", x)[1])
combi_df.ix[combi_df.Title.isin(["Col", "Don", "Major", "Sir", "Capt"]), "Title"] = "Sir"
combi_df.ix[combi_df.Title.isin(["Mlle", "Mme"]), "Title"] = "Mlle"
combi_df.ix[combi_df.Title.isin(["Dona", "Jonkheer", "Lady", "the Countess", "Ms"]), "Title"] = "Lady"
combi_df.ix[~combi_df.Title.isin(["Mr", "Miss", "Mrs", "Master", "Sir", "Lady", "Mlle"]), "Title"] = "None"
train_data["Title"] = combi_df["Title"][0:train_data.shape[0]]
test_data["Title"] = combi_df["Title"][train_data.shape[0]:].reset_index(drop=True)


# Data mungging, convert all string to integer, since sklearn only accept integer value.
test_data["Fare"].fillna(train_data["Fare"].mean(), inplace=True)
train_data["Embarked"].fillna("Q", inplace= True )

train_data["Embarked_Int"] = train_data["Embarked"].map({"S": 1, "C":2, "Q":3})
test_data["Embarked_Int"] = test_data["Embarked"].map({"S": 1, "C":2, "Q":3})


train_data["Title_Int"] = train_data["Title"].map({"Mr": 1, "Miss":2, "Mrs":3, "None" :4, "Master":5, "Sir":6, "Mlle":7, "Lady":8})
test_data["Title_Int"] = test_data["Title"].map({"Mr": 1, "Miss":2, "Mrs":3, "None" :4, "Master":5, "Sir":6, "Mlle":7, "Lady":8})


# Training the model
# we want to build an simple decision tree on the Gender feature
# to prove that gender will be picked for the first node
class_names = ["Dead", "Survived"]
feature_columns = ["Gender", "Pclass", "Fare", "Age", "Embarked_Int", "Title_Int"]
train_labels = train_data.Survived

clf = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, criterion="entropy")
clf.fit(train_data[feature_columns], train_labels)


from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_columns, class_names=class_names,
                     filled=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("dt_one_depth.pdf")
