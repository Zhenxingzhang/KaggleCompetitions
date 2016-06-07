#import dependencies
import pandas as pd
from sklearn import tree

train_data = pd.read_csv("../Data/train.csv")
test_data = pd.read_csv("../Data/test.csv")

train_labels = train_data.Survived

train_data["Gender"] = train_data.Sex.map({"male": 0, "female": 1})

print train_data.Gender, train_labels

# we want to build an simple decision tree on the Gender feature
# to prove that gender will be picked for the first node
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(train_data["Gender"], train_labels)
