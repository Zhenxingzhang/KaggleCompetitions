__author__ = 'zhenxing'

import numpy as np
import pandas as pd

titanic_df = pd.read_csv("../Data/train.csv")

print titanic_df.head()

print("----------------------------")
titanic_df.info()
# print train

titanic_df["Sex"].value_counts()