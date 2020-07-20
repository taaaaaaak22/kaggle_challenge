import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.model_selection import train_test_split

# データ読み込み
data = pd.read_csv('../input/titanic/train.csv')

# 形式変換
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# 欠損補完
data['Age'] = data['Age'].fillna(data['Age'].median())

# 使わない列削除
data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# 乗船者のIDと生存ラベルを取り除いたデータと生存ラベルに分ける
X = data.drop(['Survived', 'PassengerId'], axis=1).values
y = data['Survived'].values

# 学習
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForest(n_estimators=100).fit(X_train, y_train)

# テスト
target = pd.read_csv('../input/titanic/test.csv')

target['Sex'] = target['Sex'].map({'female': 0, 'male': 1})
target['Age'] = target['Age'].fillna(target['Age'].median())
target['Fare'] = target['Fare'].fillna(target['Fare'].median())
target = target.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# 乗船者のIDと生存ラベルを取り除いたデータと生存ラベルに分ける
X_target = target.drop(['PassengerId'], axis=1).values
result = model.predict(X_target)

target['Survived'] = result
target[["PassengerId","Survived"]].to_csv("submission2.csv",index=False)