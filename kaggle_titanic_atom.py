
import numpy as np
import pandas as pd

# importing the Datasets.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Checking the Datasets
train.head()
test.head()

# Checking the shape of Datasets
train.shape
test.shape

# Selecting columns that will be used in Predicting.
train = train[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
train.head()

test = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
test.head()

# Checking the Datasets for missing values.
train.info()

# Filling the missing values of Age column with its mean.
train_mean = train[['Age']].mean()
train['Age'] = train['Age'].fillna(train_mean)
train.info()

# Doing the same with Test Datasets
test_mean = test['Age'].mean()
test['Age'] = test['Age'].fillna(test_mean)
test.isnull().any()

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test.info()

# Converting Sex from catergorical to numerical
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train.head()

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test.head()


# Feature Engineering: Splitting Pclass in 1st class, 2nd Class, and 3rd Class
def class_sep(data):
    print(data.values())

class_sep(train['Sex'])
train['1st'] = train['1st']

import matplotlib.pyplot as plt
plt.hist(data, bins=50)

# Splitting the Train Datasets
X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the Dataset using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=20, max_leaf_nodes=20, max_features=7, min_samples_split=10)
rfc.fit(X_train, y_train)

# Checking the Accuracy Score of the Model
from sklearn.metrics import accuracy_score
print('Training accuracy: ', accuracy_score(y_train, rfc.predict(X_train)))
print('Validation accuracy: ', accuracy_score(y_test, rfc.predict(X_test)))

# Predicting for Test Dataset
prediciton = rfc.predict(test)
test['Survived'] = prediciton
test.head()

# Creating file for Submission
test = test[['PassengerId', 'Survived']]
test.head()
test.to_csv('atom_res.csv', index=False)
