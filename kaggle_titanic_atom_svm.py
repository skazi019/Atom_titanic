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
train = train[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
train.head()

test = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test.head()

# Checking the Datasets for missing values.
train.info()

# Filling the missing values of Age column with its mean.
train_mean = train[['Age']].mean()
train['Age'] = train[['Age']].fillna(train_mean)
train.info()

# Doing the same with Test Datasets
test_mean = test[['Age']].mean()
test['Age'] = test[['Age']].fillna(test_mean)
test.isnull().any()

test['Fare'] = test[['Fare']].fillna(test[['Fare']].mean())
test.info()

# Converting Sex from catergorical to numerical
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train.head()

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test.head()

train['Embarked'].unique()
train['Embarked'].isnull().value_counts()
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'].isnull().value_counts()
train['Embarked_S'] = train['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
train['Embarked_C'] = train['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
train['Embarked_Q'] = train['Embarked'].apply(lambda x: 1 if x == 'Q' else 0)
train.drop('Embarked', axis=1, inplace=True)
train.head()

test['Embarked_C'] = test['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
test['Embarked_S'] = test['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
test['Embarked_Q'] = test['Embarked'].apply(lambda x: 1 if x == 'Q' else 0)
test.drop('Embarked', axis=1, inplace=True)
test.head()

# Feature Engineering: Splitting Pclass in 1st class, 2nd Class, and 3rd Class
train['first_class'] = train['Pclass'].apply(lambda x : 1 if x == 1 else 0)
train['second_class'] = train['Pclass'].apply(lambda x : 1 if x == 2 else 0)
train['third_class'] = train['Pclass'].apply(lambda x : 1 if x == 3 else 0)
train.drop('Pclass', axis=1, inplace=True)
train.head()

test['first_class'] = test['Pclass'].apply(lambda x : 1 if x == 1 else 0)
test['second_class'] = test['Pclass'].apply(lambda x : 1 if x == 2 else 0)
test['third_class'] = test['Pclass'].apply(lambda x : 1 if x == 3 else 0)
test.drop('Pclass', axis=1, inplace=True)
test.head()


# Splitting the Train Datasets
X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Training the Dataset using RandomForestClassifier
from sklearn import svm
svm = svm.SVC()
svm.fit(X_train, y_train)

# Checking the Accuracy Score of the Model
from sklearn.metrics import accuracy_score
print('Training accuracy: ', accuracy_score(y_train, svm.predict(X_train)))
print('Validation accuracy: ', accuracy_score(y_test, svm.predict(X_test)))

# Predicting for Test Dataset
prediciton = svm.predict(test)
test['Survived'] = prediciton
test.head()

# Creating file for Submission
test = test[['PassengerId', 'Survived']]
test.head()
test.to_csv('atom_res.csv', index=False)
