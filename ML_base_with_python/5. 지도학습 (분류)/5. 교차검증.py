import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
# 데이터셋 로드
def make_dataset2():
    iris = load_breast_cancer()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df.drop('target', axis=1), df['target']
X, y = make_dataset2()



# KFold
from sklearn.model_selection import KFold
model = DecisionTreeClassifier(random_state=0)

kfold = KFold(n_splits=5)
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(accuracy_score(y_test, pred))



# Stratified Kfold
from sklearn.model_selection import StratifiedKFold
model = DecisionTreeClassifier(random_state=0)

kfold = StratifiedKFold(n_splits=5)
for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(accuracy_score(y_test, pred))



# 사이킷런 교차검증
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=3)

# 평균 점수
scores.mean()

# 교차검증 Stratified Kfold
kfold = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=kfold)

# 평균 점수
scores.mean()