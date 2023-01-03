import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## 1. 라이브러리 및 데이터 불러오기
# 데이터 생성
from sklearn.datasets import load_breast_cancer

def make_dataset():
    iris = load_breast_cancer()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis = 1), df['target'], test_size = 0.5, random_state = 1004)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = make_dataset()
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 타겟 확인
y_train.value_counts()





## 2. 의사결정나무
# 의사결정나무
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifierrandom_state = 0
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)

# 의사결정나무 하이퍼파라미터
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    criterion = 'entropy',
    max_depth = 7,
    min_samples_split = 2,
    min_smaples_leaf = 2,
    random_state = 0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)