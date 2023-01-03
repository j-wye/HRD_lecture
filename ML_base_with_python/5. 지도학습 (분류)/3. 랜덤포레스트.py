import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

def make_dataset():
    iris = load_breast_cancer()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis = 1), df['target'], test_size = 0.5, random_state = 1004)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = make_dataset()


# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)


# 랜덤포레스트 하이퍼파라미터
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)