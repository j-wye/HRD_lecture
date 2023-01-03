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


# xgboost
from xgboost import XGBClassifier
model = XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)

# xgboost 하이퍼파라미터
model = XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss',
                      booster='gbtree',
                      objective='binary:logistic',
                      max_depth=5,
                      learning_rate=0.05,
                      n_estimators=500,
                      subsample=1,
                      colsample_bytree=1,
                      n_jobs=-1

                      # - booster(기본값 gbtree): 부스팅 알고리즘 (또는 dart, gblinear)
                      # - objective(기본값 binary:logistic): 이진분류 (다중분류: multi:softmax)
                      # - max_depth(기본값 6): 최대 한도 깊이
                      # - learning_rate(기본값 0.1): 학습률
                      # - n_estimators(기본값 100): 트리의 수
                      # - subsample(기본값 1): 훈련 샘플 개수의 비율
                      # - colsample_bytree(기본값 1): 특성 개수의 비율
                      # - n_jobs(기본값 1): 사용 코어 수 (-1: 모든 코어를 다 사용)
                      )
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)

# 조기종료
model = XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss',
                     learning_rate = 0.05,
                      n_estimators = 500)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10)
pred = model.predict(X_test)
accuracy_score(y_test, pred)

