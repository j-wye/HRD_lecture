import pandas as pd
# 이전 데이터들
# 유방암 데이터
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
cancer_df = pd.DataFrame(data=dataset.data , columns=dataset.feature_names)
cancer_df['target'] = dataset.target
# 당뇨병 데이터
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target





# 머신러닝 : 분류 (Classification)
## 1. 검증데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer_df.drop('target', axis=1), cancer_df['target'], test_size=0.3, random_state=1004)

# train 데이터 확인(X)
X_train.head()

# 학습 데이터 확인(y)
y_train.head()

# 데이터 크기
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## 2. 의사결정나무
# 머신러닝(분류)
from sklearn.tree import DecisionTreeClassifier

# 모델 선택
model = DecisionTreeClassifier()
# 학습
model.fit(X_train, y_train)
# 예측
pred = model.predict(X_test)

## 3. 정확도 accuracy_score(실제값, 예측값) => 높으면 높을수록 좋은 것
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)





# 머신러닝 : 회귀 (Regression)
## 1. 검증데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_df.drop('target', axis=1), diabetes_df['target'], test_size=0.3, random_state=1004)

# train 데이터 확인(X)
X_train.head()

# 학습 데이터 확인(y)
y_train.head()

# 데이터 크기
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## 2. 선형회귀 (LinearRegression)
# 머신러닝(회귀)
from sklearn.linear_model import LinearRegression

# 모델 선택
model = LinearRegression()
# 학습
model.fit(X_train, y_train)
# 예측
pred = model.predict(X_test)

## 3. 정확도 mean_squared_error(실제값, 예측값) => 오차값이 출력되서 낮으면 낮을수록 좋은것!!
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred)