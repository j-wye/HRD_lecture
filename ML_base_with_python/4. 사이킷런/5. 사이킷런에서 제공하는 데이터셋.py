import pandas as pd

# 1. 유방암 데이터
# 사이킷런 데이터 불러오기
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

# 피처 이름 확인
dataset.feature_names

# 타겟 확인
dataset.target

# 데이터 확인
dataset.data[:2]

# 데이터프레임으로 만들기
cancer_df = pd.DataFrame(data=dataset.data , columns=dataset.feature_names)

# 타겟 추가하기
cancer_df['target'] = dataset.target

# 데이터셋 확인 (load_*)
import sklearn.datasets
sklearn.datasets.__all__





# 2. 당뇨병 데이터
# 사이킷런 데이터 불러오기
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# 데이터프레임으로 만들기
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# 타겟 추가하기
diabetes_df['target'] = diabetes.target