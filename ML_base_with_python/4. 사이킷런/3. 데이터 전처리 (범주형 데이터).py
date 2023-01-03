import pandas as pd
data = pd.read_csv('final_modudak.csv')
df = data[['가격', '호수', '칼로리', '원산지', '살찔까요']].copy()

# type확인
df.info()


# 레이블(label) 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['원산지'])
le.transform(df['원산지'])

# fit_transform
# fit과 transform을 한번에!
le.fit_transform(df['원산지'])

# 데이터 변환
df['원산지'] = le.fit_transform(df['원산지'])

# '살찔까요' 레이블인코딩
le = LabelEncoder()
df['살찔까요'] = le.fit_transform(df['살찔까요'])

# 활용할 데이터 선택 - 원본 데이터 불러오기
df = data[['가격', '호수', '칼로리', '원산지', '살찔까요']].copy()

# 데이터가 object 타입인 column 선택 ['원산지', '살찔까요']

# 방법1
cols = ['원산지', '살찔까요']

# 방법2
cols = df.select_dtypes(include='object').columns

# 한 번에 레이블 인코딩
from sklearn.preprocessing import LabelEncoder

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])





# 원핫인코딩

# 활용할 데이터 선택
df = data[['가격', '호수', '칼로리', '원산지', '살찔까요']].copy()


# 데이터프레임을 입력받기 때문에 대괄호를 2번 써야 합니다. (df[['원산지']])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
cat = ohe.fit_transform(df[['원산지']])

# 카테고리
ohe.categories_

# 피처(컬럼)이름과 카테고리
ohe.get_feature_names_out()

# 데이터프레임으로 변환
df_cat = pd.DataFrame(cat, columns=ohe.get_feature_names_out())

# 데이터 프레임 합치기
df = pd.concat([df,df_cat],axis=1)

# 기존 컬럼 삭제
df = df.drop(['원산지'], axis=1)





# 원핫 인코딩 (심화)

# 활용할 데이터 선택
df = data[['가격', '호수', '칼로리', '원산지', '살찔까요']].copy()

# 원핫인코딩
cols = df.select_dtypes(include='object').columns
ohe = OneHotEncoder(sparse=False)
cat = ohe.fit_transform(df[cols])
df_cat = pd.DataFrame(cat, columns=ohe.get_feature_names_out())

# 데이터프레임 합치기
df = pd.concat([df, df_cat], axis=1)

# 컬럼 삭제
df = df.drop(cols, axis=1)





# [TIP] 원핫인코딩: 판다스 활용
# 활용할 데이터 선택
df = data[['가격', '호수', '칼로리', '원산지', '살찔까요']].copy()

# 판다스를 활용한 원핫 인코딩
df = pd.get_dummies(df)