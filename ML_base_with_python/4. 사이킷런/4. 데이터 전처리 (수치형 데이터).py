import pandas as pd
data = pd.read_csv('final_modudak.csv')
df = data[['가격', '호수', '칼로리', '원산지', '살찔까요']].copy()
df = pd.get_dummies(df)





# StandardScaler(표준화)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(df[['가격']])

# StandardScaler 여러 변수에 적용
cols = ['가격', '호수', '칼로리']
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])






# MinMaxScaler(정규화)
from sklearn.preprocessing import MinMaxScaler
cols = ['가격', '호수', '칼로리']
scaler = MinMaxScaler()
df[cols] = scaler.fit_transform(df[cols])