import pandas as pd
import numpy as np
df = pd.DataFrame({
    'Unnamed: 0': [0,1,2,3,4,5,6,'new',10],
    '메뉴': ['황금후라이드','승일양념치킨','간장치킨','마늘치킨','파닭','닭강정','양념반후라이드반','[인기]아이펠치킨','[베스트]풀잎치킨'],
    '가격': [12000,13000,14000,14000,14000,15000,13000,16000,9900],
    '호수' : ['10호','10호','9호','9호','11호','12호','10호','11호','10호'],
    '칼로리' : [1000.0,1400.0,1600.0,1800.0,1300.0,1500.0,1300.0,1200.0,np.nan],
    '할인율' : [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,np.nan],
    '할인가' : [9600.0,10400.0,11200.0,11200.0,11200.0,12000.0,10400.0,8000.0,np.nan],
    '원산지' : ['국내산',np.nan,np.nan,np.nan,'브라질','브라질',np.nan,'국내산',np.nan]
})

# [Tip] 데이터프레임 복사
data = df.copy()
data

# 데이터 삭제 (컬럼(열): Unnamed: 0') drop(axis=1)
# axis=1:열방향(컬럼) / axis=0:행방향
data = data.drop('Unnamed: 0', axis=1)
data

# 데이터 삭제 (컬럼: Unnamed: 0') drop(columns=[])
data = df.copy()
data = data.drop(columns=['Unnamed: 0'])
data

# 데이터 복사(data -> df)
df = data.copy()
df




# 행 삭제 (파닭)
data = data.drop(4,axis=0)
data

# 행 삭제 (양념반후라이드반) inplace=True
data.drop(6,axis=0, inplace=True)
data

# 행 삭제 (조건: 가격 >= 14000)
cond = data['가격'] >= 14000
data[cond].index

# 행 삭제 (해당 인덱스 명)
data.drop(data[cond].index, axis=0, inplace=True)
data