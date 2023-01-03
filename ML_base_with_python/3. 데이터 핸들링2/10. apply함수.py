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

# apply 예시 (칼로리가 1300 보다 큰 값과 작은 값)
def cal(x):
    if x >= 1300:
        return "yes"
    else:
        return "no"

df['칼로리'].apply(cal)

# apply 적용해서 새로운 컬럼 생성 (칼로리 컬럼 활용)
df['살찔까요'] = df['칼로리'].apply(cal)

# lambda, apply 활용
df['고민'] = df['칼로리'].apply(lambda x: '먹지말자' if x >= 1300 else '무조건먹자')

# df 저장 (final_modudak.csv)
df.to_csv('final_modudak.csv', index=False)

pd.read_csv('final_modudak.csv')