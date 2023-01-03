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

# 데이터프레임 복사
data = df.copy()
data

# 결측치가 있는 행 모두 삭제
data.dropna(axis=0)

# 결측치가 있는 컬럼 모두 삭제
data.dropna(axis=1)

# 컬럼별 결측치 확인
# df.isnull()   => True일때 결측치, False일때 값이 존재!!
df.isnull().sum()   # => column의 결측치의 개수 확인!!

# 결측값 채우기
# 원산지 -> '알수없음'으로 채우기
df['원산지'].fillna('알수없음')

# 원산지 최빈값으로 채우기
df['원산지'].fillna('국내산')

# 원산지 채우기 실행
df['원산지'] = df['원산지'].fillna('국내산')

# '칼로리'컬럼 결측치는 900으로 결측치 채움
df['칼로리'].fillna(900)

# '칼로리'컬럼 결측치는 최소값 결측치 채움
df['칼로리'].fillna(df['칼로리'].min())

# 칼로리 채우기 실행
df['칼로리'] = df['칼로리'].fillna(df['칼로리'].min())