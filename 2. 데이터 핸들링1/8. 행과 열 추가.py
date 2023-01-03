import pandas as pd
df = pd.read_csv('modudak.csv')

### 1. 열 (column) 추가
# 할인율 컬럼 추가 (값은 0.2)
df['할인율'] = 0.2

# 할인가 컬럼 추가
df['할인가'] = df['가격'] * (1-df['할인율'])

# 결측값으로 추가, 원산지 컬럼을 만들고 결측값(NaN)으로 대입
import numpy as np
df['원산지'] = np.nan
df




### 2. 행(row) 추가
# 리스트 형태로 데이터(행) 추가
aiffel = ['아이펠치킨', 16000, '11호', 1200, 0.5, 8000, '국내산']
df.loc['new'] = aiffel

# 딕셔너리 형태로 데이터(행) 추가
flip = {'메뉴':'풀잎치킨', '가격':10000, '호수':'10호'}
df.loc[10] = flip
df