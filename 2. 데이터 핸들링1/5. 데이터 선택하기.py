import pandas as pd
df = pd.read_csv('modudak.csv')

### 1. 열(column) 선택
# 시리즈 선택 (메뉴)
df['메뉴']

# 데이터프레임 선택 (메뉴, 가격)
df[['메뉴', '가격']]

# type 확인
type(df[['메뉴', '가격']])

# 14000원 보다 큰 가격
df['가격'] >= 14000




### 2. 행(row) 선택
# 14000원 보다 큰 가격 데이터선택
cond = df['가격']  >= 14000
df[cond]

# 조건 2개 이상 일때 (AND)
# 가격 >= 14000
# 호수 == 9호
cond1 = df['가격'] >= 14000
cond2 = df['호수'] == '9호'
df[cond1 & cond2]

# 조건 2개 이상 일때 (or)
# 가격 >= 14000
# 호수 == 9호
cond1 = df['가격'] >= 14000
cond2 = df['호수'] == '9호'
df[cond1 | cond2]

# isin() 함수 활용
cond = df['호수'].isin(['9호','10호'])
df[cond]