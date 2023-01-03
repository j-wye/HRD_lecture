import pandas as pd
df = pd.read_csv('modudak.csv')

### 1. set_index(keys=[컬럼명])
df = df.set_index(keys=['메뉴'])

# loc 슬라이싱
# 양념치킨에서 마늘치킨까지 호수에서 칼로리까지
df.loc["양념치킨":"마늘치킨", "호수":"칼로리"]

# iloc 슬라이싱
# 양념치킨에서 마늘치킨까지 호수에서 칼로리까지
df.iloc[1:4, 1:]




### 2. reset_index(기본값 drop=False)
df.reset_index()