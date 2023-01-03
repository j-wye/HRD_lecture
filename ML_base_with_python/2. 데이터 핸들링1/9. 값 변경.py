import pandas as pd
df = pd.read_csv('modudak.csv')

# 후라이드의 원산지는 '국내산' 값 추가(loc 활용)
df.loc[0, "원산지"] = '국내산'

# 파닭과 닭강정의 원산지는 '브라질' 값 추가 (iloc 활용)
df.iloc[4, -1] = '브라질'
df.iloc[5, -1] = '브라질'

# replace 활용 (양반후반 -> 양념반후라이드반)   =>  데이터값 변환
df = df.replace('양반후반', '양념반후라이드반')

# replace 활용 (아이펠치킨 -> [인기]아이펠치킨, 풀잎치킨 -> [베스트]풀잎치킨)   => 두개 이상을 동시에 변환
df = df.replace('아이펠치킨','[인기]아이펠치킨').replace('풀잎치킨','[베스트]풀잎치킨')

# replace, 딕셔너리 활용 (후라이드 -> 황금후라이드, 양념치킨 -> 승일양념치킨)
d = {'후라이드':'황금후라이드','양념치킨':'승일양념치킨'}
df = df.replace(d)

# 숫자 변경 : 10000 -> 9900 
df = df.replace(10000, 9900)

# new_modudak.csv로 저장
df.to_csv('new_modudak.csv')