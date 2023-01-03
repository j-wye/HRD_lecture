# 데이터 생성: csv파일 (data.csv) 생성 
import pandas as pd
data = {
    "메뉴":['후라이드', '양념치킨', '간장치킨', '마늘치킨', '파닭', '닭강정', '양반후반'],
    "가격":[12000, 13000, 14000, 14000, 14000, 15000, 13000],
    "호수":['10호', '10호', '9호', '9호', '11호', '12호', '10호'],
    "칼로리":[1000, 1400, 1600, 1800, 1300, 1500, 1300],
}
data = pd.DataFrame(data)
data.to_csv('modudak.csv', index=False)

# 데이터 불러오기
pd.read_csv('modudak.csv')

# 데이터 불러와서 변수에 담기
df = pd.read_csv('modudak.csv')

# 데이터 샘플 확인: 앞에서 부터 n개 (기본 5개)
df.head()

# 데이터 샘플 확인: 앞에서 부터 2개
df.head(2)

# 데이터 샘플 확인: 뒤에서 부터 n개 (기본 5개)
df.tail()

# 데이터 샘플 확인: 뒤에서 부터 2개
df.tail(2)

# csv파일로 저장하기
df.to_csv('modudak2.csv', index=False)

# 저장된 csv확인
new_df = pd.read_csv('modudak2.csv')
new_df.head()