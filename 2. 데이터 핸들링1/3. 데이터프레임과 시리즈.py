import pandas as pd
import numpy as np

# 시리즈 만들기(문자열)     메뉴 : "후라이드", "양념치킨", "양반후반"
menu_list = ["후라이드", "양념치킨", "양반후반"]
menu = pd.Series(menu_list)

# 시리즈 만들기(정수형)     가격 : 12000, 13000, 13000
price_list = [12000, 13000, 13000]
price = pd.Series(price_list)

# 데이터프레임 만들기 pd.DataFrame({"컬럼명":데이터})
data = {
    "메뉴":['후라이드', '양념치킨', '양반후반'],
    "가격":[12000, 13000, 13000],
    "호수":['10호', '10호', '9호']
}
df = pd.DataFrame(data)