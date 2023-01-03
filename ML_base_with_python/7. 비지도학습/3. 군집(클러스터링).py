import pandas as pd
## 1. 비계층적 군집분석
# KMeans 라이브러리
from sklearn.cluster import KMeans

# 데이터 불러오기
from sklearn.datasets import load_wine
dataset = load_wine()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df = data.copy()

# KMeans 선언 및 학습
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

# KMeans 예측
pred = kmeans.predict(df)

# 실제값과 예측값 비교
df['target'] = dataset.target
df['pred'] = pred

# PCA
from sklearn.decomposition import PCA
df = data.copy()
pca = PCA(n_components=2)
df = pca.fit_transform(df)
df = pd.DataFrame(df)
df.head()

# Kmeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)
pred = kmeans.predict(df)
df['kmeans'] = pred


# 각 군집의 중심점
centers = kmeans.cluster_centers_

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=df['kmeans'])
plt.scatter(centers[:,0], centers[:,1])

# 군집 수 찾기
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=2022)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)






## 2. 계층적 군집분석
# 덴드로그램
from scipy.cluster.hierarchy import dendrogram, ward

# PCA
df = data.copy()
pca = PCA(n_components=2)
df = pca.fit_transform(df)
df = pd.DataFrame(df)

# 샘플링
kdata = df[:3].copy()
kdata = kdata.append(df[70:73])
kdata = kdata.append(df[160:163])

# 덴드로그램
linkage = ward(kdata)
dendrogram(linkage)