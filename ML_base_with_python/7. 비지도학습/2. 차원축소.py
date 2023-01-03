# 라이브러리 불러오기
import pandas as pd

# 와인 데이터 불러오기
from sklearn.datasets import load_wine
dataset = load_wine()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# 데이터 샘플 확인
print(data.shape)
data.head()






## 1. 스탠다드 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)
data[:3]

# 데이터 복사 (df변수)
df = data.copy()





## 2. PCA
from sklearn.decomposition import PCA
# PCA
pca = PCA(n_components=2)

# PCA 변환
df = pca.fit_transform(df)

# 데이터 크기 확인
df.shape

# 샘플 데이터 확인
df[:3]






## 3. PCA(2차원) 시각화
# 시각화 라이브러리
import matplotlib.pyplot as plt

# 데이터 프레임 변환
df = pd.DataFrame(df)
df.head()

# scatter
plt.scatter(df.iloc[:,0], df.iloc[:,1])

# scatter (c=target)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=dataset.target)

# 데이터 설명(%)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())






## 4. 지도학습 (PCA 전후 비교)
# PCA후 데이터로 머신러닝 (2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(random_state=2022)
cross_val_score(model, df, dataset.target, scoring='accuracy', cv=5).mean()

# PCA전 데이터로 머신러닝 (13)
cross_val_score(model, data, dataset.target, scoring='accuracy', cv=5).mean()






## 5. PCA(3차원) 시각화
# 새 데이터 복사
df = data.copy()

# pca (n_components=3)
pca = PCA(n_components=3)

# pca 변환
df = pca.fit_transform(df)

# 데이터 프레임 변환
df = pd.DataFrame(df)

# 시각화 (3차원)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.iloc[:,0], df.iloc[:,1],df.iloc[:,2], c=dataset.target)






## 6. 점진적 PCA
# 라이브러리 불러오기
from sklearn.decomposition import IncrementalPCA

# 데이터 복사
df = data.copy()

# 점진적 PCA
pca = IncrementalPCA(n_components=2, batch_size=16)

# 점진적 PCA 변환
df = pca.fit_transform(df)

# 시각화
df = pd.DataFrame(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=dataset.target)

# 데이터 설명(%)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())






## 7. 희소 PCA
# 라이브러리 불러오기
from sklearn.decomposition import SparsePCA

# 데이터 복사
df = data.copy()

# 희소 PCA
pca = SparsePCA(n_components=2, alpha=0.01)

# 희소 PCA 변환
df = pca.fit_transform(df)

# 시각화
df = pd.DataFrame(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=dataset.target)






## 8. 커널 PCA
# 라이브러리 불러오기
from sklearn.decomposition import KernelPCA

# 데이터 복사
df = data.copy()

# 커널 PCA
pca = KernelPCA(n_components=2)

# 커널 PCA 변환
df = pca.fit_transform(df)

# 시각화
df = pd.DataFrame(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=dataset.target)






## 9. SVD (Singular Value Decomposition) 특이값 분해
# 라이브러리 불러오기
from sklearn.decomposition import TruncatedSVD

# SVD적용
df = data.copy()
pca = TruncatedSVD(n_components=2)
df = pca.fit_transform(df)

# 시각화
df = pd.DataFrame(df)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=dataset.target)