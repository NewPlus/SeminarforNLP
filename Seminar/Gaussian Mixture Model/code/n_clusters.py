import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
 
import warnings
warnings.filterwarnings('ignore')
 
np.random.seed(3)
sample_size=100
x1 = 3*np.random.rand(sample_size)
x2 = np.random.rand(sample_size)+1
x3 = 2*np.random.rand(sample_size)+2
 
y1 = 2*x1+2+np.random.randn(sample_size)
y2 = 4*x2+5+np.random.randn(sample_size)
y3 = 0.2*x3+np.random.randn(sample_size)
 
y = np.array(list(y1)+list(y2)+list(y3))
x = np.array(list(x1)+list(x2)+list(x3))
 
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.scatter(x, y, color='k')
plt.show()

X = np.empty((0,2), int)
for i, j in zip(x, y):
    X = np.append(X, np.array([[i, j]]), axis=0)

init_lists_3 = [[1.5,4], [3.5,0], [1.5, 11]]
init_lists_7 = [[1.5,4], [3.5,0], [1.5, 11], [1.5,4], [3.5,0], [1.5, 11], [1.5,4]]

## GMM_3 ##
from sklearn.mixture import GaussianMixture
 
init_centers = np.array(init_lists_3) # 초기 평균값
# 클러스터 개수 3개, means_init으로 초기 평균 벡터 설정, 결과 재생산성을 위해 random_state 설정
gmm = GaussianMixture(n_components=3, means_init=init_centers, random_state=100)
gmm.fit(X) # GMM 클러스터링 수행
labels = gmm.predict(X) # 최종 클러스터 라벨링
 
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()

## GMM_7 ##
from sklearn.mixture import GaussianMixture
 
init_centers = np.array(init_lists_7) # 초기 평균값
# 클러스터 개수 7개, means_init으로 초기 평균 벡터 설정, 결과 재생산성을 위해 random_state 설정
gmm = GaussianMixture(n_components=7, means_init=init_centers, random_state=100)
gmm.fit(X) # GMM 클러스터링 수행
labels = gmm.predict(X) # 최종 클러스터 라벨링
 
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()