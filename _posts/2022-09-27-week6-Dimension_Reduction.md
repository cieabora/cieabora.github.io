---
layout: post
title: "[ML]week6-Dimension Reduction"
date: 2022-09-27 18:18
categories: ML
---

<h1>
  차원 축소(Dimension Reduction)
</h1>
<h2>
  차원 축소의 장점
</h2>
<ol>
  <li>학습 데이터 크기를 줄여서 학습 시간 절약</li>
  <li>불필요한 피처들을 줄여서 모델 성능 향상에 기여(주로 이미지 관련 데이터)</li>
  <li>다차원의 데이터를 3차원 이하의 차원 축소를 통해서 시각적으로 보다 쉽게 데이터 패턴 인지</li>
</ol>
<h2>
  피처 선택과 피처 추출
</h2>
일반적으로 차원 축소는 피처 선택과 피처 추출로 나눌 수 있다.<br>
피처 선택<br>
특정 피처에 종속성이 강한 불필요한 피처는 아예 제거하고, 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 것.<br>
피처 추출<br>
피처(특성) 추출은 기존 피처를 저차원의 중요 피처로 압축해서 추출하는 것이다. 이렇게 새롭게 추출된 중요 특성은 기존의 피처를 반영해 압축된 것이지만, 새로운 피처로 추출하는 것이다.
<h2>
  차원 축소의 의미
</h2>
차원 축소는 단순히 데이터의 압축을 의미하는 것이 아니다. 더 중요한 의미는 차원 축소를 통해 좀 더 데이터를 잘 설명할 수 있는 잠재적인 요소를 추출하는 데에 있다. (추천 엔진, 이미지 분류 및 변환, 문서 토픽 모델링)
<h2>
  PCA(Principal Component Analysis)의 이해
</h2>
고차원의 원본 데이터를 저 차원의 부분 공간으로 투영하여 데이터를 축소하는 기법.<br>
예를 들어, 10차원의 데이터를 2차원의 부분 공간으로 투영하여 데이터를 축소.<br>
PCA는 원본 데이터가 가지는 데이터 변동성을 가장 중요한 정보로 간주하며, 이 변동성에 기반한 원본 데이터 투영으로 차원 축소를 수행.<br>
<h2>
  PCA 변환과 수행 절차
</h2>
PCA 변환<br>
입력 데이터의 공분산 행렬이 고유 벡터와 고유 값으로 분해될 수 있으며, 이렇게 분해된 고유 벡터를 이용해 입력 데이터를 선형 변환하는 방식<br>
PCA 변환 수행 절차<br>
1. 입력 데이터 세트의 공분산 행렬을 생성.<br>
2. 공분산 행렬의 고유 벡터와 고유 값을 계산.<br>
3. 고유 값이 가장 큰 순으로 K개(PCA 변환 차수)만큼 고유 벡터를 추출.<br>
4. 고유 값이 가장 큰 순으로 추출된 고유 벡터를 이용해 새롭게 입력 데이터를 변환.
<h2>
  사이킷런 PCA
</h2>
사이킷런은 PCA를 위해 PCA 클래스를 제공한다.<br>
n_components: PCA 축의 개수. 즉, 변환 차원을 의미<br>
PCA를 적용하기 전에 입력 데이터의 개별 피처들을 스케일링해야 한다. PCA는 여러 피처들의 값을 연산해야 하므로 피처들의 스케일에 영향을 받는다. 따라서 여러 속성을 PCA로 압축하기 전에 각 피처들의 값을 동일한 스케일로 변환하는 것이 필요하다. 일반적으로 평균이 0, 분산이 1인 표준 정규 분포로 변환한다.<br>
PCA 변환이 완료된 사이킷런 PCA 객체는 전체 변동성에서 개별 PCA 컴포넌트별로 차지하는 변동성 비율을 explained_variance_ratio_ 속성으로 제공한다.<br>
<h2>
  PCA 붓꽃 데이터 실습 코드
</h2>
code:


```python
# 붓꽃 데이터로 PCA 변환을 위한 데이터 로딩 및 시각화 
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# 사이킷런 내장 데이터 셋 API 호출
iris = load_iris()

# 넘파이 데이터 셋을 Pandas DataFrame으로 변환
columns = ['sepal_length','sepal_width','petal_length','petal_width']
irisDF = pd.DataFrame(iris.data , columns=columns)
irisDF['target']=iris.target
irisDF.head(3)

#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

#setosa의 target 값은 0, versicolor는 1, virginica는 2. 각 target 별로 다른 shape으로 scatter plot 
for i, marker in enumerate(markers):
    x_axis_data = irisDF[irisDF['target']==i]['sepal_length']
    y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

from sklearn.preprocessing import StandardScaler

# Target 값을 제외한 모든 속성 값을 StandardScaler를 이용하여 표준 정규 분포를 가지는 값들로 변환
iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:, :-1])

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

#fit( )과 transform( ) 을 호출하여 PCA 변환 데이터 반환
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)

# PCA 환된 데이터의 컬럼명을 각각 pca_component_1, pca_component_2로 명명
pca_columns=['pca_component_1','pca_component_2']
irisDF_pca = pd.DataFrame(iris_pca, columns=pca_columns)
irisDF_pca['target']=iris.target
irisDF_pca.head(3)

# PCA로 차원 축소된 피처들로 데이터 산포도 시각화
#setosa를 세모, versicolor를 네모, virginica를 동그라미로 표시
markers=['^', 's', 'o']

#pca_component_1 을 x축, pc_component_2를 y축으로 scatter plot 수행. 
for i, marker in enumerate(markers):
    x_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_1']
    y_axis_data = irisDF_pca[irisDF_pca['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()

# 각 PCA Component별 변동성 비율
print(pca.explained_variance_ratio_)

# 원본 데이터와 PCA 변환된 데이터 기반에서 예측 성능 비교
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf = RandomForestClassifier(random_state=156)
scores = cross_val_score(rcf, iris.data, iris.target,scoring='accuracy',cv=3)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:', np.mean(scores))

pca_X = irisDF_pca[['pca_component_1', 'pca_component_2']]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring='accuracy', cv=3 )
print('PCA 변환 데이터 교차 검증 개별 정확도:',scores_pca)
print('PCA 변환 데이터 평균 정확도:', np.mean(scores_pca))
```

<h2>
  LDA(Linear Discriminant Analysis)의 이해
</h2>
LDA는 선형 판별 분석법으로 불리며, PCA와 매우 유사하다.<br>
LDA는 PCA와 유사하게 입력 데이터 세트를 저차원 공간에 투영해 차원을 축소하는 기법이지만, 중요한 차이는 LDA는 지도학습의 분류에서 사용하기 쉽도록 개별 클래스를 분별할 수 있는 기준을 최대한 유지하면서 차원을 축소한다. PCA는 입력 데이터의 변동성의 가장 큰 축을 찾았지만, LDA는 입력 데이터의 결정 값 클래스를 최대한으로 분리할 수 있는 축을 찾는다.<br>
LDA는 같은 클래스의 데이터는 최대한 근접해서, 다른 클래스의 데이터는 최대한 떨어뜨리는 축 매핑을 한다.
<h2>
  LDA 차원 축소 방식
</h2>
LDA는 특정 공간상에서 클래스 분리를 최대화하는 축을 찾기 위해 클래스 간 분산과 클래스 내부 분산의 비율을 최대화하는 방식으로 차원을 축소한다. 즉, 클래스 간 분산은 최대한 크게 가져가고, 클래스 내부의 분산은 최대한 작게 가져가는 방식이다.
