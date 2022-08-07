---
layout: post
title: "[ML]week2-scikit learn"
date: 2022-08-07 22:37
categories: ML
---
<h2>아이리스 품종 예측하기</h2>
아래와 같은 코드로 아이리스 예측을 위한 데이터 셋을 로딩할 수 있다.


```python
from sklearn.datasets import load_iris
import pandas as pd

# 아이리스 데이터 셋 로딩
iris = load_iris()
```
아이리스를 예측하기 위한 피쳐 값, 피쳐 이름, 레이블 값, 레이블 이름 등은 아래와 같이 정리돼 있다.


```python
# 피쳐로만 된 ndarray
iris_data = iris.data

# 피처 이름
iris_feature_name = iris.feature_names

# 레이블 값 ndarray
iris_label = iris.target

# 레이블 이름
iris_label_name = iris.target_names
```
이를 데이터 프레임 함수를 이용하여 판다스의 데이터 프레임 형식으로 변환한다. 이 때 파라미터 중 data에는 피처 값, columns에는 피처의 이름을 입력한다. 그리고 데이터 프레임에 레이블의 값도 추가해준다.


```python
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
```
여기서 학습 데이터 셋으로 학습만 하게 된다면 얼마나 정확한지 알 수가 없다. 이미 학습한 데이터로 테스트를 하게 된다면 정확도가 100%가 나오기 때문이다. 따라서 학습용 데이터 셋도 학습용과 테스트용으로 분리해서 학습, 테스트를 진행한다. 이 때 쓰는 함수가 train_test_split 함수이다.


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
```
여기서 x_train, x_test는 각각 학습과 테스트에 사용할 피처 값이고, y_train, y_test는 각각 학습과 테스트에 사용할 레이블 값이다. 이 함수의 파라미터 중 test_size는 테스트용 데이터 셋에 얼만큼을 할당할지 정하는 파라미터이고 위에서는 0.2로 두었으니 20%를 할당한다는 의미이다.<br>
학습을 수행하기 위해서는 분류기 객체가 필요하다. 이번 실습에서는 decision tree 분류기를 사용했다.


```python
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
dt_clf.fit(x_train, y_train)
```
이제 predict 함수를 사용하여 예측을 할 수 있는데 정확도를 판단하기 위해서는 아래와 같은 코드를 사용하면 된다.


```python
from sklearn.metrics import accuracy_score
print("예상 정확도: {0:.4f}".format(accuracy_score(y_test, dt_clf.predict(x_test)))
```
<br>
<h2>Model Selection</h2>
<h3>교차 검증</h3>
<h4>K 폴드</h4>
K 폴드 방식은 K 개의 데이터 폴드 세트를 만들어서 K 번만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법이다. 아래는 그 코드이다.


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])

n_iter = 0

# KFold객체의 split( ) 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환  
for train_index, test_index  in kfold.split(features): 
    # kfold.split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n_iter += 1
    
    # 반복 시 마다 정확도 측정 
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    
    cv_accuracy.append(accuracy)
    
# 개별 iteration별 정확도를 합하여 평균 정확도 계산 
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy)) 
```
위의 코드에서 features도 ndarray이고 for문에서의 train_index, test_index도 ndarray이다. 아마 kfold를 split하면 해당 원소를 반환하는게 아니고 인덱스를 반환해서 fancy indexing을 하는 모양이다. 이 부분을 제외하면 나머지는 아이리스 품종 예측과 유사하다.<br>
<h4>Stratified K 폴드</h4>
Stratified K 폴드는 불균형한 분포도를 가진 레이블 데이터 집합을 위한 K 폴드 방식이다. 밑은 그 예시이다.


```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()
```
result


```console
0    50
1    50
2    50
Name: label, dtype: int64
```
code


```python
kfold = KFold(n_splits=3)
# kfold.split(X)는 폴드 세트를 3번 반복할 때마다 달라지는 학습/테스트 용 데이터 로우 인덱스 번호 반환. 
n_iter =0
for train_index, test_index  in kfold.split(iris_df):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
```
result


```console
## 교차 검증: 1
학습 레이블 데이터 분포:
 1    50
2    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 0    50
Name: label, dtype: int64
## 교차 검증: 2
학습 레이블 데이터 분포:
 0    50
2    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 1    50
Name: label, dtype: int64
## 교차 검증: 3
학습 레이블 데이터 분포:
 0    50
1    50
Name: label, dtype: int64
검증 레이블 데이터 분포:
 2    50
Name: label, dtype: int64
```
위와 같이 교차 검증 1에서 0인 레이블을 하나도 가지지 않고 학습을 한 다음 결과를 예측하면 당연히 정확도가 매우 떨어지는 것은 자명하다. 그래서 Stratified K 폴드를 사용하는데, 이는 위의 코드에서 KFold를 StratifiedKFold로만 바꿔주면 된다.


```python
dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy=[]

# StratifiedKFold의 split( ) 호출시 반드시 레이블 데이터 셋도 추가 입력 필요  
for train_index, test_index  in skfold.split(features, label):
    # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)

    # 반복 시 마다 정확도 측정 
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
# 교차 검증별 정확도 및 평균 정확도 계산 
print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy)) 
```
<br>
<h4>cross_val_score</h4>
이 함수는 Stratified K 폴드 방식을 간편하게 할 수 있게 해주는 함수이다. 코드는 아래와 같다. 


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도(accuracy) , 교차 검증 세트는 3개 
scores = cross_val_score(dt_clf , data , label , scoring='accuracy',cv=3)
#print(scores, type(scores))
print('교차 검증별 정확도:',np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))
```
<br>
<h4>GridSearchCV</h4>
이 방식은 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에 할 수 있는 방식이다. 그런데 방식 이해를 아직 잘 하지는 못해서 다음에 추가하겠다. TODO<br><br>
<h2>데이터 전처리(Data Preprocessing)</h2>
<h3>데이터 인코딩</h3>
<h4>레이블 인코딩</h4>
머신 러닝에서는 레이블의 값이 무조건 숫자여야 한다. 그런데 레이블의 값이 숫자로 들어오지 않는 경우가 있는데 이럴 때에는 레이블의 값들을 리스트에 넣고 각각의 값들을 리스트의 인덱스로 치환하는 방식을 택하는데 이러한 방식을 레이블 인코딩이라고 한다. 코드는 아래와 같다.


```python
from sklearn.preprocessing import LabelEncoder
items=['TV', '냉장고', '전자렌지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print(labels)
```
result


```console
[0 1 4 5 3 3 2 2]
```
보면 TV가 0, 냉장고가 1, 이런 식으로 각각에 대응되는 인덱스 번호가 생겼다. 그리고 여기서는 레이블을 인코딩 하는 과정에서 fit을 쓰고 transform 함수를 사용하여 labels을 반환한 것을 볼 수 있다.<br>
<h4>원 핫 인코딩</h4>
레이블 인코딩에서는 각각의 레이블의 값들이 서로 다른 인덱스 값을 가져 1차원 배열로 레이블을 뽑을 수 있었다. 원 핫 인코딩은 위와 유사하지만 조금은 다른데 레이블의 값이 2차원 배열이고 각각의 인덱스를 반환하는 것이 아닌 각각의 인덱스에 해당하는 값에 1이 들어간다. 예제를 보겠다.


```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items=['TV','냉장고','전자렌지','컴퓨터','선풍기','선풍기','믹서','믹서']

# 2차원 ndarray로 변환합니다. 
items = np.array(items).reshape(-1, 1)

# 원-핫 인코딩을 적용합니다. 
oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels = oh_encoder.transform(items)

# OneHotEncoder로 변환한 결과는 희소행렬(Sparse Matrix)이므로 toarray()를 이용하여 밀집 행렬(Dense Matrix)로 변환. 
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)
```
result


```console
원-핫 인코딩 데이터
[[1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]]
원-핫 인코딩 데이터 차원
(8, 6)
```
