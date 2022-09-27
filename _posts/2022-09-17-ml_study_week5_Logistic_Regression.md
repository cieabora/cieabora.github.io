---
layout: post
title: "[ML]week5-Logistic Regression"
date: 2022-09-17 23:49
categories: ML
---
<h1>
  로지스틱 회귀(Logistic Regression)
</h1>
<h2>
  로지스틱 회귀 개요
</h2>
로지스틱 회귀는 선형 회귀 방식을 분류에 적용한 알고리즘이다. 즉, 로지스틱 회귀는 분류에 사용된다. <br>
로지스틱 회귀가 선형 회귀와 다른 점은 선형 함수의 회귀 최적선을 찾는 것이 아니라, 시그모이드(Sigmoid) 함수의 최적선을 찾고 이 시그모이드 함수의 반환 값을 확률로 간주해 확률에 따라 분류를 결정한다는 것이다. 
<h2>
  로지스틱 회귀 예측
</h2>
로지스틱 회귀는 주로 이진 분류에 사용된다. 로지스틱 회귀에서 예측 값은 임계값에 대한 예측 확률을 의미한다. 로지스틱 회귀의 예측 확률은 시그모이드 함수의 출력 값으로 계산된다.<br>
단순 선형 회귀: y = w1x + w0 가 있다고 할 때,<br>
로지스틱 회귀는 0과 1을 예측하기에 단순 회귀식에 적용할 수는 없다. 하지만, Odds(성공확률/실패확률)을 통해 선형 회귀식에 확률을 적용한다.<br>
Odds(p)=p / (1 - p) <br>
하지만 확률 p의 범위가 0 ~ 1 사이이고, 선형 회귀의 반환값인 -무한대 ~ +무한대 값에 대응하기 위해서는 로그 변환을 수행하고 아래와 같이 선형 회귀를 적용한다.<br>
Log(Odds(p)) = w1x + w0<br>
해당 식을 데이터 값 x의 확률 p로 정리하면 아래와 같다.<br>
p(x) = 1 / (1 + e^(-(w1x + w0)))<br>
로지스틱 회귀는 학습을 통해 시그모이드 함수의 w를 최적화하여 예측하는 것이다.
<h2>
  로지스틱 회귀 특징
</h2>
로지스틱 회귀는 가볍고, 빠르며, 이진 분류 예측 성능도 뛰어나다. 특히 희소한 데이터 세트 분류에서 성능이 좋아서 텍스트 분류에 자주 사용된다.
<h2>
  로지스틱 회귀 사이킷런
</h2>
사이킷런은 로지스틱 회귀를 LogisticRegression 클래스로 구현한다.<br>
LogisticRegression의 주요 하이퍼 파라미터로 penalty, C, solver 가 있다. penalty는 규제 유형을 설정하며 'l2'로 설정 시 L2 규제를, 'l1'으로 설정 시 L1 규제를 뜻한다. C는 규제 강도를 조절하는 alpha 값의 역수이다. 즉 C = 1 / alpha 이다. C 값이 작을 수록 규제 강도가 크다.<br>
solver는 회귀 계수 최적화를 위한 다양한 최적화 방식이다.
<h2>
  로지스틱 회귀 코드
</h2>
code:


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# StandardScaler( )로 평균이 0, 분산 1로 데이터 분포도 변환
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train , X_test, y_train , y_test = train_test_split(data_scaled, cancer.target, test_size=0.3, random_state=0)

from sklearn.metrics import accuracy_score, roc_auc_score

# 로지스틱 회귀를 이용하여 학습 및 예측 수행. 
# solver인자값을 생성자로 입력하지 않으면 solver='lbfgs'  
lr_clf = LogisticRegression() # solver='lbfgs'
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)

# accuracy와 roc_auc 측정
print('accuracy: {0:.3f}, roc_auc:{1:.3f}'.format(accuracy_score(y_test, lr_preds),
                                                 roc_auc_score(y_test , lr_preds)))

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
# 여러개의 solver값 별로 LogisticRegression 학습 후 성능 평가
for solver in solvers:
    lr_clf = LogisticRegression(solver=solver, max_iter=600)
    lr_clf.fit(X_train, y_train)
    lr_preds = lr_clf.predict(X_test)

    # accuracy와 roc_auc 측정
    print('solver:{0}, accuracy: {1:.3f}, roc_auc:{2:.3f}'.format(solver, 
                                                                  accuracy_score(y_test, lr_preds),
                                                                  roc_auc_score(y_test , lr_preds)))                              

from sklearn.model_selection import GridSearchCV

params={'solver':['liblinear', 'lbfgs'],
        'penalty':['l2', 'l1'],
        'C':[0.01, 0.1, 1, 1, 5, 10]}

lr_clf = LogisticRegression()

grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3 )
grid_clf.fit(data_scaled, cancer.target)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, 
                                                  grid_clf.best_score_))
```
