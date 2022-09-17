---
layout: post
title: "[ML]week5-Linear Regression"
date: 2022-09-14 18:12
categories: ML
---
<h2>
  RSS
</h2>
오류 값의 제곱을 구해서 더하는 방식. 일반적으로 미분 등의 계산을 편리하게 하기 위해서 RSS 방식으로 오류 합을 구한다. 즉 에러의 제곱은 RSS.<br>
RSS는 회귀식의 독립변수 X, 종속변수 Y가 중심 변수가 아니라 w 변수(회귀 계수)가 중심 변수임을 인지하는 것이 매우 중요하다.(학습 데이터로 입력되는 독립변수와 종속변수는 RSS에서 모두 상수로 간주한다.)<br>
RSS == 비용 함수(회귀에서)

<h2>
  경사 하강법
</h2>
반복적으로 비용 함수의 반환 값, 즉 예측값과 실제 값의 차이가 작아지는 방향성을 가지고 W 파라미터를 지속해서 보정해 나간다.

<h3>
  경사 하강법 수행 프로세스
</h3>

Step 1. w1, w0를 임의의 값으로 설정하고 첫 비용 함수의 값을 계산한다.<br>
Step 2. w1과 w0를 각각 편미분한 값 * 학습률을 곱해 업데이트 한 후 다시 비용 함수의 값을 계산한다.<br>
Step 3. step2를 주어진 횟수만큼 반복한다.<br>

<h2>
  Linear Regression
</h2>
규제가 없음. 

<h3>
  선형 회귀의 다중 공선성 문제
</h3>
일반적으로 선형 회귀는 입력 피처의 독립성에 많은 영향을 받는다. 피처간의 상관관계가 매우 높은 경우 분산이 매우 커져서 오류에 매우 민감해진다. 이러한 현상을 다중 공선성(multi-collinearity) 문제라고 한다. 일반적으로 상관관계가 높은 피처가 많은 경우 독립적인 중요한 피처만 남기고 제거하거나 규제를 적용한다.
<h3>
  회귀 평가 지표
</h3>
<ol>
  <li>MAE(Mean Absolute Error): 실제 값과 예측 값의 차이를 절댓값으로 변환해 평균한 것.</li>
  <li>MSE(Mean Squared Error): 실제 값과 예측 값의 차이를 제곱해 평균</li>
  <li>RMSE(Root Mean Squared Error): MSE에 루트를 씌운 것</li>
  <li>RMSLE(Root Mean Squared Log Error): RMSE에 로그를 적용한 것. 결정값이 클 수록 오류 값도 커지기 때문에 일부 큰 오류값들로 인해 전체 오류값이 커지는 것을  막아준다.</li>
  <li>R^2: 분산 기반으로 예측 성능을 평가한다. 실제 값의 분산 대비 예측 값의 분산 비율을 지표로 하며, 1에 가까울수록 예측 정확도가 높다.</li>
</ol>
MAE에 비해서 RMSE는 큰 오류값에 상대적인 패널티를 더 부여한다.

<h3>
  사이킷런 회귀 평가 API
</h3>
<ol>
  <li>MAE: metrics.mean_absolute_error</li>
  <li>MSE: metrics.mean_squared_error</li>
  <li>RMSE: metrics.mean_squared_error를 그대로 사용하되 squared 파라미터를 False로 설정</li>
  <li>MSLE: metrics.mean_squared_log_error</li>
  <li>R^2: metrics.r2_score</li>
</ol>

<h3>
  Scroing 함수 적용 값
</h3>
<ol>
  <li>MAE: neg_mean_absolute_error</li>
  <li>MSE: neg_mean_squared_error</li>
  <li>RMSE: neg_root_mean_squared_error</li>
  <li>MSLE: neg_mean_squared_log_error</li>
  <li>R^2: r2</li>
</ol>

<h3>
  사이킷런 Scoring 함수에 회귀 평가 적용 시 유의 사항
</h3>
MAE의 사이킷런 scoring 파라미터는 음수를 가질 수 없다. scoring 함수를 음수로 반환하는 이유는 사이킷런의 Scoring 함수가 score 값이 클수록 좋은 평가 결과로 자동 평가하기 때문이다. 따라서 -1을 원래의 평가 지표 값에 곱해서 음수를 만들어 작은 오류 값이 더 큰 숫자로 인식하게 한다.

<h2> 사이킷런 보스턴 집 값 예측 실습 코드 </h2>
code:


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')  #사이킷런 1.2 부터는 보스턴 주택가격 데이터가 없어진다는 warning 메시지 출력 제거
%matplotlib inline

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)
bostonDF.head()

# 2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4x2개의 ax를 가짐.
fig, axs = plt.subplots(figsize=(16,8) , ncols=4 , nrows=2)
lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
for i , feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature , y='PRICE',data=bostonDF , ax=axs[row][col])
    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)

X_train , X_test , y_train , y_test = train_test_split(X_data , y_target ,test_size=0.3, random_state=156)

# Linear Regression OLS로 학습/예측/평가 수행. 
lr = LinearRegression()
lr.fit(X_train ,y_train )
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))

# 회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성. index가 컬럼명에 유의
coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns )
coeff.sort_values(ascending=False)

from sklearn.model_selection import cross_val_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)
lr = LinearRegression()

# cross_val_score( )로 5 Fold 셋으로 MSE 를 구한 뒤 이를 기반으로 다시  RMSE 구함. 
neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# cross_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수 
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))
```
