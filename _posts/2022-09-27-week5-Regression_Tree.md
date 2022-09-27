---
layout: post
title: "[ML]week5-Regression Tree"
date: 2022-09-27 18:04
categories: ML
---
<h1>
  회귀 트리
</h1>
<h2>
  회귀 트리 개요
</h2>
사이킷런의 결정 트리 및 결정 트리 기반의 앙상블 알고리즘은 분류 뿐만 아니라 회귀도 가능하다. 이는 트리가 CART(Classification and Regression Tree)를 기반으로 만들어 졌기 때문이다. CART는 분류 뿐만 아니라 회귀도 가능한 트리 분할 알고리즘이다. <br>
CART 회귀 트리는 분류와 유사하게 분할을 하며, 최종 분할이 완료된 후에 각 분할 영역에 있는 데이터 결정값들의 평균 값으로 학습/예측한다.
<h2>
  회귀 트리 프로세스
</h2>
1. 기준에 따라 트리 분할<br>
2. 최종 분할된 영역에 있는 데이터들의 평균값들로 학습/예측
<h2>
  사이킷런의 회귀 트리 지원
</h2>
알고리즘, 회귀 Estimator 클래스, 분류 Estimator 클래스<br>
Decision Tree, DecisionTreeRegressor, DecisionTreeClassifier<br>
Gradient Boosting, GradientBoostingRegressor, GradientBoostingClassifier<br>
XGBoost, XGBRegressor, XGBClassifier<br>
LightBGM, LGBMRegressor, LGBMClassifier<br>
<h2>
  회귀 트리의 오버 피팅
</h2>
회귀 트리 역시 복잡한 트리 구조를 가질 경우 오버 피팅하기 쉬우므로 트리의 크기와 노드 개수의 제한 등의 방법을 통해 오버 피팅을 개선할 수 있다.
<h2>
  회귀 트리 실습 코드
</h2>
code:


```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  #사이킷런 1.2 부터는 보스턴 주택가격 데이터가 없어진다는 warning 메시지 출력 제거

# 보스턴 데이터 세트 로드
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF['PRICE'] = boston.target
y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1,inplace=False)

rf = RandomForestRegressor(random_state=0, n_estimators=1000)
neg_mse_scores = cross_val_score(rf, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 교차 검증의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 교차 검증의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))

def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
    rmse_scores  = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print('##### ',model.__class__.__name__ , ' #####')
    print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
xgb_reg = XGBRegressor(n_estimators=1000)
lgb_reg = LGBMRegressor(n_estimators=1000)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:  
    get_model_cv_prediction(model, X_data, y_target)
```
