---
layout: post
title: "[ML]week3-Prediction Evaluation"
date: 2022-08-13 06:39
categories: ML
---
<head>
<style>
.backslash {
  background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg"><line x1="0" y1="0" x2="100%" y2="100%" stroke="gray" /></svg>');
}
.backslash { text-align: left; }
.backslash div { text-align: left; }
table {
    border-collapse: collapse;
    border: 1.01px solid gray;
}  
th, td {
    border: 1px solid gray;
    padding: 10px;
    text-align: center;
}
</style>
</head>
<h3>정확도(Accuracy)와 오차 행렬(confusion matrix)</h3>
머신 러닝에서 학습을 기반으로 한 예측의 결과는 아래의 표와 같이 4가지로 분류할 수 있다.<br>
<table>
<tr>
<th class='backslash' style='width:140px'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;예측 값<br>실제 값</th>
<td>False</td>
<td>True</td>
</tr>
<tr>
<td>False</td>
<td>TN</td>
<td>FP</td>
</tr>
<tr>
<td>True</td>
<td>FN</td>
<td>TP</td>
</tr>
</table>
이를 행렬로 나타낸 것을 오차 행렬(confusion matrix)라고 하는데, 이는 아래의 코드로 확인할 수 있다.


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, clf.predict(X_test))
```
위의 지표들로 봤을 때 예측이 성공한 경우는 TP와 TN이다. 그리고 sckit-learn에서 말하는 정확도란, (TN + TP) / (TN + TP + FN + FP)이다. 즉, 전체 데이터 중에서 예측을 올바르게 한 데이터의 비율이다. <br>
그런데 이 정확도만 보고 판단을 하기에는 문제가 있는데, 만약 Positive에 대한 예측을 하나도 시행하지 않은 모델이 있다고 했을 때, Negative에 대한 예측을 올바르게 하면 정확도가 높게 나온다는 것이다. 그래서 정밀도(Precision)와 재현율(Recall)이라는 개념을 추가로 도입한다. <br><br>
<h3>정밀도(Precision)와 재현율(Recall)</h3>
정밀도는 TP / (TP + FP), 재현율은 TP / (TP + FN) 으로 정의된다. 즉, 정밀도는 True라고 예측한 결과 중에서의 정확도이고, 재현율은 실제로 True인 것들 중에서의 정확도이다.<br>
정확도, 정밀도, 재현율은 아래의 코드로 확인할 수 있다.


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
```
<h3>Precision/Recall Trade-Off</h3>
predict_proba 함수를 사용하면 각각의 데이터에 대해서 얼마나 0에 가까운지, 얼마나 1에 가까운 지를 볼 수 있다.


```python
pred_proba = clf.predict_proba(X_test)
```
위의 코드를 실행하면 각각의 데이터에 대해서 분류의 0에 얼마나 가까운 지, 1에 얼마나 가까운 지가 나오는데 이 값들이 threshold의 대소에 따라서 0, 1로 분류한다. 여기서 binarizer를 사용하면 threshold의 값을 바꿀 수 있는데 코드는 아래와 같다.


```python
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=theta)
print(binarizer.fit_transform(X))
```
위의 코드를 실행하면 설정된 threshold에 대해서 0인지 1인지를 분류한 ndarray를 출력한다. 여기서 분류 결정 threshold를 기반으로 예측값을 변환할 수 있는데 코드는 아래와 같다.


```python
from sklearn.preprocessing import Binarizer
custom_threshold = 0.5
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba)
```
여기서 중요한 점은 threshold 값을 낮추게 되면 False라고 예측하는 비율이 낮아져 TN과 FN의 비율이 낮아지고, threshold 값을 높이게 되면 True라고 예측하는 비율이 낮아져 FP와 TP의 비율이 낮아진다. 즉, threshold 값을 조정하면 정밀도와 재현율을 조정할 수 있는데 이 과정이 trade-off이다. <br>
predict_proba(X_test)[:, 1]의 키워드를 이용하면 1에 얼마나 근사한 지에 대한 값이 나온다. 이것과 precision_recall_curve 함수를 이용해 여러 임곗값을 대입하여 구한 정밀도, 재현율, 임곗값을 알 수 있는데 코드는 아래와 같다.


```python
from sklearn.metrics import precision_recall_curve
pred_proba_class1 = clf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
```
이러면 precisions, recalls, thresholds에 각각 정밀도, 재현율, 이 둘을 구하는 데 썼던 임곗값이 들어간다. 아래는 임곗값의 변경에 따른 정밀도-재현율 변화 곡선 그래프 그리는 코드이다.


```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )
```
<h3>F1 Score</h3>
F1 Score는 정밀도와 재현율의 조화평균이다. 정의는 다음과 같이 된다. <br>
P = Precision, R = Recall일 때, <br>
F1 = 2 * (1 / (1 / P) + (1 / R)) = 2 * (P * R) / (P + R)<br>
아래는 F1 score를 구하는 코드이다.


```python
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
```

