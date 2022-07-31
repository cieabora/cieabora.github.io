---
layout: post
title: "[ML]week1"
date: 2022-07-31 23:40
categories: ML
---
<h2>ndarray</h2>
numpy는 선형대수를 구현하기 위해 만든 파이썬 라이브러리이다. array 함수로 ndarray 객체를 생성할 수 있다.


```python
import numpy as np
arr1 = np.array([1, 2, 3, 4], dtype = 'int32')
```
array 를 조작하는 함수들이 있는데 아래와 같다.
```python
arr2 = np.zeros((3, 2), dtype='int32')
arr3 = np.ones((3, 2), dtype='int32')
```
위와 같이 numpy 객체를 생성하면 arr2는 0만 있는 3 by 2 행렬이 생성되고, arr3는 1만 있는 3 by 2 행렬이 생성된다.<br> 
이미 만들어진 넘파이 객체의 모양을 바꾸는 reshape라는 멤버 함수가 존재한다. 이 함수의 첫번째 인자는 행, 두번째 인자로는 열이 들어가게 되는데 원래 넘파이 객체를 해당 크기의 행렬로 쪼갠 값을 리턴한다.
```python
arr4 = np.array([1, 2, 3, 4]).reshape(2, 2)
```
이렇게 하면 arr4에는 [[1, 2], [3, 4]] 꼴의 행렬이 생성된다. ndarray의 원형은 꼭 인자로 들어온 행과 열의 곱과 같아야 한다.<br>
그리고 인자로 -1이 들어오게 된다면 -1이 아닌 인자를 기준으로 채워 놓은 다음 -1에 나머지를 채우게 된다. 
```python
arr1 = np.array([1, 2, 3, 4, 5, 6])
arr5 = arr1.reshape(2, -1)
```
위와 같이 하면 arr5에는 2 by 3 행렬이 입력된다. 열벡터나 행벡터를 만들고 싶다면 위와 비슷한 방식으로 하면 된다.
```python
arr1 = np.array([1, 2, 3, 4])
row_vector = arr1.reshape(1, -1)
column_vector = arr1.reshape(-1, 1)
```
본디 행 벡터나 열 벡터는 1차원 벡터이지만 넘파이의 한계 때문에 열벡터는 2차원 벡터로 밖에 표현이 되지 않는다. <br><br>

<h2> fancy index, boolean index</h2>
fancy index는 정상적인 인덱스와 비슷한데 인덱싱을 하는 대괄호 안에 리스트를 넣는다. 그리고 리스트 안에는 원하는 추출을 원하는 원소들의 인덱스를 넣어서 여러 값을 뽑는다.
```python
arr1 = np.arange(start=1, stop=13)
print(arr1[[10, 4, 2]])
```
위와 같이 코드를 입력하면 array([11, 5, 3])이 출력된다. <br>
boolean index는 정말 유용하게 사용했던 인덱싱 기법인데 넘파이 객체에 boolean 연산을 하게 되면 해당 연산을 만족하는 인덱스에는 True, 그렇지 않은 인덱스에는 False가 들어가 있는 새로운 넘파이 객체가 생성된다. 그리고 이 넘파이 객체를 이용하여 인덱싱을 하게 되면 True에 해당하는 인덱스에 있는 원소만 추출, 수정이 가능하다. 
```python
bool_arr = arr1 % 2 == 0
print(bool_arr)
print(arr1[bool_arr])
```
위와 같은 코드를 실행하면 
```console
array([False,  True, False,  True, False,  True, False,  True, False,
        True, False,  True])
array([2, 4, 6, 8, 10, 12])
```
가 출력된다. 이 기법은 영상처리를 할 때 사용했는데 세그멘테이션 된 이미지의 색을 클래스화 시키기 위해서 한 픽셀씩 돌면 사진 한 장당 2 ~ 3 초의 시간이 소요 됐고 이런 사진을 2000장 처리했어야 했다. 그런데 이 boolean index를 이용해서 내가 원하는 색에 해당하는 픽셀들을 뽑고 수정하니까 한 장당 0.01초 정도로 시간이 줄어드는 마법을 경험했다.<br>
<h2>axis 기준</h2>
axis의 기준은 간단하다. n차원 행렬을 파이썬의 리스트로 풀어서 쓴다고 생각했을 때 인덱스의 순서이다.<br>
1 2 3 <br>
4 5 6 <br>
7 8 9 <br>
위와 같은 3 * 3 행렬로 봤을 때 파이썬의 리스트로 풀어서 쓰면 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]가 된다. 여기서 이 리스트의 이름을 arr라고 했을 때 arr[0]은 [1, 2, 3]이 된다. 그리고 arr[0][0]은 1이다. 여기서 [1, 2, 3]이 axis가 1인 축이다. 즉, 다차원 리스트에 대해서 인덱스의 순서가 axis의 순서가 되는 것이다. 아래 예제를 보면 이해가 쉽다.
```python
arr1 = np.arange(start=1, stop=28).reshape(3, 3, 3)
print(arr1)
print(arr1.sum(axis=0))
print(arr1.sum(axis=1))
print(arr1.sum(axis=2))
```
result
```console
array([[[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9]],

       [[10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]],

       [[19, 20, 21],
        [22, 23, 24],
        [25, 26, 27]]])
array([[30, 33, 36],
       [39, 42, 45],
       [48, 51, 54]])
array([[12, 15, 18],
       [39, 42, 45],
       [66, 69, 72]])
array([[ 6, 15, 24],
       [33, 42, 51],
       [60, 69, 78]])
```
보면 axis가 0인 sum에서는 arr1[0], arr[1], arr[2] 가 각각 3 x 3 인 2차원 배열이라고 생각했을 때 arr1[0] + arr[1] + arr[2] 의 결과가 나왔다.<br>
axis가 1인 sum에서는 arr[n][0] + arr[n][1] + arr[n][2] 의 결과가 나왔고, <br>
axis가 2인 sum에서는 arr[n][m][0] + arr[n][m][1] + arr[n][m][2] 의 결과가 나온 걸 볼 수 있다.
<br>
<h2>pandas</h2>
판다스는 csv 파일을 처리하는 라이브러리이다. csv는 comma-seperated values의 약자로 쉼표로 분리 된 값들이라는 뜻이다. 즉, 엑셀과 비슷한 형식으로 저장 돼있는 파일이다.<br>
데이터 프레임은 기본적으로 첫번째 열은 데이터의 순서를 나타내는 숫자이고 첫번째 행은 해당 데이터가 나타내는 의미이다(ex 이름, 성별, 연령 etc).<br>
 그래서 딕셔너리로 변환할 경우에 key는 첫번째 행의 인자들이 되고, value는 해당 인자가 속해있는 열의 값들이 된다.
