---
layout: post
title: "[ML]week4-Bayesian Optimization"
date: 2022-09-10 22:44
categories: ML
---
<h2>베이지안 최적화 수행단계</h2>
<ol>
<li>최초에는 랜덤하게 하이퍼 파라미터들을 샘플링하여 성능 결과를 관측</li>
<li>관측된 값을 기반으로 대체 모델은 최적 함수를 예측 추정</li>
<li>획득 함수에서 다음으로 관측할 하이퍼 파라미터를 추출</li>
<li>해당 하이퍼 파라미터로 관측된 값을 기반으로 대체 모델은 다시 최적 함수 예측 추정</li>
</ol>

<h2>베이지안 최적화 구현 요소</h2>
<ol>
<li>입력 값 범위(Search Space)</li>
<li>목적 함수의 출력값</li>
<li>목적 함수 변환 최소값 유추</li>
</ol>


