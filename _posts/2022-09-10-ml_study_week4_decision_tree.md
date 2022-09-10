---
layout: post
title: "[ML]week4-Decision Tree"
date: 2022-09-10 08:29
categories: ML
---
<h1>Decision Tree</h1>
<h2>Decision Tree의 주요 하이퍼 파라미터</h2>
<h3>max_depth </h3>
<ol>
<li>트리의 최대 깊이를 규정</li>
<li>디폴트는 None. None으로 설정하면 완벽하게 클래스 결정 값이 될 때까지 깊이를 계속 키우며 분할하거나 노드가 가지는 데이터 개수가 min_samples_split보다 작아질 때까지 계속 깊이를 증가시킴.</li>
<li>깊이가 깊어지면 min_samples_split 설정대로 최대 분할하여 과적합할 수 있으므로 적절한 값으로 제어 필요.</li>
</ol>
<h3>max_features</h3>
<ol>
<li>최적의 분할을 위해 고려할 최대 피처 개수. 디폴트는 None으로 데이터 세트의 모든 피처를 사용해 분할 수행.</li>
<li>int 형으로 지정하면 대상 피처의 개수, float 형으로 지정하면 전체 피처 중 대상 피처의 퍼센트</li>
<li>'sqrt'는 전체 피처 중 sqrt(전체 피처 개수) 만큼 선정</li>
<li>'auto'로 지정하면 sqrt와 동일</li>
<li>'log'는 전체 피처 중 log2(전체 피처 개수)로 선정</li>
<li>'None'은 전체 피처 선정</li>
</ol>
<h3>min_samples_split </h3>
<ol>
<li>노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합을 제어하는 데 사용됨.</li>
<li>디폴트는 2이고 작게 설정할수록 분할되는 노드가 많아져서 과적합 가능성 증가.</li>
</ol>
<h3>min_samples_leaf </h3>
<ol>
<li>분할이 될 경우 왼쪽과 오른쪽의 브랜치 노드에서 가져야 할 최소한의 샘플 데이터 수</li>
<li>큰 값으로 설정 될수록, 분할될 경우 왼쪽과 오른쪽의 브랜치 노드에서 가져야 할 최소한의 샘플 데이터 수 조건을 만족시키기가 어려우므로 노드 분할을 상대적으로 덜 수행함.</li>
</ol>
<h3>max_leaf_nodes </h3>
<ol>
<li>말단 노드의 최대 개수</li>
</ol>
<br>
<h2>결정트리 피처 중요도</h2>
트리 계열에는 feature_importances_ 라는 속성이 있다. 이는 각각의 feature에 대한 중요도를 담고 있는데, 각각의 feature에서 얼마나 분할했는지에 대한 지표를 나타낸다.
<h2>GridSearchCV</h2>
GridSearchCV 함수를 이용하여 하이퍼 파라미터에 따라서 학습을 해보고 테스트를 수행하여 가장 높은 정확도를 가지는 파라미터를 찾을 수 있다.
<br><br>

