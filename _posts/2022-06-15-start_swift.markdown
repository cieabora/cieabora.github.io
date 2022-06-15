---
layout: post
title: "Start Swift"
date: 2022-06-15
categories: Swift
---
#swift는 변수를 선언할 때 
#var 변수이름 : 데이터타입 = 값
#과 같은 형식으로 선언한다.
#출력은 파이썬과 유사하게 하고, 문자열을 감쌀 때에는 무조건 쌍 따옴표로 감싸야 한다.

#if문의 형식은 C언어와 유사하고 한 줄 if문도 가능하다.

```swift
import UIKit

// 다크모드 여부
var isDarkMode : Bool = true


if (isDarkMode == true){
    print("다크모드 입니다.")
}
else{
    print("다크모드가 아닙니다")
}

// if 다크모드가 true -> 다크모드 입니다.
var title : String = isDarkMode ? "다크모드 입니다." : "다크모드가 아닙니다."

print(title)

```
