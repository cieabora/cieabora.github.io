---
layout: post
title: "[SWIFT]Method Parameter Name"
date: 2022-06-21 16:04
categories: Swift
---
메소드의 매개변수의 이름은 기본적으로 메소드를 정의할 때와 메소드를 호출할 때 동일하게 쓰인다.
```swift
// 메소드 정의
func myFunction(name: String) -> String{
    return "안녕하세요?! \(name) 입니다."
}

// 메소드 호출
myFunction(name: "씅우")
```
  
그런데 메소드를 정의할 때와 호출할 때 다르게 사용하고 싶으면 아래와 같이 사용하면 된다.
```swift
// 메소드 정의
func myFunctionSecond(your_name hi: String) -> String{
    return "안녕하세요?! \(hi) 입니다."
}

// method call
myFunctionSecond(your_name: "씅우")
```
  
혹은 메소드를 호출할 때 매개변수의 이름을 쓰기 귀찮다면 _을 추가하면 된다.
```swift
// 메소드 정의
func myFunctionThird(_ name: String) -> String{
    return "안녕하세요?! \(name) 입니다."
}

// method call
myFunctionThird("씅우")
```
