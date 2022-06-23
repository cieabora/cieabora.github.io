---
layout: post
title: "[SWIFT]Parameter Inout"
date: 2022-06-23 14:29
categories: Swift
---
메소드 안에서 파라미터를 변경하려고 하면 에러가 발생한다.
```swift
func error_func(_ name: String){
    name = "개발하는 " + name
    print("안녕?! 난 \(name) 라고 해")
}
```
error message:
```console
expression failed to parse:
error: ex1.playground:10:5: error: cannot assign to value: 'name' is a 'let' constant
    name = "개발하는 " + name
```
아마도 이게 파라미터를 받아올 때 상수 취급을 해서 그러는 것 같다. 
메소드 안에서 파라미터의 값을 바꾸고 싶다면 inout 키워드를 이용하면 된다.
```swift
func non_error_func(_ name:  inout String){
    name = "개발하는 " + name
    print("안녕?! 난 \(name) 라고 해")
}
```
대신 여기서도 기본 문자열은 상수 취급을 받기 때문에 함수를 호출할 때 원하는 문자열을 변수에 넣고 주소 값을 넘기는 키워드인 &를 붙여야 한다.
```swift
var name: String = "씅우"
non_error_func(&name)
```
