---
layout: post
title: "[SWIFT]Property Observer"
date: 2022-06-21 15:48
categories: Swift
---
프로퍼티 옵저버란 값이 변할 때 함수를 실행하는 기능이다.  
```swift
var myAge = 0{
    willSet{
        print("값이 설정될 예정이다. / myAge: \(myAge)")
    }
    didSet{
        print("값이 설정되었다. / myAge: \(myAge)")
    }
}
```
willSet은 myAge라는 변수의 값이 바뀌는 시점에 호출 된다. 그리고 값이 바뀌고 난 뒤 didSet이 호출된다.  
예제 코드)
```swift
import UIKit

var myAge = 0{
    willSet{
        print("값이 설정될 예정이다. / myAge: \(myAge)")
    }
    didSet{
        print("값이 설정되었다. / myAge: \(myAge)")
    }
}

myAge = 10
myAge = 20
```
결과)
```console
값이 설정될 예정이다. / myAge: 0
값이 설정되었다. / myAge: 10
값이 설정될 예정이다. / myAge: 10
값이 설정되었다. / myAge: 20
```
