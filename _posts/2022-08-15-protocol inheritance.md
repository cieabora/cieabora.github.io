---
layout: post
title: "[SWIFT]Protocol Inheritance"
date: 2022-08-15 01:41
categories: Swift
---
프로토콜도 상속이 가능한데 별 내용은 없다. 아래와 같이 구현하면 된다.


```swift
protocol Naming{
    var name : String{ get set }
    func getName() -> String
}

protocol Aging{
    var age: Int{ get set }
    
}

protocol UserNotifiable : Naming, Aging{

}

struct hi: UserNotifable{
    var name: String
    var age
}
```
위에서 hi라는 메소드를 만들 때 에러가 발생한다. getName이라는 메소드가 없기 때문이다.
