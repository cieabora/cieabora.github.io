---
layout: post
title: "[SWIFT]Protocol Extension"
date: 2022-08-15 01:50
categories: Swift
---
프로토콜에서는 정의만 가능하다. 그리고 구현은 할 수 없는데 extension이라는 키워드를 사용하면 구현이 가능하다. 아래의 코드를 보자
```swift
protocol Naming{
    var lastname : String{ get set }
    var firstname: String{ get set }
    func getName() -> String
}

extension Naming{
    func getFullname() -> String{
        return self.lastname + " " + self.firstname
    }
}

struct Friend : Naming{
    var lastname: String
    var firstname: String
    func getName() -> String{
        return self.lastname
    }
}

let myFriend = Friend(lastname: "박", firstname: "승우")
myFriend.getName()
myFriend.getFullname()
```
extension Naming 이라는 키워드를 통해서 getFullname이라는 메소드의 구현을 할 수 있었다.
