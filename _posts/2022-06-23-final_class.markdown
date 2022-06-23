---
layout: post
title: "[SWIFT]Final Class"
date: 2022-06-23 14:08
categories: Swift
---
swift에는 final class라고 상속이 되지 않는 클래스가 있다.
```swift
import UIKit

final class Friend{
    var name: String
    
    init(name: String){
        self.name = name
    }
}

class BestFriend: Friend{
    
    override init(name: String){
        super.init(name: "베프 " + name)
    }
}

let myFriend = Friend(name: "승우")
let myBF = BestFriend(name: "혁")

```
error message:
```console
expression failed to parse:
error: ex1.playground:11:7: error: inheritance from a final class 'Friend'
class BestFriend: Friend{
      ^
```

