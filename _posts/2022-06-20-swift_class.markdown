---
layout: post
title: "[SWIFT]Struct and Class"
date: 2022-06-20 04:02
categories: Swift
---
swift에도 struct와 구조체라는 개념이 있다.  
구조체를 선언할 때에는 아래와 같이 선언할 수 있다.
```swift
// 유튜버 (데이터) 모델 - struct / 구조체
struct YoutuberStruct{
    var name : String
    var subscriberCount : Int
}
```
이 구조체의 객체를 만들 때에는
```swift
var devPark = YoutuberStruct(name: "박승우", subscriberCount: 999)
```
위와 같이 만들 수 있다.  
구조체의 기본 복사는 깊은 복사로 이루어진다. 예제를 보면
```swift
struct YoutuberStruct{
    var name : String
    var subscriberCount : Int
}

var devPark = YoutuberStruct(name: "박승우", subscriberCount: 999)
var devParkClone = devPark // deep copy

print("before devParkClone.name : ", devParkClone.name)

devParkClone.name = "호롤롤로"

print("after devParkClone.name : ", devParkClone.name)
print("after devPark.name : ", devPark.name)
```
위의 예제를 실행시키면 아래와 같은 결과가 나온다.
```console
before devParkClone.name :  박승우
after devParkClone.name :  호롤롤로
after devPark.name :  박승우
```
devParkClone의 이름을 바꿨는데 devPark의 이름이 바뀌지 않은 것을 보면 deep copy임을 알 수 있다.  
  
이제 클래스를 보겠다.
생성자를 포함한 클래스의 기본 선언 형식은 아래와 같다.
```swift
class YoutuberClass{
    var name : String
    var subscriberCount : Int
    // 생성자
    // init으로 매개변수를 가진 생성자 메소드를 만들어야
    // 매개변수를 넣어서 그 값을 가진 객체를 만들 수 있다.
    init(_ name: String, _ subscribersCount: Int){
        self.name = name
        self.subscriberCount = subscribersCount
    }
}
```
그리고 위의 struct 예제와 동일한 예제를 보면
```swift
class YoutuberClass{
    var name : String
    var subscriberCount : Int
    // 생성자
    // init으로 매개변수를 가진 생성자 메소드를 만들어야
    // 매개변수를 넣어서 그 값을 가진 객체를 만들 수 있다.
    init(_ name: String, _ subscribersCount: Int){
        self.name = name
        self.subscriberCount = subscribersCount
    }
}

var SeungWoo = YoutuberClass("박승우", 999)
var SeungWooclone = SeungWoo // shallow copy

print("before SeungWoo.name : ", SeungWoo.name)

SeungWooclone.name = "호롤롤로"

print("after SeungWooclone.name : ", SeungWooclone.name)
print("after SeungWoo.name : ", SeungWoo.name)
```
아래와 같은 결과가 나온다.
```console
before SeungWoo.name :  박승우
after SeungWooclone.name :  호롤롤로
after SeungWoo.name :  호롤롤로
```
이를 통해 struct는 deep copy, class는 shallow copy가 이루어 지는 것을 알 수 있다.
