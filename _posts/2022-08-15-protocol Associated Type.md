---
layout: post
title: "[SWIFT]Protocol Associated Type"
date: 2022-08-15 02:04
categories: Swift
---
프로토콜에서는 associated type이라는 타입이 존재하는데 제네릭과 유사하다. 들어오는 타입을 오토 캐스팅 해주는 키워드인거 같다.
```swift
protocol PetHaving{
    associatedtype T
    var pets: [T] { get set }
    mutating func gotNewPet(_ newPet: T)
}

extension PetHaving{
    mutating func gotNewPet(_ newPet: T){
        self.pets.append(newPet)
    }
}

enum Animal{
    case cat, bird, dog
}

struct Friend: PetHaving{
    var pets: [Animal] = []
}

struct Family: PetHaving{
    var pets: [String] = []
}

var myFriend = Friend()

myFriend.gotNewPet(Animal.bird)
myFriend.gotNewPet(Animal.cat)
myFriend.gotNewPet(Animal.dog)
myFriend.pets

var myFamily = Family()
myFamily.gotNewPet("turtle")
myFamily.gotNewPet("rabbit")
myFamily.gotNewPet("puppy")
myFamily.pets
```
