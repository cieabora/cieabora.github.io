---
layout: post
title: "[SWIFT]Operate Property"
date: 2022-10-01 17:13
categories: Swift
---
계산 프로퍼티는 클래스나 구조체 등에서 스스로 계산을 할 수 있는 유사 함수 또는 매크로의 개념이다. 안 그래도 C++이나 파이썬에서 이런 류의 변수가 있으면 편하다고 생각했는데 스위프트에는 이러한 기능이 있었다.<br>
사용법은 간단하다.
```swift
struct A{
    var B: Int{
        get{
            ~~~~
            return ~~~
        }
        set(param){
            ~~~~
        }
    }
}
```
위의 형식으로 사용하면 된다. get에는 해당 프로퍼티를 호출했을 때 원하는 값을 return에 넣으면 되고, set에는 해당 프로퍼티에 값을 세팅했을 때, 값이 들어갈 변수의 이름을 param에 넣고, 어떠한 연산을 할지 바디 블럭에 넣으면 된다.<br>
예제)



```swift
struct Stock{
    var averagePrice: Int
    var quantity: Int
    var purchasePrice: Int{
        get{
            return averagePrice * quantity
        }
        
        set(newPrice){
            averagePrice = newPrice / quantity
        }
    }
}

var stock = Stock(averagePrice: 2300, quantity: 3)

print(stock)
print(stock.purchasePrice)
stock.purchasePrice = 3000
print(stock.averagePrice)
```
실행 결과)


```console
Stock(averagePrice: 2300, quantity: 3)
6900
1000
```
