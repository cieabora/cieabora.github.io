---
layout: post
title: "[JAVA]java start"
date: 2022-06-16 09:47
categories: java
---
자바의 기본 포맷은 아래와 같다.
```java
public class Sample{
    public static void main(String args[]){
        System.out.println("hello java!");
    }
}
```
여기서 주의할 점은 자바의 파일 이름과 매인 클래스의 이름이 같아야 한다.  
자바에서의 배열은 C언어와 유사하고 for each문은 아래와 같이 쓴다.
```java
public class Sample{
    public static void main(String args[]){
        int a[] = {10, 20, 30, 40, 50};
        for(int num: a){
            System.out.println(num);
        }
    }
}
```

그리고 array 기반의 리스트를 쓸 수도 있는데 그럴 때에는 아래와 같이 써야한다.
```java
import java.util.ArrayList;
import java.util.Arrays;

public class Sample{
    public static void main(String args[]){
        ArrayList<Integer> a = new ArrayList<>(Arrays.asList(1, 2, 3, 4));
    }
}
```
