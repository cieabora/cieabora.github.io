---
layout: post
title: "[SWIFT]iOS-Screen Action"
date: 2022-10-02 13:48
categories: Swift
---
iOS에서 화면을 전환하는 방법에는 크게 4가지가 있다.<br>
1. View Controller의 View 위에 다른 View를 덮어 씌우기<br>
2. View Controller에서 다른 View Controller 를 호출하여 전환하기<br>
3. Navigation Controller를 사용하여 화면 전환하기<br>
4. 화면 전환용 객체 세그웨이(Segueway)를 사용하여 화면 전환하기<br>
위에서 1번은 메모리 누수의 이슈도 있고 보안의 취약성 이슈도 있기 때문에 쓰지 말아야 할 방식이라고 한다. 아래는 각각의 방식에서 사용하는 함수이다.<br>
2. present, dismiss<br>
3. push, pop<br>
4. 2, 3<br>

<h2> 각각의 실습 방식 및 코드 </h2>
<h3> segue로 push </h3>
다른 view controller를 만들고 cocoa touch 파일을 만들어서 해당 파일과 view controller의 클래스와 연결해준다. 그 다음,view controller에서 컨트롤을 누르고 드래그를 하면 옵션이 뜨는데 show를 눌러준다. 그리고 밑은 back button의 코드이다.


```swift
    @IBAction func tapBackButton(_ sender: UIButton) {
        self.navigationController?.popViewController(animated: true)
    }
```
<h3> segue로 present</h3>
위와 방식은 비슷하고 back button의 코드는 조금 다르다.


```swift
    @IBAction func tapBackButton(_ sender: UIButton) {
        self.presentingViewController?.dismiss(animated: true, completion: nil)
    }
```
<h3> 코드로 push </h3>
이 방식은 버튼에 액션을 입히는 방식으로 코드를 작성했다. 아래는 push 하는 코드와 back button의 코드이다.


```swift
    // push code
    @IBAction func tapCodePush(_ sender: UIButton) {
        guard let viewController = self.storyboard?.instantiateViewController(identifier: "codePushViewController") else{ return }
        self.navigationController?.pushViewController(viewController, animated: true)
    }
    
    // back button code
        @IBAction func pushBackButton(_ sender: UIButton) {
        self.navigationController?.popViewController(animated: true)
    }
```
<h3> 코드로 present </h3>
위와 동일하다


```swift
    // present code
        @IBAction func tapCodePresent(_ sender: UIButton) {
        guard let viewController = self.storyboard?.instantiateViewController(identifier: "codePresentViewController") else{ return }
        viewController.modalPresentationStyle = .fullScreen
        self.present(viewController, animated: true, completion: nil)
    }

    // back button code
        @IBAction func tapBackButton(_ sender: UIButton) {
        self.dismiss(animated: true)
    }
```
여기서 주의할 점은 코드에서 instantiateViewController 함수는 옵셔널 값을 반환하기 때문에 옵셔널 바인딩을 해줘야 한다는 점이다.
