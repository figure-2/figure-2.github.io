---
title: SWEA 알고리즘 - Tree, Binary Tree, Heap
categories:
- 1.TIL
- 1-4.ALGORITHM

tags:
- algorithm
- swea
- tree
- binary-tree
- heap
- expression-tree
- binary-search-tree
date: 2023-08-08 09:00:00 +0900
toc: true
comments: false
mermaid: true
math: true
---

# SWEA 알고리즘 - Tree, Binary Tree, Heap

> **SW Expert Academy** 파이썬 SW문제해결 기본 과정의 Tree 관련 알고리즘 학습

## Tree 자료구조 기초

### 1. Tree
- 계층적 자료구조
- 노드(Node)와 간선(Edge)으로 구성
- 루트(Root), 리프(Leaf), 부모(Parent), 자식(Child)
- 트리의 높이와 깊이

### 2. Binary Tree (이진 트리)
- 각 노드가 최대 2개의 자식을 가지는 트리
- **완전 이진 트리**: 마지막 레벨을 제외하고 모든 레벨이 채워짐
- **포화 이진 트리**: 모든 레벨이 완전히 채워진 트리
- **균형 이진 트리**: 좌우 서브트리의 높이 차이가 1 이하

### 3. Expression Tree (수식 트리)
- 수학적 표현식을 트리로 표현
- 중위 순회로 원래 수식 복원
- 후위 순회로 계산 결과 도출
- 전위 순회로 수식 변환

### 4. Binary Search Tree (이진 탐색 트리)
- 왼쪽 자식 < 부모 < 오른쪽 자식
- 탐색, 삽입, 삭제 연산
- 평균 O(log n), 최악 O(n) 시간 복잡도
- 균형 이진 탐색 트리의 필요성

### 5. Heap (힙)
- 완전 이진 트리 기반 자료구조
- **최대 힙**: 부모 ≥ 자식
- **최소 힙**: 부모 ≤ 자식
- 우선순위 큐 구현에 활용

## 관련 문제

### Tree 문제들
- **[subtree](https://github.com/figure-2/Algorithm/blob/master/swea/5174_subtree/sol.py)** - 서브트리 탐색
- **[이진탐색](https://github.com/figure-2/Algorithm/blob/master/swea/5176_%EC%9D%B4%EC%A7%84%ED%83%90%EC%83%89/sol.py)** - 이진 탐색 트리
- **[이진힙](https://github.com/figure-2/Algorithm/blob/master/swea/5177_%EC%9D%B4%EC%A7%84%ED%9E%99/sol.py)** - 힙 자료구조

## 학습 포인트

### 트리 순회 (Tree Traversal)
1. **전위 순회 (Preorder)**: 루트 → 왼쪽 → 오른쪽
2. **중위 순회 (Inorder)**: 왼쪽 → 루트 → 오른쪽
3. **후위 순회 (Postorder)**: 왼쪽 → 오른쪽 → 루트
4. **레벨 순회 (Level-order)**: 레벨별로 탐색

### 힙 연산
```python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

### 성능 고려사항
- **시간 복잡도**: O(log n) 삽입/삭제
- **공간 복잡도**: O(n) 저장 공간
- **균형 유지**: 트리의 성능 최적화

### 실무 적용
- **데이터베이스 인덱스**: B-Tree, B+Tree
- **파일 시스템**: 디렉토리 구조
- **압축 알고리즘**: 허프만 코딩
- **우선순위 큐**: 작업 스케줄링
- **정렬 알고리즘**: 힙 정렬

Tree와 Heap을 통해 계층적 데이터 처리 능력을 기를 수 있습니다.
