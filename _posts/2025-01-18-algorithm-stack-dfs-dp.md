---
title: SWEA 알고리즘 - Stack, DFS, 동적 계획법
categories:
- 1.TIL
- 1-4.ALGORITHM

tags:
- algorithm
- swea
- stack
- dfs
- dp
- memoization
- backtracking
- divide-conquer
date: 2023-08-04 09:00:00 +0900
toc: true
comments: false
mermaid: true
math: true
---

# SWEA 알고리즘 - Stack, DFS, 동적 계획법

> **SW Expert Academy** 파이썬 SW문제해결 기본 과정의 Stack 관련 고급 알고리즘 학습

## Stack1 - 기본 개념

### 1. Stack 자료구조의 개념
- LIFO (Last In, First Out) 원리
- Stack의 기본 연산 (push, pop, peek)
- Python에서 Stack 구현 방법
- Stack의 활용 사례

### 2. Stack의 응용
- 괄호 검사
- 후위 표기법 계산
- 함수 호출 스택
- 되돌리기(Undo) 기능

### 3. Memoization (메모이제이션)
- 중복 계산 방지 기법
- 동적 계획법의 핵심 요소
- 메모리와 시간의 트레이드오프
- Python에서 메모이제이션 구현

### 4. DP (동적 계획법)
- 최적 부분 구조와 중복 부분 문제
- Top-down vs Bottom-up 접근법
- DP 테이블 설계
- 상태 전이 방정식

### 5. DFS (깊이 우선 탐색)
- 그래프 탐색 알고리즘
- **인접 리스트 방식**: 메모리 효율적
- **인접 행렬 방식**: 구현 간단
- 재귀와 스택을 이용한 구현

### 6. 관련 문제

#### Stack1 문제들
- **[종이붙이기](https://github.com/figure-2/Algorithm/blob/master/swea/4869_%EC%A2%85%EC%9D%B4%EB%B6%99%EC%9D%B4%EA%B8%B0/sol.py)** - DP와 재귀 활용
- **[괄호검사](https://github.com/figure-2/Algorithm/blob/master/swea/4866_%EA%B4%84%ED%98%B8%EA%B2%80%EC%82%AC/sol.py)** - Stack 자료구조 활용
- **[그래프 경로](https://github.com/figure-2/Algorithm/blob/master/swea/4871_%EA%B7%B8%EB%9E%98%ED%94%84%EA%B2%BD%EB%A1%9C/sol.py)** - DFS (인접 행렬/리스트 방식)

## Stack2 - 고급 기법

### 1. 계산기
- 중위 표기법을 후위 표기법으로 변환
- 후위 표기법 계산
- 연산자 우선순위 처리
- 스택을 이용한 계산기 구현

### 2. 백트래킹 (Backtracking)
- 모든 가능한 해를 체계적으로 탐색
- 가지치기(Pruning)를 통한 최적화
- N-Queen 문제 해결
- 백트래킹의 시간 복잡도

### 3. 분할정복 (Divide and Conquer)
- 문제를 작은 단위로 분할
- 각 부분 문제 해결
- 결과를 결합하여 최종 해 도출
- 분할정복의 대표적 예시

### 4. 관련 문제

#### Stack2 문제들
- **[미로](https://github.com/figure-2/Algorithm/blob/master/swea/4875_%EB%AF%B8%EB%A1%9C/sol.py)** - 백트래킹과 DFS
- **[배열최소합](https://github.com/figure-2/Algorithm/blob/master/swea/4881_%EB%B0%B0%EC%97%B4%EC%B5%9C%EC%86%8C%ED%95%A9/sol.py)** - 백트래킹과 최적화

## 학습 포인트

### 알고리즘 패턴
1. **Stack 활용**: LIFO 특성을 이용한 문제 해결
2. **DFS 탐색**: 깊이 우선으로 모든 경로 탐색
3. **DP 최적화**: 중복 계산 제거로 성능 향상
4. **백트래킹**: 체계적인 해 탐색

### 성능 고려사항
- **시간 복잡도**: O(2^n) vs O(n²) 최적화
- **공간 복잡도**: 스택 오버플로우 방지
- **메모이제이션**: 시간과 공간의 균형

### 실무 적용
- 컴파일러의 구문 분석
- 게임 AI의 경로 탐색
- 최적화 문제 해결
- 그래프 알고리즘 구현

Stack과 DFS, DP를 통해 복잡한 문제 해결 능력을 기를 수 있습니다.
