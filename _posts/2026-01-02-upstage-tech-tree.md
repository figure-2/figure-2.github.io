---
title: "트리 (Tree)"
categories:
- 1.TIL
- 1-2.UPSTAGE_AI_AGENT
- 1-2-2.DATA_STRUCTURE_ALGORITHM
- TECH_NOTE
tags:
- upstage
- sesac
- ai-agent
- data-structures
- tech-note
- data-structure-algorithm
toc: true
date: 2026-01-02 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# 트리 (Tree)

> **한줄 정의**
> 계층적 구조를 표현하는 비선형 자료구조. 루트 노드에서 시작하여 부모-자식 관계로 연결된 노드의 집합이다.

## 핵심 이해

이진 트리(Binary Tree)는 각 노드가 최대 두 개의 자식을 가진다. **이진 탐색 트리(BST)**는 왼쪽 자식 < 부모 < 오른쪽 자식 규칙을 따라 O(log n) 탐색이 가능하다. 트리 순회는 전위(Pre-order: 루트→좌→우), 중위(In-order: 좌→루트→우), 후위(Post-order: 좌→우→루트)가 있다.

균형 트리(AVL, Red-Black Tree)는 삽입/삭제 시 자동으로 균형을 유지하여 최악의 경우에도 O(log n)을 보장한다. 실제 응용으로 파일 시스템, 데이터베이스 인덱스(B-Tree), HTML DOM, JSON 파싱 트리 등이 있다.

## 구조 시각화

![트리 (Tree) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-tree-diagram-1.svg)

## 관련 개념

- 그래프 - 트리는 사이클 없는 연결 그래프
- 힙 - 완전 이진 트리 기반
- BFS-DFS - 트리 순회 알고리즘
- 해시 - 탐색 성능 비교
