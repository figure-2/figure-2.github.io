---
title: "그래프 (Graph)"
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
## 그래프 (Graph)

> **한줄 정의**
> 정점(Vertex)과 간선(Edge)으로 구성된 비선형 자료구조. 객체 간의 관계를 표현하는 가장 일반적인 구조다.

## 핵심 이해

그래프는 **방향 그래프(Directed)**와 **무방향 그래프(Undirected)**, **가중치 그래프(Weighted)**로 구분된다. 구현 방식은 **인접 행렬(Adjacency Matrix)**(O(V²) 공간, O(1) 간선 조회)과 **인접 리스트(Adjacency List)**(O(V+E) 공간, 메모리 효율적) 두 가지다.

대표 알고리즘으로 BFS/DFS(탐색), Dijkstra(최단 경로), 위상 정렬(DAG), 최소 신장 트리(Prim, Kruskal)가 있다. 소셜 네트워크, 지도 네비게이션, 추천 시스템, 컴파일러 의존성 분석 등 다양한 실제 문제를 모델링한다.

## 구조 시각화

![그래프 (Graph) 다이어그램 1](/assets/images/upstage-ai-agent/diagrams/02-concepts-graph-diagram-1.svg)

## 관련 개념

- 트리 - 사이클 없는 연결 그래프가 트리
- BFS-DFS - 그래프 탐색 알고리즘
- 시간복잡도 - 그래프 알고리즘 복잡도 분석
