---
title: "08. LocalLens 프로젝트 소개와 검색 기능"
description: "LocalLens의 파일 검색 구조, 모델 구성과 주요 기능을 정리합니다."
categories:
- 2.PROJECT
- 2-6. LocalLens
tags:
- 프로젝트 자료
toc: true
date: 2026-09-24 00:00:00 +0900
comments: true
mermaid: true
math: false
---

LocalLens의 파일 검색 구조, 모델 구성과 주요 기능을 정리합니다.

{% raw %}

### 프로젝트 개요

이미지, 텍스트, PDF 파일에 대해 단순 파일명이 아닌 의미론적 검색이 가능한 로컬 파일 검색기

‘자연어 질의’를 통해 사용자가 대략적으로 묘사한 내용만으로도 파일의 맥락을 파악해 정확한 결과물을 찾아냅니다.

### 프로젝트 배경

#### 기존 검색 방식의 한계

- 대부분의 파일 검색기는 파일명 기반의 단순 텍스트 매칭에 의존

- 사용자는 정확한 파일명이나 키워드를 기억해야만 원하는 파일 검색 가능

- 이미지, PDF, 문서 내용 등 비정형 데이터의 의미를 반영한 검색이 어려움

#### 핵심 기능

- 멀티모달 지원: 텍스트, 이미지, 문서 형식 지원

- 의미론적 검색: 키워드 매칭이 아닌 쿼리를 통해 파일의 내용과 맥락을 파악하여 탐색

## 구조도

![LocalLens 프로젝트 소개 이미지 2](/assets/images/source-archives/locallens/intro-02.webp)

Frontend: 사용자로부터 쿼리 입력 및 검색 대상 파일 경로 설정

Search Engine: 쿼리 임베딩 요청 및 검색 수행

Encoders: 각 모달리티별 인코더가 쿼리와 파일을 벡터화

Vector DB (FAISS/SQLite): 쿼리와 파일 간 유사도 계산 후 Top K 결과 반환

Storage Sync: 로컬 저장소와 벡터 저장소 간 동기화 유지

## 시퀀스 다이어 그램

![LocalLens 프로젝트 소개 이미지 3](/assets/images/source-archives/locallens/intro-03.webp)

## 인코더 계층 구조도

![LocalLens 프로젝트 소개 이미지 4](/assets/images/source-archives/locallens/intro-04.webp)

### 인코더 목록

|  |  |
| --- | --- |
| Text Encoder | intfloat/multilingual-e5-small |
| Vision Encoder | Siglip 2 (google/siglip2-so400m-patch16-naflex) |
| PDF Encoder | PyMuPDF + Clova VLM → Text Encoder |

## vectorstore 동기화 시퀀스 다이어그램

![LocalLens 프로젝트 소개 이미지 5](/assets/images/source-archives/locallens/intro-05.webp)

## 프로젝트 결과물

### 주요 기능

- 지정 디렉토리 내 자연어 쿼리 기반 검색

- 다양한 확장자 선택 옵션

- 검색 결과 절대 경로 반환

### 실행 화면

![LocalLens 프로젝트 소개 이미지 6](/assets/images/source-archives/locallens/intro-06.webp)

![LocalLens 프로젝트 소개 이미지 7](/assets/images/source-archives/locallens/intro-07.webp)

![LocalLens 프로젝트 소개 이미지 8](/assets/images/source-archives/locallens/intro-08.webp)

### 발표 영상

[발표 영상](https://www.youtube.com/watch?v=nFPk2s7823Q)

### 시연 영상

[시연 영상](https://oopy.lazyrockets.com/api/v2/notion/fileUrl?src=attachment%3Afe95ae96-1923-4e5c-90de-a452ecfad802%3Abandicam_2026-02-09_11-38-04-803.mp4&blockId=302f145d-e995-81df-96cc-d99f3cd20639#t=0.0001)

{% endraw %}

## 관련 글

- [09. LocalLens 모델 비교와 PDF 검색 — 발표 자료]({% post_url projects/locallens/2026-09-24-project-locallens-source-model-pdf-slides %})

- [텍스트·이미지·PDF 파일 검색 시연]({% post_url projects/locallens/2026-09-24-project-locallens-demo-walkthrough %})
