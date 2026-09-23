---
title: "10. LocalLens 시연 — 텍스트·이미지·PDF 파일 검색"
categories:
- 2.PROJECT
- 2-6. LocalLens
tags:
- LocalLens
- 서비스 시연
toc: true
date: 2026-09-24 00:00:00 +0900
comments: true
mermaid: false
math: false
---

LocalLens는 로컬 폴더의 파일을 자연어로 찾는 검색기입니다. 검색 경로와 파일 형식을 선택하고, 검색어를 입력한 뒤 결과 파일을 여는 흐름으로 구성했습니다. 아래는 텍스트·이미지·PDF를 차례로 검색한 시연 화면입니다.

## 검색할 폴더 선택

먼저 파일 경로 버튼으로 검색 대상 폴더를 선택합니다. 검색 범위는 사용자가 지정한 로컬 폴더이며, 왼쪽에서 이미지·텍스트·문서 형식을 선택할 수 있습니다.

![검색 대상 폴더 선택 창](/assets/images/project-demos/locallens/demo-07m43s.png)
_07:43 · 검색할 로컬 폴더 선택_

## 텍스트 검색과 파일 열기

텍스트 파일을 대상으로 ‘딥러닝과 관련된 기술’을 검색했습니다. 결과에는 파일 경로가 목록으로 표시되며, 파일을 열어 검색어와 관련된 본문을 확인하는 흐름으로 이어집니다. 이 장면에서는 검색 결과 중 RAG를 설명하는 텍스트 파일을 열었습니다.

![자연어 텍스트 검색 결과에서 파일을 연 화면](/assets/images/project-demos/locallens/demo-08m03s.png)
_08:03 · 텍스트 검색 결과와 선택한 파일의 본문_

## 이미지 검색 결과

이미지 검색에서는 ‘강아지’를 입력하고 Top K를 5로 설정했습니다. 결과는 이미지 파일의 경로 목록으로 표시됩니다. 아래 장면은 이미지 검색 결과를 확인한 뒤 다음 검색을 위해 문서 형식을 선택한 시점입니다. 왼쪽의 선택 상태와 결과 목록의 유형이 다른 이유도 이 전환 과정에 있습니다.

![강아지 검색 결과로 표시된 이미지 파일 경로 목록](/assets/images/project-demos/locallens/demo-08m43s.png)
_08:43 · 이미지 검색 결과 확인 후 문서 검색으로 전환_

## PDF 검색 결과

문서 형식에서는 PDF를 대상으로 ‘취업’을 검색했습니다. 결과 영역에 PDF 파일 경로 5개가 표시됩니다. 텍스트와 이미지 검색에서 사용한 입력·결과 확인 동선을 PDF 검색에도 동일하게 적용했습니다.

![취업 검색어에 대한 PDF 파일 결과 목록](/assets/images/project-demos/locallens/demo-09m03s.png)
_09:03 · PDF 검색 결과_

시연은 파일 형식별 검색과 결과 확인 동선을 보여주는 기록입니다. 화면에 보이는 검색 완료 표시는 해당 시연의 상태이며, 검색 정확도나 처리시간을 측정한 결과와는 구분합니다.

## 관련 글

- [FAISS와 SQLite로 VectorStore를 나눈 이유]({% post_url projects/locallens/2026-02-04-project-locallens-04-vectorstore-sync %})
- [PDF 검색 구조와 VLM 처리]({% post_url projects/locallens/2026-02-13-project-locallens-06-pdf-vlm-retrieval %})
- [모델 비교와 PDF 검색]({% post_url projects/locallens/2026-09-24-project-locallens-source-model-pdf-slides %})
