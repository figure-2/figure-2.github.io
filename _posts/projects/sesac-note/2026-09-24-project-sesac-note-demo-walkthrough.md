---
title: "16. SeSAC:Note 시연 — 구간별 요약과 AI 튜터"
categories:
- 2.PROJECT
- 2-5. SeSAC-Note
tags:
- SeSAC-Note
- 서비스 시연
toc: true
date: 2026-09-24 00:00:00 +0900
comments: true
mermaid: false
math: false
---

SeSAC:Note에서는 강의 영상을 구간별 요약으로 탐색하고, 이해되지 않는 내용을 AI 튜터에게 질문할 수 있도록 구성했습니다. 이번 글에서는 시연 당시의 화면을 따라 요약 확인, 질문과 응답, 새 영상 분석의 흐름을 정리합니다.

## 구간별 요약에서 질문으로

분석 화면은 왼쪽의 강의 목록, 가운데의 구간별 요약, 오른쪽의 AI 튜터로 나뉩니다. 요약에는 구간의 시간과 제목이 표시되고, 영상을 함께 보면서 내용을 확인할 수 있습니다.

![구간별 요약과 AI 튜터에 입력한 질문](/assets/images/project-demos/sesac-note/demo-0060s.png)
_01:00 · 요약을 읽으면서 AI 튜터에게 질문한 화면_

AI 튜터에는 Flash와 Thinking 모드를 두었습니다. 시연에서는 Thinking 모드를 선택하고 질문에 대한 설명이 오른쪽 패널에 나타나는 흐름을 보여줍니다.

![Thinking 모드에서 설명이 표시된 화면](/assets/images/project-demos/sesac-note/demo-0075s.png)
_01:15 · Thinking 모드의 응답 확인_

## 새 영상 분석의 진행 상태

새 영상을 선택하면 분석 단계와 진행률이 표시됩니다. 결과가 아직 나오지 않았을 때도 작업이 어느 단계에 있는지 확인할 수 있도록 구성했습니다.

![새 영상의 분석 단계와 진행률](/assets/images/project-demos/sesac-note/demo-0105s.png)
_01:45 · 새 영상 분석 중의 진행 화면_

## 요약이 추가되는 분석 화면

분석 화면으로 이동한 뒤 첫 요약이 나타나고, 이후 구간의 요약이 이어서 추가됩니다. 마지막 화면에서는 여러 구간의 제목과 내용을 스크롤하며 확인할 수 있습니다.

![여러 구간의 요약이 생성된 분석 화면](/assets/images/project-demos/sesac-note/demo-0175s.png)
_02:55 · 복수 구간의 요약을 확인하는 화면_

이 시연에서 보여주려던 흐름은 영상 분석 자체에서 끝나지 않습니다. 생성된 요약을 읽고, 영상의 해당 내용을 확인하고, 질문으로 이어지는 학습 동선을 한 화면에 연결하는 것이었습니다. 화면의 시간은 시연 영상의 위치이며 처리시간 측정값은 아닙니다.

## 관련 글

- [서비스 개발 기록 — 파이프라인부터 운영까지]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-development-record %})
- [OCR·요약·RAG 개선 기록]({% post_url projects/sesac-note/2026-09-24-project-sesac-note-optimization-record %})
- [SeSAC:Note 회고]({% post_url projects/sesac-note/2026-02-10-project-sesac-note-10-validation-retrospective %})
