---
title: "모델 변경과 프롬프트 변화"
categories:
- 3.STUDY
- 3-7.AI_ENGINEERING
tags:
- study
- prompt-engineering
- context-engineering
- model-change
toc: true
date: 2026-05-31 18:00:00 +0900
comments: false
mermaid: true
math: true
---
# 모델 변경과 프롬프트 변화

> **한줄 정의**
> 모델이 바뀌면 prompt도 바뀐다. 강한 모델일수록 과정 지시보다 목표, 제약, 성공 기준, 출력 계약이 중요해진다.

## 기준

이 글의 모델명, 가격, parameter, benchmark 관련 내용은 원본 학습 노트 기준이다. 최신 공식 사양으로 단정하지 않는다. 실제 발행 전에는 공식 문서를 다시 확인해야 한다.

## 핵심 변화

원본 노트의 중심 주장은 다음이다.

```text
예전 prompt:
  과정을 자세히 지시

새 prompt:
  목적지, 제약, 성공 기준, 출력 계약을 정의
```

모델이 충분히 강해지면 "단계별로 생각해", "전문가처럼", "항상 먼저 분석해" 같은 문장은 도움이 아니라 noise가 될 수 있다.

## Control Surface가 바뀐다

| 과거 제어 | 현재 제어 |
| --- | --- |
| prompt 문장으로 노력량 조절 | reasoning effort, thinking mode 같은 parameter |
| 역할극으로 전문성 유도 | task와 success criteria 명시 |
| few-shot으로 형식 강제 | output contract와 schema |
| 절차 나열 | 목표와 제약 정의 |
| 반복 제약 | 한 번 명확한 정책 |

사람이 설계할 것은 모델 내부 추론 단계가 아니라 model 바깥의 계약이다.

## 원본 노트의 모델 관찰

| 항목 | 정리 |
| --- | --- |
| Opus 4.8 | effort가 핵심 dial, adaptive thinking, literalness 강화, tool 호출 누락 개선 |
| GPT-5.5 | 결과 중심 prompt, 과정 명세 축소, 성공 기준과 출력 계약 강조 |

이 항목은 원본 학습 노트의 비교를 보존한 것이다. 최신 사실 여부는 별도 확인 대상이다.

## 버릴 습관

| 습관 | 문제 |
| --- | --- |
| "단계별로 생각해" | 이미 구조적 추론을 하는 모델에 noise가 될 수 있음 |
| 역할극 도입부 | task 도달 전 해석할 모호함 증가 |
| 통제 가치 없는 절차 나열 | search space를 좁힘 |
| 하드코딩된 tool 순서 | model의 더 나은 선택을 막을 수 있음 |
| 같은 제약 반복 | prompt 길이와 혼란 증가 |
| 불필요한 few-shot | 스타일에 anchor되어 다양성 감소 |

## 더해야 할 것

| 항목 | 예 |
| --- | --- |
| 목표 | 무엇을 달성해야 하는가 |
| 제약 | 바꾸면 안 되는 것, 보안 경계, 비용 한도 |
| 성공 기준 | 무엇이면 완료인가 |
| 출력 계약 | field, type, 길이, format |
| tool boundary | 언제 tool을 쓰고, 언제 멈추는가 |
| failure handling | 막히면 어떻게 보고할 것인가 |

## Old vs New

```text
Old:
당신은 B2B copywriter 전문가입니다.
단계별로 생각하고, 먼저 분석하고, 다음으로...

New:
마케팅 VP에게 보낼 cold email을 작성하라.
목표: 20분 discovery call 예약.
길이: 120단어 이내.
포함: 구체적 pain point 1개.
```

새 prompt는 짧지만 더 엄격하다. 역할극을 줄이고 성공 기준을 늘린다.

## 모델 변경 시 migration 순서

| 순서 | 작업 |
| --- | --- |
| 1 | 기존 prompt에서 주문, 역할극, 절차 반복 제거 |
| 2 | 목표, 제약, 성공 기준을 첫 문단에 배치 |
| 3 | 출력 contract를 구조화 |
| 4 | tool boundary와 approval condition 보존 |
| 5 | reasoning effort나 thinking parameter 재측정 |
| 6 | staging에서 이전 출력과 나란히 비교 |
| 7 | regression set으로 품질 하락 여부 확인 |

## 남겨야 할 제약

과정 명세는 줄여도 안전 경계는 줄이면 안 된다.

| 남길 것 | 이유 |
| --- | --- |
| secret 금지 | 보안 |
| destructive action 승인 | 복구 불가능성 |
| 비용 한도 | 운영 비용 |
| tool 권한 | 외부 상태 변경 |
| 실패 보고 | 거짓 성공 방지 |
| 출력 schema | downstream parsing |

## 내 기준

강한 모델을 쓰는 prompt는 더 화려해지지 않는다. 더 명확해진다.

```text
목표
  -> 제약
  -> 성공 기준
  -> 출력 계약
  -> 도구 경계
```

이 구조가 없으면 모델만 바꿔도 결과가 흔들린다.

## 관련 글

- [Agent Engineering]({% post_url 2026-05-23-study-agent-engineering %})
- [AX 시대를 위한 DX]({% post_url 2026-05-16-study-dx-for-ax %})
