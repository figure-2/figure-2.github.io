---
title: "09. Judge 설계: 요약 품질을 보조 평가하는 방법"
categories:
- 2.PROJECT
- 2-5. SeSAC-Note
tags:
- LLM Judge
- Evaluation
- Summarization
- AI Service
toc: true
date: 2026-01-30 09:00:00 +0900
comments: true
mermaid: true
math: true
---

SeSAC:Note에서 Judge는 정답 판별기가 아니다. Judge는 Summarizer가 만든 노트를 원본 segment와 비교해 groundedness, note quality, multimodal use를 보조적으로 점검하는 gate다.

이 구분이 중요하다. Judge 수치를 잘못 쓰면 생성 품질을 최종 판정한 것처럼 보일 수 있다. 이 글에서는 Summarizer와 Judge를 왜 나눴는지, benchmark 수치를 어떻게 제한해서 해석해야 하는지 정리한다.

## Summarizer와 Judge를 분리한 이유

Summarizer의 역할은 노트를 생성하는 것이다. Judge의 역할은 생성된 노트가 근거에 맞는지 점검하는 것이다. 둘을 분리하면 생성과 평가의 책임이 나뉜다.

```mermaid
flowchart LR
    A[Segments: STT + VLM] --> B[Summarizer]
    B --> C[AI Note]
    A --> D[Judge]
    C --> D
    D --> E[Scores / Feedback]
    E --> F[Revision or Accept]
```

Summarizer가 좋은 문장을 만들더라도 원본 근거와 맞지 않으면 학습 노트로는 위험하다. 반대로 원본 근거만 나열하면 읽기 좋은 노트가 되지 않는다. 그래서 생성과 평가를 분리해 서로 다른 관점으로 보게 했다.

Judge는 기준 점수에 미달한 결과에 대해 피드백을 만들고, 그 피드백을 바탕으로 요약을 보완하거나 재생성하는 loop로 설명할 수 있다. 다만 최종 판정 장치가 아니라 위험을 줄이는 보조 점검 loop로 보는 것이 맞다.

```mermaid
flowchart LR
    A[Summary Draft] --> B[Judge]
    B --> C{Score >= 7.0?}
    C -->|Yes| D[Accept]
    C -->|No| E[Feedback]
    E --> F[Revise / Regenerate]
    F --> B
```

여기서 `7.0`은 프로젝트 기록에 남은 gate 기준이다. 모든 강의에서 이 기준이 절대적인 품질 기준이라는 의미는 아니다.

## 평가 축

Judge는 요약을 하나의 점수로만 보지 않는다. 프로젝트 기록에서는 다음 축이 중요하게 다뤄졌다.

| 평가 축 | 의미 |
| --- | --- |
| groundedness | 생성 내용이 원본 segment 근거와 맞는가 |
| note quality | 학습자가 읽기 좋은 구조인가 |
| multimodal use | 화면 정보와 음성 정보가 함께 반영됐는가 |
| rule compliance | 출력 형식과 작성 규칙을 지켰는가 |

이 축들은 품질 보증이 아니라 점검 기준이다. LLM Judge 역시 LLM이므로 실수할 수 있다. 따라서 Judge 결과는 자동 검사의 한 종류로 보고, 제한된 benchmark 기준으로만 해석한다.

## Judge prompt 개선 흐름

초기 Judge prompt는 평가 조건이 많고 모호하면 모델의 판단이 흔들릴 수 있다. 평가 기준이 복잡하면 token 사용량도 늘고, 응답 시간도 늘어난다.

개발 과정에서는 평가 축을 줄이고, 출력 구조를 단순화하고, retry/self-correction 흐름을 조정하는 방향이 정리됐다.

| 개선 방향 | 목적 |
| --- | --- |
| 평가 축 축소 | 모델이 무엇을 봐야 하는지 명확히 함 |
| 출력 구조 단순화 | parsing과 downstream 처리를 쉽게 함 |
| prompt 압축 | token 사용량과 응답 시간을 줄임 |
| retry 조건 정리 | 불필요한 재시도를 줄임 |

프로젝트 기록 기준으로 별도 실험에서는 Judge 통과율이 80%에서 95%로 개선된 사례가 정리되어 있다. 이 수치는 해당 문서의 실험 조건 기준이며, 전체 서비스 품질을 의미하지 않는다.

요약 쪽에서는 항목마다 참고한 VLM unit id를 evidence로 남기는 방식도 정리됐다. 이렇게 하면 Judge가 생성 문장을 원본 segment와 비교하기 쉬워지고, 나중에 사용자가 "이 요약이 어디서 왔는가"를 추적할 수 있다. 핵심은 환각을 없앴다고 말하는 것이 아니라, 근거 추적 가능성을 높였다고 말하는 것이다.

## 제한된 benchmark 결과

제한된 benchmark에서는 Judge v3 결과가 다음처럼 기록되어 있다.

| 항목 | 제한된 benchmark 기준 기록 |
| --- | --- |
| Judge v3 평균 평가 시간 | 14.7초 |
| Judge v3 평균 토큰 | 14,734 tokens |
| benchmark 통과 | 5/5, 100% |
| v1 대비 평가 시간 | 53.5% 감소 |
| v1 대비 token | 10.0% 절감 |
| Summarizer v3 + Judge v3 | 10.00점, 14.8초, 13,980 tokens |

이 표에서 가장 조심해야 할 숫자는 `100%`와 `53.5%`다. `100%`는 제한된 benchmark 조건에서 5개 설정이 모두 통과했다는 뜻이다. 모든 강의 영상에서 정확하다는 의미가 아니다. `53.5%`는 Judge 평가 단계 기준 시간 감소다. 전체 서비스 처리 시간이 그만큼 줄었다는 의미가 아니다.

## 수치 해석 한계

Judge benchmark는 유용하지만 한계가 있다.

| 한계 | 해석 |
| --- | --- |
| 표본 제한 | 일부 설정과 sample 기준 결과 |
| scoring rule 변화 | 버전 간 점수 절대 비교가 어려울 수 있음 |
| LLM Judge 자체의 불확실성 | 평가 모델도 오류 가능성이 있음 |
| 서비스 전체와 분리 | Judge 단계 수치가 전체 UX를 대표하지 않음 |

따라서 "benchmark 조건에서 평가 단계 효율 개선이 관찰됐다" 정도로 해석하는 것이 안전하다.

## Judge 해석에서 주의할 점

Judge는 품질을 높이는 데 도움을 줄 수 있지만, 생성 품질을 최종 판정하지 않는다. 특히 강의 유형이 달라지면 VLM 입력, STT 품질, Fusion 정확도, 요약 난도도 달라진다.

SeSAC:Note에서 Judge의 역할은 다음으로 제한한다.

1. 생성 요약을 원본 segment와 비교한다.
2. groundedness와 note quality를 자동 점검한다.
3. benchmark 조건에서 prompt 버전 간 차이를 관찰한다.
4. 사람이 확인해야 할 위험을 줄이는 보조 gate로 사용한다.

다음 글에서는 마지막으로 프로젝트를 마치며 남은 한계와 개선 방향을 정리한다.

- 이전 글: [08. QA 설계: 영상 근거 안에서만 답하게 만들기]({% post_url 2026-01-20-project-sesac-note-08-evidence-qa %})
- 다음 글: [10. 프로젝트 회고: 멀티모달 AI 서비스에서 배운 것]({% post_url 2026-02-10-project-sesac-note-10-validation-retrospective %})
