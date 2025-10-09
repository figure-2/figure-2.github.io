---
title: PJ Parser API 비교 분석
categories:
- 2.PROJECT
- 2-2.FinIQ
tags:
- - parser
  - API
  - 비교
  - 분석
  - 업스테이지
toc: true
date: 2025-10-02
comments: false
mermaid: true
math: true
---
# PJ Parser API 비교 분석

다양한 PJ Parser API의 성능과 특징을 비교 분석해보겠습니다.

##  분석 개요

본 문서에서는 5가지 주요 PJ Parser API의 성능을 비교 분석합니다:
- **Upstage Parser API**
- **Llamaparse Parser API** 
- **OpenAI API**
- **Gemini API**
- **GoogleCloud Document AI API**

각 API는 동일한 PDF 문서를 대상으로 테스트되었으며, 바운딩 박스 정확도와 HTML 추출 품질을 중심으로 평가됩니다.

## 1. Upstage Parser API

<details>
<summary> Upstage Parser API 상세 분석 결과</summary>

###  장점: 객체 탐지 우수
- **일관성 있는 바운딩 박스**: 어노테이션 바운딩 박스 결과가 가장 일관성 있게 탐지되며, HTML 결과에서 테이블 구조를 정확하게 인식

<details>
<summary> 원본 PDF 1페이지 - 바운딩 박스 분석 결과</summary>

![Upstage Parser 바운딩 박스 결과](/assets/images/posts_img/PJ/1_parser_upstage_bbox.PNG)

</details>

<details>
<summary> HTML 추출 결과</summary>

![Upstage Parser HTML 추출 결과 1](/assets/images/posts_img/PJ/1_parser_upstage_bbox_내용.PNG)

![Upstage Parser HTML 추출 결과 2](/assets/images/posts_img/PJ/1_parser_upstage_bbox_내용2.PNG)
</details>

###  한계: 복잡한 구조 처리
- **이중 구조 표 처리**: 복잡한 이중 구조 표에서 바운딩 박스 정확도가 떨어지지만, <span style="color: #e74c3c; font-weight: bold;">HTML 결과에서는 중첩표를 정확하게 탐지</span>

<details>
<summary> 원본 PDF 2페이지 - 중첩표 바운딩 박스 분석</summary>

![Upstage Parser 중첩표 바운딩 박스 결과](/assets/images/posts_img/PJ/1_parser_upstage_bbox2.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![Upstage Parser 중첩표 HTML 추출 결과](/assets/images/posts_img/PJ/1_parser_upstage_bbox2_분석.PNG)
</details>

###  한계: 그래프 인식 부족
- **중첩 표 내 그래프**: 중첩 표 안에 있는 그래프를 인식하지 못함 

<details>
<summary> 원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![Upstage Parser 그래프 바운딩 박스 결과](/assets/images/posts_img/PJ/1_parser_upstage_bbox3.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![Upstage Parser 그래프 HTML 추출 결과](/assets/images/posts_img/PJ/1_parser_upstage_bbox3_분석.PNG)
</details>

</details>

## 2. Llamaparse Parser API

<details>
<summary> Llamaparse Parser API 상세 분석 결과</summary>

###  한계: 기본 표 처리 부족
- **표 처리**: 바운딩 박스가 부정확하며, 단일 표 형식에서도 정확한 추출이 어려움

<details>
<summary> 원본 PDF 1페이지 - 바운딩 박스 분석 결과</summary>

![Llamaparse Parser 바운딩 박스 결과](/assets/images/posts_img/PJ/02_Llamaparse_bbox.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![Llamaparse Parser HTML 추출 결과](/assets/images/posts_img/PJ/02_Llamaparse_bbox_분석.PNG)
</details>

###  혼재된 결과: 중첩표 처리
- **표 처리**: 바운딩 박스가 부정확하며 중첩표 구조에서 내용이 한 칸씩 밀림 <span style="color: #e74c3c; font-weight: bold;">(빨간색 표시)</span><br>
- **긍정적 측면**: 중첩표 구조 자체는 정확하게 인식 <span style="color: #3498db; font-weight: bold;">(파란색 표시)</span>

<details>
<summary> 원본 PDF 2페이지 - 중첩표 바운딩 박스 분석</summary>

![Llamaparse Parser 중첩표 바운딩 박스 결과](/assets/images/posts_img/PJ/02_Llamaparse_bbox2.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![Llamaparse Parser 중첩표 HTML 추출 결과](/assets/images/posts_img/PJ/02_Llamaparse_bbox2_분석.PNG)
</details>
</details>


## 3. OpenAI API

<details>
<summary> OpenAI API 상세 분석 결과</summary>

###  한계: 일관성 부족
- **표 처리**: 바운딩 박스가 부정확하며 프롬프트와 PDF 파일에 따라 결과가 달라짐
- **일관성 문제**: 추출 결과의 일관성과 정확성이 떨어짐
- **긍정적 측면**: HTML 추출에서는 중첩표 테이블을 정확하게 인식하고 복잡한 구조도 잘 처리
<details>
<summary> 원본 PDF 1,2페이지 - 바운딩 박스 분석 결과</summary>

![OpenAI API 바운딩 박스 결과 1](/assets/images/posts_img/PJ/03_openAI_bbox.PNG)
![OpenAI API 바운딩 박스 결과 2](/assets/images/posts_img/PJ/03_openAI_bbox2.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![OpenAI API HTML 추출 결과](/assets/images/posts_img/PJ/03_openAI_bbox_분석.PNG)
</details>

###  장점: 그래프 추출 가능
- **그래프 인식**: 바운딩 박스는 부정확하지만 HTML 추출에서 그래프를 정확하게 탐지하고 추출
<details>
<summary> 원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![OpenAI API 그래프 바운딩 박스 결과](/assets/images/posts_img/PJ/03_openAI_bbox3.PNG)

</details>

<details>
<summary> HTML 추출 결과</summary>

![OpenAI API 그래프 HTML 추출 결과](/assets/images/posts_img/PJ/03_openAI_bbox_분석2.PNG)
</details>

</details>

## 4. Gemini API

<details>
<summary> Gemini API 상세 분석 결과</summary>

OpenAI API와 유사한 결과를 보이며, 차이점은 HTML 추출 결과에서 그래프의 내용을 파악하고 간단한 코멘트를 제공하는 점입니다.

###  한계: 일관성 부족
- **표 처리**: 바운딩 박스가 부정확하며 프롬프트와 PDF 파일에 따라 결과가 달라짐
- **일관성 문제**: 추출 결과의 일관성과 정확성이 떨어짐
- **긍정적 측면**: HTML 추출에서는 중첩표 테이블을 정확하게 인식하고 복잡한 구조도 잘 처리
<details>
<summary> 원본 PDF 1,2페이지 - 바운딩 박스 분석 결과</summary>

![Gemini API 바운딩 박스 결과 1](/assets/images/posts_img/PJ/04_Gemini_bbox.PNG)
![Gemini API 바운딩 박스 결과 2](/assets/images/posts_img/PJ/04_Gemini_bbox2.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![Gemini API HTML 추출 결과](/assets/images/posts_img/PJ/04_Gemini_bbox_분석1.PNG)
</details>

###  장점: 그래프 추출 및 코멘트 제공
- **그래프 인식**: 바운딩 박스는 부정확하지만 HTML 추출에서 그래프를 정확하게 탐지하고 추출
- **추가 기능**: 그래프 내용을 파악하여 간단한 코멘트를 제공하는 차별화된 기능
<details>
<summary> 원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![Gemini API 그래프 바운딩 박스 결과](/assets/images/posts_img/PJ/04_Gemini_bbox3.PNG)

</details>

<details>
<summary> HTML 추출 결과</summary>

![Gemini API 그래프 HTML 추출 결과](/assets/images/posts_img/PJ/04_Gemini_bbox_분석2.PNG)
</details>
</details>


## 5. GoogleCloud Document AI API

<details>
<summary> GoogleCloud Document AI API 상세 분석 결과</summary>

GCP 환경을 이용하는 경우에는 GoogleCloud Document AI API 사용을 고려할 수 있습니다.

###  한계: 구조적 처리 부족
- **표 처리**: 바운딩 박스가 부정확하며, HTML/TXT 추출 결과가 나열식으로 출력됨
- **구조 문제**: 테이블을 별도로 정리하지 않아 청킹과 데이터 중복 문제가 예상됨
<details>
<summary> 원본 PDF 1,2페이지 - 바운딩 박스 분석 결과</summary>

![GoogleCloud Document AI 바운딩 박스 결과 1](/assets/images/posts_img/PJ/05_GoogleCloud_DocumentAI_bbox.PNG)
![GoogleCloud Document AI 바운딩 박스 결과 2](/assets/images/posts_img/PJ/05_GoogleCloud_DocumentAI_bbox2.PNG)
</details>

<details>
<summary> HTML 추출 결과</summary>

![GoogleCloud Document AI HTML 추출 결과](/assets/images/posts_img/PJ/05_GoogleCloud_DocumentAI_bbox_분석.PNG)

</details>

###  혼재된 결과: 그래프 추출
- **그래프 처리**: 바운딩 박스는 부정확하지만 HTML/TXT 추출에서 그래프를 탐지
- **데이터 문제**: 추출된 데이터와 누락된 데이터가 혼재되어 있음
- **구조적 한계**: 나열식 출력으로 인한 청킹과 데이터 중복 문제 지속
<details>
<summary> 원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![GoogleCloud Document AI 그래프 바운딩 박스 결과](/assets/images/posts_img/PJ/05_GoogleCloud_DocumentAI_bbox3.PNG)

</details>

<details>
<summary> HTML 추출 결과</summary>

![GoogleCloud Document AI 그래프 HTML 추출 결과](/assets/images/posts_img/PJ/05_GoogleCloud_DocumentAI_bbox_분석3.PNG)

</details>
</details>

##  종합 비교 분석

###  성능 순위 (바운딩 박스 정확도 기준)

| 순위 | API | 바운딩 박스 정확도 | HTML 추출 품질 | 그래프 인식 | 특별 기능 |
|------|-----|------------------|---------------|------------|----------|
| 1위 | **Upstage Parser** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 일관성 있는 구조 인식 |
| 2위 | **Gemini API** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 그래프 코멘트 제공 |
| 3위 | **OpenAI API** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 그래프 추출 우수 |
| 4위 | **Llamaparse** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 중첩표 구조 인식 |
| 5위 | **GoogleCloud Document AI** | ⭐ | ⭐⭐ | ⭐⭐ | GCP 통합 용이 |

###  사용 권장사항

#### **Upstage Parser API** 
- **추천 상황**: 정확한 바운딩 박스와 일관된 구조 인식이 필요한 경우
- **장점**: 가장 안정적이고 예측 가능한 결과
- **단점**: 그래프 인식 기능 부족

#### **Gemini API**
- **추천 상황**: 그래프 분석과 해석이 필요한 경우
- **장점**: 그래프에 대한 추가 코멘트 제공
- **단점**: 일관성 부족, 프롬프트 의존성

#### **OpenAI API**
- **추천 상황**: 복잡한 구조와 그래프를 모두 처리해야 하는 경우
- **장점**: 그래프 추출 능력 우수
- **단점**: 결과의 일관성 부족

###  공통 한계사항

1. **바운딩 박스 정확도**: 모든 API에서 복잡한 구조에서 정확도 저하
2. **일관성 문제**: LLM 기반 API들은 프롬프트와 파일에 따라 결과 변동
3. **그래프 처리**: 대부분의 API에서 그래프 인식이 상대적으로 약함

###  최적 활용 전략

- **단일 API 사용**: Upstage Parser (안정성 우선)
- **하이브리드 접근**: Upstage Parser + Gemini API (구조 + 그래프 분석)
- **GCP 환경**: GoogleCloud Document AI (통합성 우선)
