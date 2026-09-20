---
title: PJ Parser API 비교 분석
categories:
- 2.PROJECT
- 2-2. History_Docent
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
## PJ Parser API 비교 분석

같은 PDF를 5가지 Parser API로 처리하고 결과를 비교했다.

## 비교 조건

비교한 API는 다음과 같다.

- **Upstage Parser API**
- **Llamaparse Parser API** 
- **OpenAI API**
- **Gemini API**
- **GoogleCloud Document AI API**

모든 API에 동일한 PDF를 입력하고 바운딩 박스 정확도와 HTML 추출 품질을 중심으로 평가했다. 바운딩 박스는 텍스트나 표 등 문서 요소의 위치와 범위를 나타내는 사각형이다.

## 1. Upstage Parser API

> **장점: 객체 영역을 일관되게 탐지**
> 
> - **일관성 있는 바운딩 박스**: 문서 요소의 영역을 가장 일관되게 탐지했고, HTML 결과에서도 표 구조를 정확하게 인식했다.
> 
> **원본 PDF 1페이지 - 바운딩 박스 분석 결과**
> ![Upstage Parser 바운딩 박스 결과](/assets/images/PJ/1_parser_upstage_bbox.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Upstage Parser HTML 추출 결과 1](/assets/images/PJ/1_parser_upstage_bbox_내용.PNG){: width="80%"}
> ![Upstage Parser HTML 추출 결과 2](/assets/images/PJ/1_parser_upstage_bbox_내용2.PNG){: width="80%"}

---

> **한계: 복잡한 구조 처리**
> 
> - **이중 구조 표 처리**: 복잡한 이중 구조 표에서는 바운딩 박스 정확도가 떨어졌지만, **HTML 결과에서는 중첩표를 정확하게 탐지했다.**
> 
> **원본 PDF 2페이지 - 중첩표 바운딩 박스 분석**
> ![Upstage Parser 중첩표 바운딩 박스 결과](/assets/images/PJ/1_parser_upstage_bbox2.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Upstage Parser 중첩표 HTML 추출 결과](/assets/images/PJ/1_parser_upstage_bbox2_분석.PNG){: width="80%"}

---

> **한계: 그래프 인식 부족**
> 
> - **중첩 표 내 그래프**: 중첩 표 안에 있는 그래프를 인식하지 못함
> 
> **원본 PDF 3페이지 - 그래프 바운딩 박스 분석**
> ![Upstage Parser 그래프 바운딩 박스 결과](/assets/images/PJ/1_parser_upstage_bbox3.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Upstage Parser 그래프 HTML 추출 결과](/assets/images/PJ/1_parser_upstage_bbox3_분석.PNG){: width="80%"}

## 2. Llamaparse Parser API

> **한계: 기본 표 처리 부족**
> 
> - **표 처리**: 바운딩 박스가 부정확하며, 단일 표 형식에서도 정확한 추출이 어려움
> 
> **원본 PDF 1페이지 - 바운딩 박스 분석 결과**
> ![Llamaparse Parser 바운딩 박스 결과](/assets/images/PJ/02_Llamaparse_bbox.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Llamaparse Parser HTML 추출 결과](/assets/images/PJ/02_Llamaparse_bbox_분석.PNG){: width="80%"}

---

> **중첩표 처리 결과**
> 
> - **표 처리**: 바운딩 박스가 부정확하며 중첩표 구조에서 내용이 한 칸씩 밀림 **(빨간색 표시)**
> - **확인한 장점**: 중첩표 구조 자체는 정확하게 인식 **(파란색 표시)**
> 
> **원본 PDF 2페이지 - 중첩표 바운딩 박스 분석**
> ![Llamaparse Parser 중첩표 바운딩 박스 결과](/assets/images/PJ/02_Llamaparse_bbox2.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Llamaparse Parser 중첩표 HTML 추출 결과](/assets/images/PJ/02_Llamaparse_bbox2_분석.PNG){: width="80%"}

## 3. OpenAI API

> **한계: 일관성 부족**
> 
> - **표 처리**: 바운딩 박스가 부정확하며 프롬프트와 PDF 파일에 따라 결과가 달라짐
> - **일관성 문제**: 추출 결과의 일관성과 정확성이 떨어짐
> - **확인한 장점**: HTML 결과에서는 중첩표를 정확하게 인식하고 복잡한 구조도 잘 처리했다.
> 
> **원본 PDF 1,2페이지 - 바운딩 박스 분석 결과**
> ![OpenAI API 바운딩 박스 결과 1](/assets/images/PJ/03_openAI_bbox.PNG){: width="80%"}
> ![OpenAI API 바운딩 박스 결과 2](/assets/images/PJ/03_openAI_bbox2.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![OpenAI API HTML 추출 결과](/assets/images/PJ/03_openAI_bbox_분석.PNG){: width="80%"}

---

> **장점: 그래프 추출 가능**
> 
> - **그래프 인식**: 바운딩 박스는 부정확했지만 HTML 결과에서는 그래프를 정확하게 탐지하고 추출했다.
> 
> **원본 PDF 3페이지 - 그래프 바운딩 박스 분석**
> ![OpenAI API 그래프 바운딩 박스 결과](/assets/images/PJ/03_openAI_bbox3.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![OpenAI API 그래프 HTML 추출 결과](/assets/images/PJ/03_openAI_bbox_분석2.PNG){: width="80%"}

## 4. Gemini API

Gemini API의 결과는 OpenAI API와 유사했다. 차이는 HTML 추출 과정에서 그래프의 내용을 파악해 간단한 설명을 함께 제공한다는 점이다.

> **한계: 일관성 부족**
> 
> - **표 처리**: 바운딩 박스가 부정확하며 프롬프트와 PDF 파일에 따라 결과가 달라짐
> - **일관성 문제**: 추출 결과의 일관성과 정확성이 떨어짐
> - **확인한 장점**: HTML 결과에서는 중첩표를 정확하게 인식하고 복잡한 구조도 잘 처리했다.
> 
> **원본 PDF 1,2페이지 - 바운딩 박스 분석 결과**
> ![Gemini API 바운딩 박스 결과 1](/assets/images/PJ/04_Gemini_bbox.PNG){: width="80%"}
> ![Gemini API 바운딩 박스 결과 2](/assets/images/PJ/04_Gemini_bbox2.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Gemini API HTML 추출 결과](/assets/images/PJ/04_Gemini_bbox_분석1.PNG){: width="80%"}

---

> **장점: 그래프 추출과 내용 설명**
> 
> - **그래프 인식**: 바운딩 박스는 부정확했지만 HTML 결과에서는 그래프를 정확하게 탐지하고 추출했다.
> - **추가 기능**: 그래프의 내용을 파악해 간단한 설명을 함께 제공했다.
> 
> **원본 PDF 3페이지 - 그래프 바운딩 박스 분석**
> ![Gemini API 그래프 바운딩 박스 결과](/assets/images/PJ/04_Gemini_bbox3.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![Gemini API 그래프 HTML 추출 결과](/assets/images/PJ/04_Gemini_bbox_분석2.PNG){: width="80%"}

## 5. GoogleCloud Document AI API

GoogleCloud Document AI API는 GCP 환경과 함께 사용할 수 있다는 장점이 있다.

> **한계: 구조적 처리 부족**
> 
> - **표 처리**: 바운딩 박스가 부정확하며, HTML/TXT 추출 결과가 나열식으로 출력됨
> - **구조 문제**: 테이블을 별도로 정리하지 않아 청킹과 데이터 중복 문제가 예상됨
> 
> **원본 PDF 1,2페이지 - 바운딩 박스 분석 결과**
> ![GoogleCloud Document AI 바운딩 박스 결과 1](/assets/images/PJ/05_GoogleCloud_DocumentAI_bbox.PNG){: width="80%"}
> ![GoogleCloud Document AI 바운딩 박스 결과 2](/assets/images/PJ/05_GoogleCloud_DocumentAI_bbox2.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![GoogleCloud Document AI HTML 추출 결과](/assets/images/PJ/05_GoogleCloud_DocumentAI_bbox_분석.PNG){: width="80%"}

---

> **그래프 추출 결과**
> 
> - **그래프 처리**: 바운딩 박스는 부정확하지만 HTML/TXT 추출에서 그래프를 탐지
> - **데이터 문제**: 추출된 데이터와 누락된 데이터가 혼재되어 있음
> - **구조적 한계**: 나열식 출력으로 인한 청킹과 데이터 중복 문제 지속
> 
> **원본 PDF 3페이지 - 그래프 바운딩 박스 분석**
> ![GoogleCloud Document AI 그래프 바운딩 박스 결과](/assets/images/PJ/05_GoogleCloud_DocumentAI_bbox3.PNG){: width="80%"}
> 
> **HTML 추출 결과**
> ![GoogleCloud Document AI 그래프 HTML 추출 결과](/assets/images/PJ/05_GoogleCloud_DocumentAI_bbox_분석3.PNG){: width="80%"}

## 종합 비교 분석

### 성능 순위 (바운딩 박스 정확도 기준)

| 순위 | API | 바운딩 박스 정확도 | HTML 추출 품질 | 그래프 인식 | 특별 기능 |
|------|-----|------------------|---------------|------------|----------|
| 1위 | **Upstage Parser** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 일관성 있는 구조 인식 |
| 2위 | **Gemini API** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 그래프 코멘트 제공 |
| 3위 | **OpenAI API** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 그래프 추출 우수 |
| 4위 | **Llamaparse** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 중첩표 구조 인식 |
| 5위 | **GoogleCloud Document AI** | ⭐ | ⭐⭐ | ⭐⭐ | GCP 통합 용이 |

### 조건별 선택 기준

#### **Upstage Parser API** 

- **추천 상황**: 정확한 바운딩 박스와 일관된 구조 인식이 필요한 경우
- **장점**: 가장 안정적이고 예측 가능한 결과
- **단점**: 그래프 인식 기능 부족

#### **Gemini API**

- **추천 상황**: 그래프 분석과 해석이 필요한 경우
- **장점**: 그래프 내용에 대한 추가 설명 제공
- **단점**: 일관성 부족, 프롬프트 의존성

#### **OpenAI API**

- **추천 상황**: 복잡한 구조와 그래프를 모두 처리해야 하는 경우
- **장점**: 그래프 추출 능력 우수
- **단점**: 결과의 일관성 부족

### 공통으로 확인한 한계

1. **바운딩 박스 정확도**: 복잡한 구조에서는 모든 API의 정확도가 떨어졌다.
2. **일관성 문제**: LLM 기반 API들은 프롬프트와 파일에 따라 결과 변동
3. **그래프 처리**: 대부분의 API에서 그래프 인식이 상대적으로 약함

### 단독 사용과 조합

- **단일 API 사용**: Upstage Parser (안정성 우선)
- **API 조합**: Upstage Parser + Gemini API (구조 + 그래프 분석)
- **GCP 환경**: GoogleCloud Document AI (통합성 우선)
