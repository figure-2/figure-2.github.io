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

## 분석 개요

본 문서에서는 5가지 주요 PJ Parser API의 성능을 비교 분석합니다:

- **Upstage Parser API**
- **Llamaparse Parser API**
- **OpenAI API**
- **Gemini API**
- **GoogleCloud Document AI API**

각 API는 동일한 PDF 문서를 대상으로 테스트되었으며, 바운딩 박스 정확도와 HTML 추출 품질을 중심으로 평가됩니다.

---

## 1. Upstage Parser API

<details>
<summary>Upstage Parser API 상세 분석 결과</summary>

### 장점: 객체 탐지 우수

- **일관성 있는 바운딩 박스**: 어노테이션 바운딩 박스 결과가 가장 일관성 있게 탐지되며, HTML 결과에서 테이블 구조를 정확하게 인식

<details>
<summary>원본 PDF 1페이지 - 바운딩 박스 분석 결과</summary>

![Upstage Parser 바운딩 박스 결과](/assets/images/PJ/1_parser_upstage_bbox.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Upstage Parser HTML 추출 결과 1](/assets/images/PJ/1_parser_upstage_bbox_내용.PNG)  
![Upstage Parser HTML 추출 결과 2](/assets/images/PJ/1_parser_upstage_bbox_내용2.PNG)

</details>

### 한계: 복잡한 구조 처리

- **이중 구조 표 처리**: 복잡한 이중 구조 표에서 바운딩 박스 정확도가 떨어지지만, <span style="color: #e74c3c; font-weight: bold;">HTML 결과에서는 중첩표를 정확하게 탐지</span>

<details>
<summary>원본 PDF 2페이지 - 중첩표 바운딩 박스 분석</summary>

![Upstage Parser 중첩표 바운딩 박스 결과](/assets/images/PJ/1_parser_upstage_bbox2.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Upstage Parser 중첩표 HTML 추출 결과](/assets/images/PJ/1_parser_upstage_bbox2_분석.PNG)

</details>

### 한계: 그래프 인식 부족

- **중첩 표 내 그래프**: 중첩 표 안에 있는 그래프를 인식하지 못함

<details>
<summary>원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![Upstage Parser 그래프 바운딩 박스 결과](/assets/images/PJ/1_parser_upstage_bbox3.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Upstage Parser 그래프 HTML 추출 결과](/assets/images/PJ/1_parser_upstage_bbox3_분석.PNG)

</details>

</details>

---

## 2. Llamaparse Parser API

<details>
<summary>Llamaparse Parser API 상세 분석 결과</summary>

### 한계: 기본 표 처리 부족

- **표 처리**: 바운딩 박스가 부정확하며, 단일 표 형식에서도 정확한 추출이 어려움

<details>
<summary>원본 PDF 1페이지 - 바운딩 박스 분석 결과</summary>

![Llamaparse Parser 바운딩 박스 결과](/assets/images/PJ/02_Llamaparse_bbox.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Llamaparse Parser HTML 추출 결과](/assets/images/PJ/02_Llamaparse_bbox_분석.PNG)

</details>

### 혼재된 결과: 중첩표 처리

- **표 처리**: 바운딩 박스가 부정확하며 중첩표 구조에서 내용이 한 칸씩 밀림 <span style="color: #e74c3c; font-weight: bold;">(빨간색 표시)</span>  
- **긍정적 측면**: 중첩표 구조 자체는 정확하게 인식 <span style="color: #3498db; font-weight: bold;">(파란색 표시)</span>

<details>
<summary>원본 PDF 2페이지 - 중첩표 바운딩 박스 분석</summary>

![Llamaparse Parser 중첩표 바운딩 박스 결과](/assets/images/PJ/02_Llamaparse_bbox2.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Llamaparse Parser 중첩표 HTML 추출 결과](/assets/images/PJ/02_Llamaparse_bbox2_분석.PNG)

</details>

</details>

---

## 3. OpenAI API

<details>
<summary>OpenAI API 상세 분석 결과</summary>

### 한계: 일관성 부족

- **표 처리**: 바운딩 박스가 부정확하며 프롬프트와 PDF 파일에 따라 결과가 달라짐
- **일관성 문제**: 추출 결과의 일관성과 정확성이 떨어짐
- **긍정적 측면**: HTML 추출에서는 중첩표 테이블을 정확하게 인식하고 복잡한 구조도 잘 처리

<details>
<summary>원본 PDF 1,2페이지 - 바운딩 박스 분석 결과</summary>

![OpenAI API 바운딩 박스 결과 1](/assets/images/PJ/03_openAI_bbox.PNG)  
![OpenAI API 바운딩 박스 결과 2](/assets/images/PJ/03_openAI_bbox2.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![OpenAI API HTML 추출 결과](/assets/images/PJ/03_openAI_bbox_분석.PNG)

</details>

### 장점: 그래프 추출 가능

- **그래프 인식**: 바운딩 박스는 부정확하지만 HTML 추출에서 그래프를 정확하게 탐지하고 추출

<details>
<summary>원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![OpenAI API 그래프 바운딩 박스 결과](/assets/images/PJ/03_openAI_bbox3.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![OpenAI API 그래프 HTML 추출 결과](/assets/images/PJ/03_openAI_bbox_분석2.PNG)

</details>

</details>

---

## 4. Gemini API

<details>
<summary>Gemini API 상세 분석 결과</summary>

### 한계: 일관성 부족

- **표 처리**: 바운딩 박스가 부정확하며 프롬프트와 PDF 파일에 따라 결과가 달라짐
- **일관성 문제**: 추출 결과의 일관성과 정확성이 떨어짐
- **긍정적 측면**: HTML 추출에서는 중첩표 테이블을 정확하게 인식하고 복잡한 구조도 잘 처리

<details>
<summary>원본 PDF 1,2페이지 - 바운딩 박스 분석 결과</summary>

![Gemini API 바운딩 박스 결과 1](/assets/images/PJ/04_Gemini_bbox.PNG)  
![Gemini API 바운딩 박스 결과 2](/assets/images/PJ/04_Gemini_bbox2.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Gemini API HTML 추출 결과](/assets/images/PJ/04_Gemini_bbox_분석1.PNG)

</details>

### 장점: 그래프 추출 및 코멘트 제공

- **그래프 인식**: 바운딩 박스는 부정확하지만 HTML 추출에서 그래프를 정확하게 탐지하고 추출
- **추가 기능**: 그래프 내용을 파악하여 간단한 코멘트를 제공하는 차별화된 기능

<details>
<summary>원본 PDF 3페이지 - 그래프 바운딩 박스 분석</summary>

![Gemini API 그래프 바운딩 박스 결과](/assets/images/PJ/04_Gemini_bbox3.PNG)

</details>

<details>
<summary>HTML 추출 결과</summary>

![Gemini API 그래프 HTML 추출 결과](/assets/images/PJ/04_Gemini_bbox_분석2.PNG)

</details>

</details>

---

## 5. GoogleCloud Document AI API

<details>
<summary>GoogleCloud Document AI API 상세 분석 결과</summary>

### 한계: 구조적 처리 부족

- **표 처리**: 바운딩 박스가 부정확하며, HTML/TXT 추출 결과가 나열식으로 출력됨
- **구조 문제**: 테이블을 별도로 정리하지 않아 청킹과 데이터 중복 문제가 예상
