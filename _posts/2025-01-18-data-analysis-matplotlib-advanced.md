---
title: Matplotlib 고급 시각화 - 데이터 분석을 위한 전문 기법
categories:
- 1.TIL
- 1-7.DATA_ANALYSIS
tags:
- matplotlib
- 고급시각화
- seaborn
- plotly
- 데이터시각화
- 분석도구
toc: true
date: 2023-09-24 09:00:00 +0900
comments: false
mermaid: true
math: true
---
# Matplotlib 고급 시각화 - 데이터 분석을 위한 전문 기법

## 개요

데이터 분석을 위한 고급 시각화 기법을 학습합니다:

- **시각화 라이브러리**: Matplotlib, Seaborn, Plotly 비교
- **고급 기법**: 분포, 관계, 시간, 비율 시각화
- **스타일링**: 한글 폰트, 테마, 레이아웃 설정
- **실무 활용**: 분석 과정과 보고서용 시각화

## 1. 시각화 라이브러리 개요

### 1-1. 주요 시각화 라이브러리

```python
print("=== 주요 시각화 라이브러리 ===")

print("1. Matplotlib")
print("   - MATLAB 기반의 기본 라이브러리")
print("   - 사용법이 복잡하지만 세밀한 제어 가능")
print("   - 판다스에 포함되어 빠른 코딩 가능")
print("   - 정적 시각화의 한계")

print("\n2. Seaborn")
print("   - Matplotlib 기반의 고급 라이브러리")
print("   - 통계적 시각화에 특화")
print("   - 아름다운 기본 스타일")
print("   - 복잡한 그래프를 간단하게 생성")

print("\n3. Plotly")
print("   - 인터랙티브 시각화")
print("   - 웹 기반 대시보드")
print("   - 3D 시각화 지원")
print("   - 실시간 데이터 업데이트")

print("\n4. Tableau")
print("   - 전문적인 비즈니스 인텔리전스")
print("   - 드래그 앤 드롭 인터페이스")
print("   - 고급 분석 기능")
print("   - 엔터프라이즈급 도구")
```

### 1-2. Matplotlib의 특징과 활용

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=== Matplotlib 활용 시나리오 ===")
print("✅ 적합한 경우:")
print("   - 빠른 데이터 탐색")
print("   - 분석 과정에서의 임시 시각화")
print("   - 커스터마이징이 필요한 경우")
print("   - 프로그래밍 방식의 시각화")

print("\n❌ 부적합한 경우:")
print("   - 최종 보고서용 시각화")
print("   - 인터랙티브 대시보드")
print("   - 복잡한 통계 시각화")
print("   - 실시간 데이터 모니터링")
```

## 2. Matplotlib 기본 구조

### 2-1. Figure와 Axes 이해

```python
# Figure와 Axes의 관계
fig, ax = plt.subplots(figsize=(10, 6))

# Figure: 전체 그래프 컨테이너
print(f"Figure 크기: {fig.get_size_inches()}")
print(f"Figure DPI: {fig.dpi}")

# Axes: 실제 그래프가 그려지는 좌표 평면
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, label='sin(x)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Figure와 Axes 예제')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2-2. 다중 서브플롯

```python
# 복잡한 서브플롯 구성
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 각 서브플롯에 다른 그래프
x = np.linspace(0, 10, 100)

# 1. 선 그래프
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine Wave')

# 2. 산점도
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)
axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6)
axes[0, 1].set_title('Scatter Plot')

# 3. 히스토그램
data_hist = np.random.normal(0, 1, 1000)
axes[0, 2].hist(data_hist, bins=30, alpha=0.7)
axes[0, 2].set_title('Histogram')

# 4. 박스플롯
data_box = [np.random.normal(0, 1, 100), 
           np.random.normal(2, 1, 100)]
axes[1, 0].boxplot(data_box, labels=['Group A', 'Group B'])
axes[1, 0].set_title('Box Plot')

# 5. 막대 그래프
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 1].bar(categories, values, color=['red', 'green', 'blue', 'orange'])
axes[1, 1].set_title('Bar Chart')

# 6. 파이 차트
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
axes[1, 2].pie(sizes, labels=labels, autopct='%1.1f%%')
axes[1, 2].set_title('Pie Chart')

plt.tight_layout()
plt.show()
```

## 3. 고급 시각화 기법

### 3-1. 분포 시각화

```python
# 다양한 분포 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 히스토그램
np.random.seed(42)
data1 = np.random.normal(100, 15, 1000)
axes[0, 0].hist(data1, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 2. 커널 밀도 추정 (KDE)
from scipy import stats
axes[0, 1].hist(data1, bins=30, density=True, alpha=0.7, color='lightgreen')
kde = stats.gaussian_kde(data1)
x_kde = np.linspace(data1.min(), data1.max(), 100)
axes[0, 1].plot(x_kde, kde(x_kde), 'r-', linewidth=2)
axes[0, 1].set_title('Histogram with KDE')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')

# 3. 박스플롯
data_box = [np.random.normal(0, 1, 100), 
           np.random.normal(2, 1, 100),
           np.random.normal(-1, 1.5, 100)]
axes[1, 0].boxplot(data_box, labels=['Group A', 'Group B', 'Group C'])
axes[1, 0].set_title('Multiple Box Plots')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3)

# 4. 바이올린 플롯 (Seaborn 스타일)
# Matplotlib로 바이올린 플롯 구현
parts = axes[1, 1].violinplot(data_box, positions=[1, 2, 3], showmeans=True)
axes[1, 1].set_title('Violin Plot')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_xticks([1, 2, 3])
axes[1, 1].set_xticklabels(['Group A', 'Group B', 'Group C'])

plt.tight_layout()
plt.show()
```

### 3-2. 관계 시각화

```python
# 관계 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 산점도
np.random.seed(42)
x = np.random.randn(200)
y = 2 * x + np.random.randn(200) * 0.5
colors = np.random.rand(200)
sizes = 1000 * np.random.rand(200)

axes[0, 0].scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
axes[0, 0].set_title('Scatter Plot with Color and Size')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# 2. 상관관계 히트맵
import seaborn as sns
data_corr = np.random.randn(100, 5)
df_corr = pd.DataFrame(data_corr, columns=['A', 'B', 'C', 'D', 'E'])
correlation_matrix = df_corr.corr()

im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
axes[0, 1].set_title('Correlation Heatmap')
axes[0, 1].set_xticks(range(len(correlation_matrix.columns)))
axes[0, 1].set_yticks(range(len(correlation_matrix.columns)))
axes[0, 1].set_xticklabels(correlation_matrix.columns)
axes[0, 1].set_yticklabels(correlation_matrix.columns)

# 컬러바 추가
plt.colorbar(im, ax=axes[0, 1])

# 3. 산점도 매트릭스 (일부)
x1 = np.random.randn(100)
x2 = np.random.randn(100)
x3 = np.random.randn(100)

# 상관관계가 있는 데이터 생성
x2 = 0.5 * x1 + 0.5 * np.random.randn(100)
x3 = -0.3 * x1 + 0.7 * np.random.randn(100)

axes[1, 0].scatter(x1, x2, alpha=0.6, color='blue')
axes[1, 0].set_title('X1 vs X2')
axes[1, 0].set_xlabel('X1')
axes[1, 0].set_ylabel('X2')
axes[1, 0].grid(True, alpha=0.3)

# 4. 회귀선이 있는 산점도
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 선형 회귀
lr = LinearRegression()
lr.fit(x1.reshape(-1, 1), x2)

# 다항 회귀
poly = PolynomialFeatures(degree=2)
x1_poly = poly.fit_transform(x1.reshape(-1, 1))
lr_poly = LinearRegression()
lr_poly.fit(x1_poly, x2)

# 시각화
axes[1, 1].scatter(x1, x2, alpha=0.6, color='blue', label='Data')

# 선형 회귀선
x_line = np.linspace(x1.min(), x1.max(), 100)
y_line = lr.predict(x_line.reshape(-1, 1))
axes[1, 1].plot(x_line, y_line, 'r-', linewidth=2, label='Linear')

# 다항 회귀선
x_line_poly = poly.transform(x_line.reshape(-1, 1))
y_line_poly = lr_poly.predict(x_line_poly)
axes[1, 1].plot(x_line, y_line_poly, 'g--', linewidth=2, label='Polynomial')

axes[1, 1].set_title('Regression Lines')
axes[1, 1].set_xlabel('X1')
axes[1, 1].set_ylabel('X2')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3-3. 시간 시각화

```python
# 시간 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 라인 차트
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

axes[0, 0].plot(dates, values, marker='o', linewidth=2, markersize=4)
axes[0, 0].set_title('Time Series Line Chart')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Value')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. 다중 라인 차트
values2 = np.cumsum(np.random.randn(100)) + 80
values3 = np.cumsum(np.random.randn(100)) + 120

axes[0, 1].plot(dates, values, label='Series A', linewidth=2)
axes[0, 1].plot(dates, values2, label='Series B', linewidth=2)
axes[0, 1].plot(dates, values3, label='Series C', linewidth=2)
axes[0, 1].set_title('Multiple Time Series')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# 3. 막대 그래프 (월별 집계)
monthly_data = pd.DataFrame({
    'date': dates,
    'value': values
})
monthly_data['month'] = monthly_data['date'].dt.to_period('M')
monthly_summary = monthly_data.groupby('month')['value'].mean()

axes[1, 0].bar(range(len(monthly_summary)), monthly_summary.values, 
               color='steelblue', alpha=0.7)
axes[1, 0].set_title('Monthly Average')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Average Value')
axes[1, 0].set_xticks(range(len(monthly_summary)))
axes[1, 0].set_xticklabels([str(month) for month in monthly_summary.index], rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# 4. 영역 차트
axes[1, 1].fill_between(dates, values, alpha=0.3, color='blue', label='Series A')
axes[1, 1].fill_between(dates, values2, alpha=0.3, color='red', label='Series B')
axes[1, 1].plot(dates, values, color='blue', linewidth=1)
axes[1, 1].plot(dates, values2, color='red', linewidth=1)
axes[1, 1].set_title('Area Chart')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Value')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3-4. 비율 시각화

```python
# 비율 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 파이 차트
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

wedges, texts, autotexts = axes[0, 0].pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Pie Chart')

# 2. 도넛 차트
axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, pctdistance=0.85)
# 도넛 모양 만들기
centre_circle = plt.Circle((0,0), 0.70, fc='white')
axes[0, 1].add_artist(centre_circle)
axes[0, 1].set_title('Donut Chart')

# 3. 누적 막대 그래프
categories = ['A', 'B', 'C', 'D']
values1 = [20, 35, 30, 35]
values2 = [25, 32, 34, 20]
values3 = [30, 25, 20, 25]

x = np.arange(len(categories))
width = 0.25

axes[1, 0].bar(x - width, values1, width, label='Series 1', color='skyblue')
axes[1, 0].bar(x, values2, width, label='Series 2', color='lightgreen')
axes[1, 0].bar(x + width, values3, width, label='Series 3', color='lightcoral')

axes[1, 0].set_title('Grouped Bar Chart')
axes[1, 0].set_xlabel('Categories')
axes[1, 0].set_ylabel('Values')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 누적 영역 차트
x_stack = np.arange(len(categories))
axes[1, 1].bar(x_stack, values1, label='Series 1', color='skyblue')
axes[1, 1].bar(x_stack, values2, bottom=values1, label='Series 2', color='lightgreen')
axes[1, 1].bar(x_stack, values3, bottom=np.array(values1) + np.array(values2), 
               label='Series 3', color='lightcoral')

axes[1, 1].set_title('Stacked Bar Chart')
axes[1, 1].set_xlabel('Categories')
axes[1, 1].set_ylabel('Values')
axes[1, 1].set_xticks(x_stack)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 4. 고급 설정

### 4-1. 한글 폰트 설정

```python
import matplotlib.font_manager as fm

# 시스템에 설치된 한글 폰트 확인
font_list = [f.name for f in fm.fontManager.ttflist if '한글' in f.name or 'Korean' in f.name or 'Malgun' in f.name or 'Nanum' in f.name]
print("사용 가능한 한글 폰트:")
for font in font_list[:10]:  # 상위 10개만 표시
    print(f"- {font}")

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정 함수"""
    try:
        # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        try:
            # Mac
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            try:
                # Linux
                plt.rcParams['font.family'] = 'NanumGothic'
            except:
                print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 폰트 설정 적용
set_korean_font()

# 테스트 그래프
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, label='사인 함수')
ax.set_xlabel('X축')
ax.set_ylabel('Y축')
ax.set_title('한글 폰트 테스트')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4-2. 스타일 설정

```python
# 다양한 스타일 비교
styles = ['default', 'seaborn-v0_8', 'ggplot', 'fivethirtyeight', 'bmh']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

for i, style in enumerate(styles):
    if i < len(axes):
        plt.style.use(style)
        
        axes[i].plot(x, y1, label='sin(x)', linewidth=2)
        axes[i].plot(x, y2, label='cos(x)', linewidth=2)
        axes[i].set_title(f'Style: {style}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

# 마지막 subplot 제거
if len(styles) < len(axes):
    axes[-1].remove()

plt.tight_layout()
plt.show()

# 기본 스타일로 복원
plt.style.use('default')
```

### 4-3. 커스텀 스타일

```python
# 커스텀 스타일 설정
def set_custom_style():
    """커스텀 스타일 설정"""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

# 커스텀 스타일 적용
set_custom_style()

# 테스트 그래프
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, label='sin(x)', linewidth=2, color='steelblue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Custom Style Example')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 5. 실무 활용 팁

### 5-1. 분석 과정에서의 활용

```python
# 빠른 데이터 탐색을 위한 함수
def quick_explore(data, title="Data Exploration"):
    """빠른 데이터 탐색"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 히스토그램
    axes[0, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # 박스플롯
    axes[0, 1].boxplot(data)
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Value')
    
    # Q-Q 플롯
    from scipy import stats
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # 통계 요약
    axes[1, 1].axis('off')
    stats_text = f"""
    통계 요약:
    
    평균: {np.mean(data):.2f}
    중앙값: {np.median(data):.2f}
    표준편차: {np.std(data):.2f}
    최소값: {np.min(data):.2f}
    최대값: {np.max(data):.2f}
    왜도: {stats.skew(data):.2f}
    첨도: {stats.kurtosis(data):.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 샘플 데이터로 테스트
np.random.seed(42)
sample_data = np.random.normal(100, 15, 1000)
quick_explore(sample_data, "Sample Data Exploration")
```

### 5-2. 성능 최적화

```python
# 대용량 데이터 시각화 최적화
def optimize_large_data_plot(x, y, sample_size=10000):
    """대용량 데이터 시각화 최적화"""
    
    # 데이터가 너무 크면 샘플링
    if len(x) > sample_size:
        indices = np.random.choice(len(x), sample_size, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        print(f"데이터 크기: {len(x)} → {len(x_sample)} (샘플링)")
    else:
        x_sample = x
        y_sample = y
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 일반 산점도
    axes[0].scatter(x_sample, y_sample, alpha=0.6, s=1)
    axes[0].set_title('Scatter Plot (Optimized)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    
    # 헥스빈 플롯 (밀도 기반)
    axes[1].hexbin(x_sample, y_sample, gridsize=50, cmap='Blues')
    axes[1].set_title('Hexbin Plot (Density)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

# 대용량 데이터 생성 및 테스트
np.random.seed(42)
large_x = np.random.randn(100000)
large_y = 2 * large_x + np.random.randn(100000) * 0.5

optimize_large_data_plot(large_x, large_y)
```

### 5-3. 보고서용 시각화

```python
# 보고서용 고품질 시각화
def create_publication_plot():
    """출판용 고품질 그래프 생성"""
    
    # 고해상도 설정
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 데이터 생성
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 고품질 그래프
    ax.plot(x, y1, label='sin(x)', linewidth=2, color='#1f77b4')
    ax.plot(x, y2, label='cos(x)', linewidth=2, color='#ff7f0e')
    
    # 스타일링
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax.set_title('Publication Quality Plot', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 축 스타일링
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 고해상도로 저장
    plt.savefig('publication_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("고품질 그래프가 'publication_plot.png'로 저장되었습니다.")

create_publication_plot()
```

## 6. 시각화 원칙과 모범 사례

### 6-1. 시각화 원칙

```python
print("=== 시각화 원칙 ===")

print("1. 명확성 (Clarity)")
print("   - 데이터를 명확하게 전달")
print("   - 불필요한 요소 제거")
print("   - 적절한 축 라벨과 제목")

print("\n2. 간결성 (Simplicity)")
print("   - 복잡한 정보를 단순하게 표현")
print("   - 과도한 장식 피하기")
print("   - 핵심 메시지에 집중")

print("\n3. 일관성 (Consistency)")
print("   - 통일된 색상 팔레트")
print("   - 일관된 스타일 적용")
print("   - 표준화된 레이아웃")

print("\n4. 정확성 (Accuracy)")
print("   - 데이터 왜곡 방지")
print("   - 적절한 축 스케일")
print("   - 통계적 유의성 고려")
```

### 6-2. 색상 활용 가이드

```python
# 색상 팔레트 예제
def demonstrate_color_palettes():
    """색상 팔레트 시연"""
    
    # 다양한 색상 팔레트
    palettes = {
        'Sequential': ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac'],
        'Diverging': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9641'],
        'Qualitative': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for i, (palette_name, colors) in enumerate(palettes.items()):
        # 색상 팔레트 시각화
        for j, color in enumerate(colors):
            axes[i].barh(0, 1, left=j, height=0.8, color=color, edgecolor='white', linewidth=2)
        
        axes[i].set_xlim(0, len(colors))
        axes[i].set_ylim(-0.5, 0.5)
        axes[i].set_title(f'{palette_name} Palette', fontsize=14, fontweight='bold')
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

demonstrate_color_palettes()
```

## 마무리

Matplotlib 고급 시각화를 통해 데이터 분석의 전문성을 높일 수 있습니다:

### 핵심 학습 내용
- **시각화 라이브러리**: Matplotlib, Seaborn, Plotly의 특성과 활용
- **고급 기법**: 분포, 관계, 시간, 비율 시각화
- **스타일링**: 한글 폰트, 테마, 커스텀 스타일
- **성능 최적화**: 대용량 데이터 처리와 메모리 관리

### 실무 적용
- **분석 과정**: 빠른 데이터 탐색과 패턴 발견
- **보고서 작성**: 고품질 시각화와 출판용 그래프
- **시각화 원칙**: 명확성, 간결성, 일관성, 정확성
- **색상 활용**: 색맹 고려와 의미 있는 색상 선택

Matplotlib는 데이터 분석의 핵심 도구로, 적절한 시각화를 통해 데이터의 인사이트를 효과적으로 전달할 수 있습니다.
