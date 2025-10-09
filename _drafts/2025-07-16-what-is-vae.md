---
title: "🎨 VAE(Variational Autoencoder): 확률적 생성 모델의 시작"
date: 2025-07-16 21:36:00 +0900
categories: 
tags:
  - 급발진거북이
toc: true
comments: false
mermaid: true
math: true
---
## 📦 사용하는 python package

- torch==2.0+
- numpy==1.24+
- matplotlib==3.7+
- torchvision==0.15+
- scikit-learn==1.3+

## 🚀 TL;DR

- **VAE**는 **오토인코더**에 **확률론적 관점**을 추가하여 **새로운 데이터를 생성**할 수 있는 모델
- 잠재변수가 **표준 정규분포**를 따르도록 학습하여 **쉽게 샘플링** 가능한 생성 모델 구현
- **ELBO(Evidence Lower Bound)** 최적화를 통해 **Reconstruction Loss**와 **KL Divergence**를 동시에 학습
- **Reparameterization Trick**을 사용하여 확률적 샘플링 과정에서도 **역전파** 가능
- 생성된 이미지는 다소 **흐릿하지만** 의미있는 **잠재 표현**을 학습하여 **이미지 편집** 및 **특성 변환** 가능
- **GAN 이전** 주요 생성 모델로서 **안정적인 학습**과 **해석 가능한 잠재공간** 제공

## 📓 실습 Jupyter Notebook

- [VAE 구현 및 실습](https://github.com/yuiyeong/notebooks/blob/main/deep_learning/vae_implementation.ipynb)

## 🔄 오토인코더에서 VAE로의 진화

### 기존 오토인코더의 한계

기존 **오토인코더(Autoencoder)**는 입력 데이터를 압축된 **잠재 표현(Latent Representation)**으로 변환한 후 다시 원본으로 복원하는 구조다. 하지만 치명적인 한계가 있었다.

- **생성 불가능**: 잠재공간에서 임의의 점을 샘플링해도 의미있는 이미지가 생성되지 않음
- **불연속적 잠재공간**: 학습된 잠재 표현들 사이의 보간이 자연스럽지 않음
- **구조적 제약**: 단순한 압축-복원 구조로 인한 표현력 한계

### VAE의 핵심 아이디어

**VAE(Variational Autoencoder)**는 이러한 한계를 **확률론적 관점**으로 해결했다.

> VAE의 핵심은 **"잠재변수가 특정 확률분포(보통 표준 정규분포)를 따르도록 강제"**하는 것이다. 이를 통해 우리는 알고 있는 분포에서 샘플링하여 새로운 데이터를 생성할 수 있게 된다. {: .prompt-tip}

## 🎭 물 끓이기 비유: VAE의 직관적 이해

### 측정할 수 없는 깊은 곳의 온도

VAE의 학습 원리를 이해하기 위해 **물 끓이기 비유**를 사용해보자.

[시각적 표현 넣기: 냄비에서 물을 끓이는 모습, 온도계가 바닥까지 닿지 않는 상황]

우리가 물을 끓이고 있는데, **온도계가 작아서 바닥까지 측정할 수 없다**고 가정하자. 하지만 **바닥이 70도 이상이 되면 불을 끄고 싶다**.

- **목표**: 바닥 온도가 70도 이상인지 확인 (직접 측정 불가)
- **방법**: 불에서 멀리 떨어진 **상층부 온도**를 측정
- **논리**: 상층부가 70도라면, 불에 가까운 바닥은 확실히 70도 이상일 것

### VAE에서의 적용

- **바닥 온도** = 실제 데이터 분포의 가능도 $$P(X)$$ (직접 계산 불가)
- **상층부 온도** = **ELBO**(Evidence Lower Bound) (계산 가능)
- **목표**: ELBO를 최대화하여 간접적으로 실제 가능도 향상

## 🧮 VAE의 수학적 원리

### 확률론적 관점에서의 문제 정의

VAE는 **관측된 데이터 X를 가장 잘 설명하는 모델**을 만들고자 한다. 이는 **가능도 $$P(X)$$를 최대화**하는 것과 같다.

하지만 $$P(X)$$를 직접 계산하는 것은 매우 어렵다. 대신 우리는 이를 **세 개의 항으로 분해**할 수 있다:

$$ \log P(X) = E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z)) + D_{KL}(q_\phi(z|x) \parallel p_\theta(z|x)) $$

여기서:

- **첫 번째 항**: **재구성 항** (Reconstruction Term)
- **두 번째 항**: **정규화 항** (Regularization Term)
- **세 번째 항**: 계산 불가능한 항

### ELBO(Evidence Lower Bound)

세 번째 항은 **KL Divergence**이므로 항상 **0 이상**이다. 따라서:

$$ \log P(X) \geq E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z)) $$

우변을 **ELBO(Evidence Lower Bound)**라고 하며, 이것이 우리가 최대화할 목적함수다.

$$ \mathcal{L}_{ELBO} = E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z)) $$

### 두 가지 손실함수

#### 1. 재구성 손실(Reconstruction Loss)

$$ \mathcal{L}_{recon} = -E_{q_\phi(z|x)}[\log p_\theta(x|z)] $$

이는 **인코더로 얻은 잠재변수**를 **디코더로 복원**했을 때 **원본과의 차이**를 측정한다.

```python
# 실제 구현에서는 MSE나 BCE 사용
reconstruction_loss = F.mse_loss(reconstructed_x, original_x)
# 또는
reconstruction_loss = F.binary_cross_entropy(reconstructed_x, original_x)
```

#### 2. KL 발산 손실(KL Divergence Loss)

$$ \mathcal{L}_{KL} = D_{KL}(q_\phi(z|x) \parallel p(z)) $$

이는 **학습된 잠재분포**가 **표준 정규분포**에 가까워지도록 강제한다.

표준 정규분포를 prior로 사용할 때, **닫힌 형태의 해**가 존재한다:

$$ D_{KL}(q_\phi(z|x) \parallel \mathcal{N}(0,I)) = \frac{1}{2}\sum_{j=1}^{J}(1 + \log((\sigma_j)^2) - (\mu_j)^2 - (\sigma_j)^2) $$

## ⚡ Reparameterization Trick

### 문제: 미분 불가능한 샘플링

VAE 학습의 핵심 문제는 **확률적 샘플링 과정에서 미분이 불가능**하다는 점이다.

[시각적 표현 넣기: 일반적인 샘플링 vs Reparameterization 과정 도식화]

### 해결책: 재매개변수화

**정규분포의 성질**을 이용하여 문제를 해결한다:

$$ z \sim \mathcal{N}(\mu, \sigma^2) \Leftrightarrow z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

- **기존 방식**: $$z$$를 $$\mathcal{N}(\mu, \sigma^2)$$에서 직접 샘플링 (미분 불가)
- **새로운 방식**: $$\epsilon$$을 $$\mathcal{N}(0, I)$$에서 샘플링하고 변환 (미분 가능)

```python
def reparameterize(mu, log_var):
    """Reparameterization trick"""
    std = torch.exp(0.5 * log_var)  # 표준편차 계산
    eps = torch.randn_like(std)     # 표준 정규분포에서 샘플링
    z = mu + eps * std              # 재매개변수화
    return z
```

## 🏗️ VAE 구현

### 전체 VAE 모델 구조

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # 인코더
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)     # 평균
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # 로그 분산
        
        # 디코더
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """인코더: 입력을 평균과 로그분산으로 변환"""
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        log_var = self.fc_logvar(h1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """재매개변수화 트릭"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """디코더: 잠재변수를 이미지로 변환"""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# 손실함수 정의
def vae_loss(recon_x, x, mu, log_var):
    """VAE 손실함수: 재구성 손실 + KL 발산"""
    # 재구성 손실 (BCE)
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL 발산 손실
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss

# 학습 루프 예시
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 순전파
        recon_batch, mu, log_var = model(data)
        
        # 손실 계산
        loss = vae_loss(recon_batch, data, mu, log_var)
        
        # 역전파
        loss.backward()
        optimizer.step()
```

### 새로운 이미지 생성

```python
def generate_images(model, num_samples=16, latent_dim=20):
    """표준 정규분포에서 샘플링하여 새로운 이미지 생성"""
    model.eval()
    with torch.no_grad():
        # 표준 정규분포에서 샘플링
        z = torch.randn(num_samples, latent_dim)
        
        # 디코더를 통해 이미지 생성
        generated = model.decode(z)
        
        return generated.view(-1, 28, 28)  # MNIST 형태로 reshape

# 사용 예시
generated_images = generate_images(model)
print(f"생성된 이미지 shape: {generated_images.shape}")
# 출력: 생성된 이미지 shape: torch.Size([16, 28, 28])
```

## 🎨 VAE의 활용 사례

### 1. 잠재공간 보간(Latent Space Interpolation)

```python
def interpolate_images(model, img1, img2, steps=10):
    """두 이미지 사이의 잠재공간 보간"""
    model.eval()
    with torch.no_grad():
        # 두 이미지를 잠재공간으로 인코딩
        mu1, _ = model.encode(img1.view(-1, 784))
        mu2, _ = model.encode(img2.view(-1, 784))
        
        # 선형 보간
        interpolated = []
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            img_interp = model.decode(z_interp)
            interpolated.append(img_interp.view(28, 28))
        
        return interpolated

# 숫자 '0'에서 '1'로 부드럽게 변화하는 이미지 시퀀스 생성
interpolated = interpolate_images(model, digit_0, digit_1)
```

### 2. 의미있는 특성 편집

VAE의 **잠재공간은 의미있는 방향**을 학습한다. 예를 들어:

- **얼굴 이미지**: 나이, 성별, 머리색, 안경 착용 여부 등
- **패션 아이템**: 색상, 스타일, 길이 등

```python
def edit_attribute(model, image, direction_vector, strength=1.0):
    """특정 속성을 편집"""
    model.eval()
    with torch.no_grad():
        # 원본 이미지 인코딩
        mu, _ = model.encode(image.view(-1, 784))
        
        # 속성 방향으로 이동
        edited_z = mu + strength * direction_vector
        
        # 편집된 이미지 생성
        edited_image = model.decode(edited_z)
        
        return edited_image.view(28, 28)

# 사용 예시: 숫자를 더 굵게 만들기
thick_direction = find_thickness_direction(model, dataset)  # 사전에 찾은 방향벡터
thick_digit = edit_attribute(model, original_digit, thick_direction, strength=2.0)
```

## ⚖️ VAE의 장단점

### 장점

- **안정적인 학습**: GAN에 비해 **모드 붕괴(Mode Collapse)** 문제가 적음
- **해석 가능한 잠재공간**: 각 차원이 **의미있는 특성**을 나타내는 경우가 많음
- **확률적 프레임워크**: 불확실성을 **정량화**할 수 있음
- **다양성 보장**: 표준 정규분포에서 샘플링하므로 **다양한 출력** 생성

### 단점

- **흐릿한 이미지**: **평균 제곱 오차** 기반 재구성 손실로 인한 **과도한 평활화**
- **제한된 표현력**: **가우시안 가정**이 복잡한 분포 모델링에 한계
- **후방 붕괴(Posterior Collapse)**: KL 정규화가 너무 강하면 잠재변수가 **무시**될 수 있음

> VAE는 **"안정성과 해석가능성을 중시하는 생성 모델"**이다. 높은 품질의 이미지보다는 **의미있는 표현 학습**과 **안정적인 생성**에 초점을 맞춘다. {: .prompt-tip}

### 개선된 변형 모델들

#### β-VAE

```python
def beta_vae_loss(recon_x, x, mu, log_var, beta=4.0):
    """β-VAE: KL 발산에 가중치 부여"""
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + beta * kl_loss  # β > 1로 설정하여 더 강한 정규화
```

**β-VAE**는 **KL 발산 항에 β 가중치**를 적용하여:

- **더 독립적인 잠재 표현** 학습
- **해석가능성** 향상
- **특성별 편집** 능력 강화

## 🔮 VAE의 현재와 미래

### 현재 활용 분야

- **약물 발견**: 분자 구조 생성 및 최적화
- **이미지 편집**: 의미론적 이미지 조작
- **데이터 증강**: 제한된 데이터셋 확장
- **이상 탐지**: 정상 분포에서 벗어난 데이터 감지

### 최신 발전 방향

- **Conditional VAE**: 조건부 생성을 위한 확장
- **Hierarchical VAE**: 다층 잠재구조로 표현력 향상
- **Normalizing Flows**: 더 유연한 후방 분포 모델링
- **VQ-VAE**: 이산적 잠재표현을 통한 품질 개선

> VAE는 **생성 모델의 기초**를 제공했으며, 현재도 **안정성과 해석가능성**이 중요한 응용에서 널리 사용되고 있다. 특히 **과학적 발견**과 **창작 지원 도구**에서 그 가치가 인정받고 있다. {: .prompt-tip}