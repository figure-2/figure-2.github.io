---
title: 🎭 GANs (Generative Adversarial Networks) - 적대적 생성 신경망의 이해
date: 2025-07-17 12:58:00 +0900
categories: 
tags:
  - 급발진거북이
toc: true
comments: false
mermaid: true
math: true
---
## 📦 사용하는 python package

- torch==2.0.1
- torchvision==0.15.2
- matplotlib==3.7.1
- numpy==1.24.3
- PIL==9.5.0

## 🚀 TL;DR

> **GANs는 두 개의 신경망이 적대적으로 경쟁하며 학습하는 생성 모델**로, 위조지폐범(Generator)과 경찰(Discriminator)의 경쟁 구조와 같다. Generator는 실제와 구분할 수 없는 가짜 데이터를 생성하려 하고, Discriminator는 진짜와 가짜를 구별하려 한다. 이 과정에서 **VAE보다 선명한 이미지**를 생성할 수 있지만 **mode collapse**, **학습 불안정성** 등의 문제가 있다. 하지만 **설명 가능성과 다양한 응용** 가능성으로 인해 현재까지도 널리 활용되는 핵심 생성 모델 기술이다.

## 📓 실습 Jupyter Notebook

- [GANs Implementation and Training](https://github.com/yuiyeong/notebooks/blob/main/deep_learning/gans_tutorial.ipynb)

## 🎭 GANs(Generative Adversarial Networks)란?

**Generative Adversarial Networks(GANs)**는 한국어로 **적대적 생성 신경망**이라고 불리며, 두 개의 신경망이 서로 경쟁하며 학습하는 생성 모델이다.

**Generative**(생성), **Adversarial**(적대적), **Networks**(신경망)의 조합으로, 신경망들이 적대적으로 구성되어 데이터를 생성하는 프레임워크를 의미한다.

> GANs는 얀 르쿤(Yann LeCun)이 "지난 10년간 머신러닝에서 가장 흥미로운 아이디어"라고 평가할 정도로 혁신적인 기술이다. {: .prompt-tip}

### GANs vs VAE: 접근 방식의 차이

**VAE(Variational Autoencoder)**와 **GANs**는 모두 생성 모델이지만 근본적으로 다른 접근 방식을 취한다.

- **VAE**: 입력 분포를 직접 근사하는 과정에서 정규화(regularization)를 통해 데이터 생성 방법을 학습
- **GANs**: 분포를 직접 추정하지 않고, 한 모델이 다른 모델을 가이드하는 방식으로 학습

이러한 차이로 인해 GANs는 **복잡한 목적 함수 정의가 불필요**하고 **구조상 트릭이 필요하지 않다**는 장점을 가진다.

## 🏗️ GANs의 구조: Generator와 Discriminator

GANs는 두 개의 신경망으로 구성된다.

### Generator (생성자)

[시각적 표현 넣기: Generator 구조도 - 노이즈 벡터에서 이미지 생성까지의 과정]

**Generator**는 **위조지폐범**과 같은 역할을 한다. 실제 돈을 직접 본 적은 없지만, 경찰(Discriminator)의 반응만 보고 어떻게 더 정교한 위조지폐를 만들지 학습한다.

- **입력**: 가우시안 분포나 균등 분포에서 샘플링한 노이즈 벡터
- **출력**: 실제 데이터와 유사한 가짜 데이터 (예: 이미지)
- **구조**: Autoencoder의 Decoder와 유사한 구조

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# 사용 예시
latent_dim = 100
generator = Generator(latent_dim)

# 노이즈 벡터 생성
z = torch.randn(64, latent_dim)  # 배치 크기 64
fake_images = generator(z)
print(f"생성된 이미지 shape: {fake_images.shape}")  # torch.Size([64, 1, 28, 28])
```

### Discriminator (판별자)

[시각적 표현 넣기: Discriminator 구조도 - 이미지를 받아 진짜/가짜 판별하는 과정]

**Discriminator**는 **경찰**과 같은 역할을 한다. Generator가 생성한 가짜 데이터와 실제 데이터를 구분하는 이진 분류 문제를 해결한다.

- **입력**: 실제 데이터 또는 Generator가 생성한 가짜 데이터
- **출력**: 입력이 실제 데이터일 확률 (0~1 사이의 값)
- **구조**: 일반적인 분류기와 동일한 구조

```python
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 사용 예시
discriminator = Discriminator()

# 실제 이미지와 가짜 이미지 판별
real_images = torch.randn(64, 1, 28, 28)
fake_images = generator(z)

real_validity = discriminator(real_images)
fake_validity = discriminator(fake_images)

print(f"실제 이미지 판별 결과: {real_validity.mean().item():.4f}")  # 1에 가까울수록 진짜로 판별
print(f"가짜 이미지 판별 결과: {fake_validity.mean().item():.4f}")  # 0에 가까울수록 가짜로 판별
```

## 📊 GANs의 학습 원리

[시각적 표현 넣기: GANs 학습 과정 시각화 - 초록색(Generator 분포), 파란색(Discriminator 경계), 검은색(실제 데이터 분포)]

GANs의 학습은 **두 플레이어 제로섬 게임**의 형태로 진행된다. Generator는 분포를 업데이트하고, Discriminator는 진짜와 가짜를 구별하는 경계를 업데이트한다.

### 학습 과정의 동역학

1. **초기 상태**: Generator는 랜덤한 분포, Discriminator는 부정확한 판별
2. **Generator 개선**: Discriminator를 속이기 위해 더 실제같은 데이터 생성
3. **Discriminator 개선**: 더 정교한 판별 능력 획득
4. **균형점 도달**: Generator가 실제 분포를 완벽히 모사

> Generator는 실제 데이터를 직접 관찰하지 않고 Discriminator의 피드백만을 활용하기 때문에, **Discriminator의 품질이 GANs 성능을 좌우한다**. {: .prompt-tip}

## 🎯 GANs의 목적 함수

GANs의 목적 함수는 **minimax 게임**의 형태로 정의된다.

### 수학적 표현

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

여기서:

- **D(x)**: Discriminator가 실제 데이터 x에 대해 출력하는 확률
- **G(z)**: Generator가 노이즈 z로부터 생성한 데이터
- **첫 번째 항**: 실제 데이터를 진짜로 판별하는 것을 최대화
- **두 번째 항**: 가짜 데이터를 가짜로 판별하는 것을 최대화

### Discriminator의 목적

Discriminator는 위 목적 함수를 **최대화**한다.

- **실제 데이터에 대해**: $\log D(x)$를 최대화 → D(x) = 1로 만들려 함
- **가짜 데이터에 대해**: $\log(1 - D(G(z)))$를 최대화 → D(G(z)) = 0으로 만들려 함

### Generator의 목적

Generator는 같은 목적 함수를 **최소화**한다.

- **가짜 데이터에 대해**: $\log(1 - D(G(z)))$를 최소화 → D(G(z)) = 1로 만들려 함

### 실제 구현에서의 개선

원래 목적 함수는 학습 초기에 **기울기 소실 문제**를 일으킨다. 따라서 실제로는 다음과 같이 수정된다.

$$ \max_G \mathbb{E}_{z \sim p_z(z)}[\log D(G(z))] $$

```python
import torch.nn.functional as F

def train_step(generator, discriminator, real_images, latent_dim, device):
    batch_size = real_images.size(0)
    
    # 실제 이미지와 가짜 이미지 레이블
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    
    # ===== Discriminator 학습 =====
    discriminator.zero_grad()
    
    # 실제 이미지에 대한 손실
    real_outputs = discriminator(real_images)
    real_loss = F.binary_cross_entropy(real_outputs, real_labels)
    
    # 가짜 이미지에 대한 손실
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_images = generator(z)
    fake_outputs = discriminator(fake_images.detach())  # generator 기울기 차단
    fake_loss = F.binary_cross_entropy(fake_outputs, fake_labels)
    
    # Discriminator 전체 손실
    d_loss = real_loss + fake_loss
    d_loss.backward()
    
    # ===== Generator 학습 =====
    generator.zero_grad()
    
    # 수정된 목적 함수 사용
    fake_outputs = discriminator(fake_images)
    g_loss = F.binary_cross_entropy(fake_outputs, real_labels)  # 가짜를 진짜로 속이기
    g_loss.backward()
    
    return d_loss.item(), g_loss.item()
```

## ⚠️ GANs의 주요 문제점들

### Mode Collapse (모드 붕괴)

[시각적 표현 넣기: Mode Collapse 예시 - 다양한 숫자 대신 0만 생성하는 예시]

**Mode Collapse**는 Generator가 실제 데이터의 일부 모드(mode)만 학습하여 다양성이 떨어지는 문제다.

```python
# Mode Collapse 탐지 예시
def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    생성된 샘플들의 다양성을 측정하여 mode collapse 탐지
    """
    # 각 샘플 간의 평균 거리 계산
    distances = []
    for i in range(len(generated_samples)):
        for j in range(i+1, len(generated_samples)):
            dist = torch.norm(generated_samples[i] - generated_samples[j])
            distances.append(dist.item())
    
    avg_distance = sum(distances) / len(distances)
    
    if avg_distance < threshold:
        print("⚠️ Mode Collapse 의심! 생성된 샘플들이 너무 유사합니다.")
        return True
    else:
        print("✅ 적절한 다양성을 보입니다.")
        return False

# 사용 예시
z_samples = torch.randn(100, latent_dim)
generated_samples = generator(z_samples)
detect_mode_collapse(generated_samples)
```

> Mode Collapse는 Discriminator가 다양한 샘플을 생성하도록 충분한 피드백을 제공하지 못할 때 발생한다. 이를 해결하기 위해 다양한 정규화 기법과 목적 함수 개선이 연구되고 있다. {: .prompt-warning}

### 학습 불안정성

GANs는 두 네트워크의 균형이 중요하기 때문에 학습이 불안정할 수 있다.

```python
def balanced_training(generator, discriminator, d_optimizer, g_optimizer, 
                     real_images, latent_dim, k=1):
    """
    균형잡힌 GANs 학습
    k: Discriminator를 몇 번 더 학습시킬지 결정
    """
    
    # Discriminator를 k번 학습
    for _ in range(k):
        d_loss, _ = train_step(generator, discriminator, real_images, latent_dim, device)
        d_optimizer.step()
    
    # Generator를 1번 학습
    _, g_loss = train_step(generator, discriminator, real_images, latent_dim, device)
    g_optimizer.step()
    
    return d_loss, g_loss
```

> 실제로는 K=1을 주로 사용한다. Discriminator가 너무 앞서가면 Generator에게 유의미한 기울기를 제공할 수 없기 때문이다. {: .prompt-tip}

## 🚀 GANs의 개선 방향과 변형들

### 다양한 목적 함수

기본 GANs는 **Jensen-Shannon Divergence**를 최적화하는 것과 동치다. 이를 다른 거리 측도로 바꾸어 개선할 수 있다.

- **WGAN**: Wasserstein Distance 사용
- **LSGAN**: Least Squares Loss 사용
- **f-GAN**: 일반화된 f-divergence 프레임워크

### 구조적 개선

- **DCGAN**: Convolutional layers를 활용한 안정적 학습
- **Progressive GAN**: 점진적으로 해상도를 증가시키는 학습
- **StyleGAN**: 스타일을 제어할 수 있는 고품질 이미지 생성

## 🎨 GANs vs VAE: 결과물 비교

[시각적 표현 넣기: GANs와 VAE 생성 결과 비교 이미지]

### GANs의 특징

- **장점**: 선명하고 뚜렷한 이미지 생성
- **단점**: 때로는 비현실적인 왜곡이나 아티팩트 포함

### VAE의 특징

- **장점**: 부드럽고 안정적인 이미지, 잠재 공간의 연속성
- **단점**: 상대적으로 흐릿한 결과물

```python
# 두 모델의 결과 비교 코드
def compare_models(vae, gan_generator, test_input):
    """
    VAE와 GAN의 생성 결과를 비교
    """
    with torch.no_grad():
        # VAE 결과
        vae_output, _, _ = vae(test_input)
        
        # GAN 결과  
        z = torch.randn(test_input.size(0), latent_dim)
        gan_output = gan_generator(z)
        
        # 시각화 코드
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        for i in range(4):
            # VAE 결과
            axes[0, i].imshow(vae_output[i].squeeze(), cmap='gray')
            axes[0, i].set_title('VAE')
            axes[0, i].axis('off')
            
            # GAN 결과
            axes[1, i].imshow(gan_output[i].squeeze(), cmap='gray')
            axes[1, i].set_title('GAN')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# 사용 예시
# compare_models(vae_model, generator, test_images)
```

## 🌟 GANs의 실제 활용 사례

### 이미지 생성 및 편집

- **DeepFake**: 얼굴 바꾸기 기술
- **Style Transfer**: 예술 작품 스타일 변환
- **Super Resolution**: 저해상도 이미지를 고해상도로 변환

### 데이터 증강

- **의료 영상**: 희귀 질병 데이터 생성으로 학습 데이터 부족 해결
- **자율주행**: 다양한 주행 시나리오 데이터 생성

### 창작 지원

- **게임 산업**: 게임 내 캐릭터, 배경 자동 생성
- **패션**: 새로운 디자인 패턴 생성
- **음악**: 새로운 멜로디 생성

## 💡 GANs 구현 시 실무 팁

### 학습 안정화 기법

```python
# 학습률 스케줄링
def get_lr_scheduler(optimizer, step_size=50, gamma=0.5):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# 가중치 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 모델에 적용
generator.apply(weights_init)
discriminator.apply(weights_init)
```

### 학습 모니터링

```python
def monitor_training(d_losses, g_losses, epoch):
    """
    학습 과정 모니터링
    """
    # 손실 기록
    print(f"Epoch {epoch}: D_loss={d_losses[-1]:.4f}, G_loss={g_losses[-1]:.4f}")
    
    # 손실 균형 체크
    if len(d_losses) > 10:
        recent_d_loss = sum(d_losses[-10:]) / 10
        recent_g_loss = sum(g_losses[-10:]) / 10
        
        if recent_d_loss < 0.1:
            print("⚠️ Discriminator가 너무 강함 - Generator 학습률 증가 고려")
        elif recent_g_loss < 0.1:
            print("⚠️ Generator가 너무 강함 - Discriminator 학습률 증가 고려")
```

## 🔮 GANs의 미래와 최신 동향

GANs는 현재까지도 활발히 연구되고 있으며, **Diffusion Model**과 함께 사용되어 품질과 속도 측면에서 더욱 발전하고 있다.

### 최신 연구 방향

- **조건부 생성**: 특정 조건을 만족하는 데이터 생성
- **다중 모달**: 텍스트-이미지, 오디오-비디오 등 다양한 모달리티 연결
- **효율성 개선**: 적은 데이터로도 고품질 결과 생성

> GANs는 단순히 이미지 생성을 넘어서 **창의적 AI**, **데이터 프라이버시**, **시뮬레이션** 등 다양한 분야에서 핵심 기술로 자리잡고 있다. {: .prompt-tip}

GANs는 생성 모델의 패러다임을 바꾼 혁신적인 기술로, 이론적 이해와 함께 실제 구현을 통해 그 가능성을 체험해볼 수 있다. 비록 학습의 어려움이 있지만, 적절한 기법들을 활용하면 놀라운 결과를 얻을 수 있는 강력한 도구다.